import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import scipy.integrate as integrate
import sys
import os
from ..utils.wind import Wind

sys.path.append(os.path.join("..", "..", ".."))
from parameters import params_environment, params_triangle_soaring, params_decision_maker
from subtasks.updraft_exploiter import params_updraft_exploiter
from subtasks.vertex_tracker import params_vertex_tracker


class gliderEnv3D(gym.Env):

    ########################################################################################################################

    def __init__(self, agent='vertex_tracker'):

        # instantiate parameters
        self._params_glider = params_environment.params_glider()
        self._params_physics = params_environment.params_physics()
        self._params_sim = params_environment.params_sim()
        self._params_wind = params_environment.params_wind()
        self._params_task = params_triangle_soaring.TaskParameters()

        if agent == 'vertex_tracker':
            self.agent = agent
            self._params_agent = params_vertex_tracker.params_agent()
            self.current_task = self._params_task.TASK
        elif agent == 'updraft_exploiter':
            self.agent = agent
            self._params_agent = params_updraft_exploiter.params_agent()
            self.current_task = 'exploitation'
        elif agent == 'decision_maker':
            self.agent = agent
            self._params_agent = params_decision_maker.AgentParameters()
            self.current_task = self._params_task.TASK
        else:
            sys.exit("not a valid agent passed for env setup")

        # set wind function
        self._wind_fun = Wind()

        # set integrator
        if self._params_sim.USE_RK45:
            self._integrator = integrate.ode(self.buildDynamic3D).set_integrator('dopri5', rtol=1e-2, atol=1e-4)
        else:
            self._integrator = 'euler'

        # set random seed
        self.seed()

        # initialize further member variables
        self.lb = np.min(self._params_agent.ACTION_SPACE, 1) * (np.pi / 180)
        self.ub = np.max(self._params_agent.ACTION_SPACE, 1) * (np.pi / 180)
        self.state = None
        self.time = None
        self.control = None
        self.active_vertex = None
        self.vertex_counter = None
        self.lap_counter = None
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    ########################################################################################################################

    def reset(self):
        if self.agent == 'vertex_tracker':
            initState = self.np_random.uniform(self._params_agent.INITIAL_SPACE[:, 0],
                                               self._params_agent.INITIAL_SPACE[:, 1])
            self.active_vertex = self.np_random.randint(1, 4)
            self.time = 0
        elif self.agent == 'updraft_exploiter':
            initState = self.np_random.uniform(self._params_agent.INITIAL_SPACE[:, 0],
                                               self._params_agent.INITIAL_SPACE[:, 1])
            self.active_vertex = self.np_random.randint(1, 4)  # should not matter
            self.time = 0
        elif self.agent == 'decision_maker':
            initState = self.np_random.multivariate_normal(self._params_agent.INITIAL_STATE,
                                                           np.diag(np.square(self._params_agent.INITIAL_STD)))
            self.time = 0
            self.active_vertex = 1
        else:
            sys.exit("not a valid agent passed for env setup")

        if self._params_wind.ALWAYS_RESET:
            self._wind_fun.reset_wind()

        self.vertex_counter = 0
        self.lap_counter = 0
        self.state = np.copy(initState)
        return self.state

    ########################################################################################################################

    def step(self, action, timestep=None):
        timestep = self._params_agent.TIMESTEP_CTRL if not timestep else timestep

        self.control = self.action2control(action)
        self.integrate(timestep)
        observation = self.get_observation()
        reward, done = self.get_reward_and_done(timestep)
        info = self.getInfo()
        return observation, reward, done, info

    ########################################################################################################################

    def action2control(self, action):
        control = self.lb + (action + 1.) * 0.5 * (self.ub - self.lb)
        control = np.clip(control, self.lb, self.ub)
        return control

    ########################################################################################################################

    def integrate(self, timestep):
        if self._integrator == 'euler':
            t0 = self.time
            while self.time < (t0 + timestep):
                x_dot = self.buildDynamic3D(self.time, self.state)
                dt = np.minimum((t0 + timestep) - self.time, self._params_sim.TIMESTEP_SIM)
                self.state += (dt * x_dot)
                self.time += dt
        else:
            r = self._integrator
            r.set_initial_value(self.state)
            r.integrate(timestep)
            self.time += r.t
            self.state = r.y

    ########################################################################################################################

    def buildDynamic3D(self, t, x):

        # control variables assignment
        mu_a = self.control.item(0)
        alpha = self.control.item(1)

        # get wind vector at current aircraft position
        g_v_W = self._wind_fun.get_current_wind(x[0:3])

        # track speed in local NED coordinates
        g_v_K = x[3:6].reshape(3, 1)

        # airspeed in local NED coordinates: airspeed = groundspeed - windspeed
        g_v_A = g_v_K - g_v_W

        # air-path angles
        v_A_norm = np.maximum(np.linalg.norm(g_v_A), .1)
        gamma_a = -np.arcsin(np.clip((g_v_A[2] / v_A_norm), -1, 1))
        chi_a = np.arctan2(g_v_A[1], g_v_A[0])

        # aerodynamic force in aerodynamic coordinates
        cl = 2 * np.pi * (self._params_glider.ST / (self._params_glider.ST + 2)) * alpha
        cd = self._params_glider.CD0 + (1 / (np.pi * self._params_glider.ST * self._params_glider.OE)) * np.power(cl, 2)
        a_f_A = (self._params_physics.RHO / 2) * self._params_glider.S * np.power(v_A_norm, 2) * np.array(
            [[-cd], [0], [-cl]])

        # aerodynamic force in local NED coordinates
        g_T_a = self.getRotationMatrix(-chi_a.item(), 3) \
                @ self.getRotationMatrix(-gamma_a.item(), 2) \
                @ self.getRotationMatrix(-mu_a, 1)
        g_f_A = g_T_a @ a_f_A

        # track acceleration in local NED coordinates
        g_a_K = (g_f_A / self._params_glider.M) + np.array([[0], [0], [self._params_physics.G]])

        # state derivative
        xp = np.append(g_v_K, g_a_K)

        if np.isnan(xp).any():
            print("xp is not a number: {}".format(xp))
        return xp

    def getRotationMatrix(self, angle, axis):

        if axis == 1:
            rotationMatrix = np.array([[1, 0, 0],
                                       [0, np.cos(angle), np.sin(angle)],
                                       [0, -np.sin(angle), np.cos(angle)]])
        elif axis == 2:
            rotationMatrix = np.array([[np.cos(angle), 0, -np.sin(angle)],
                                       [0, 1, 0],
                                       [np.sin(angle), 0, np.cos(angle)]])
        elif axis == 3:
            rotationMatrix = np.array([[np.cos(angle), np.sin(angle), 0],
                                       [-np.sin(angle), np.cos(angle), 0],
                                       [0, 0, 1]])
        else:
            sys.exit("not a valid rotation axis")

        return rotationMatrix

    ########################################################################################################################

    def get_observation(self):
        if self.agent == 'vertex_tracker':
            observation = self.get_full_observation()
        elif self.agent == 'updraft_exploiter':
            observation = self.get_updraft_positions()
        elif self.agent == 'decision_maker':
            observation = self.get_sparse_observation()
        else:
            sys.exit("not a valid agent passed for env setup")

        return observation

    def get_full_observation(self):
        # vector from previous vertex to active vertex in g-coordinates (-> r-frame x-axis = g_ref/norm(g_ref))
        previous_vertex = np.mod((self.active_vertex - 1), 3)
        g_previous2active = self._params_task.TRIANGLE[:, (self.active_vertex - 1)] \
                            - self._params_task.TRIANGLE[:, (previous_vertex - 1)]

        # polar angle of r-frame wrt g-frame
        g_phi_r = np.arctan2(g_previous2active[1], g_previous2active[0]).item()

        # rotation matrix from local NED-coordinates to reference coordinates
        r_T_g = self.getRotationMatrix(g_phi_r, 3)

        # vector from origin of r-frame (i.e., previous vertex) to aircraft in g-coordinates
        g_previous2aircraft = self.state[0:2] - self._params_task.TRIANGLE[:, (previous_vertex - 1)]

        # aircraft position in cartesian r-coordinates
        r_p = r_T_g @ np.append(g_previous2aircraft, self.state[2]).reshape(3, 1)

        # polar angle of aircraft position wrt x_r-axis
        r_phi = np.arctan2(r_p[1], r_p[0]).item()

        # vector from active vertex to aircraft in g-coordinates
        g_active2aircraft = self.state[0:2] - self._params_task.TRIANGLE[:, (self.active_vertex - 1)]

        # aircraft distance from active vertex
        dist2active = np.linalg.norm(g_active2aircraft)

        # tack speed in r-coordinates
        g_v_K = self.state[3:6].reshape(3, 1)
        r_v_K = r_T_g @ g_v_K

        # norm, azimuth wrt x_r-axis, and flight path angle of track speed vector
        V_K = np.linalg.norm(r_v_K)
        r_chi = np.arctan2(r_v_K[1], r_v_K[0]).item()
        gamma = -np.arcsin(r_v_K[2] / V_K).item()

        # stack observation and normalize:
        # o = [distance to next vertex, polar angle wrt x_r, height, v_K in polar coords. wrt r-frame]
        observation = (np.array([dist2active, r_phi, -self.state[2],
                                 V_K, r_chi, gamma])
                       - self._params_agent.OBS_MEAN) / self._params_agent.OBS_STD

        return observation

    def get_sparse_observation(self):
        # vector from active vertex to aircraft in g-coordinates
        g_active_to_aircraft = self.state[0:2] - self._params_task.TRIANGLE[:, (self.active_vertex - 1)]

        # aircraft distance from active vertex
        dist_to_active_vertex = np.linalg.norm(g_active_to_aircraft)

        # triangle dimensions
        len_base = np.linalg.norm(self._params_task.TRIANGLE[:, 0] - self._params_task.TRIANGLE[:, 2])
        len_legs = len_base / np.sqrt(2)

        # distance to finish line
        if self.vertex_counter != 3:
            dist_to_finish = dist_to_active_vertex + (3 - self.active_vertex) * len_legs + 0.5 * len_base
        else:
            T_pos_ac = np.transpose(self._params_task.G_T_T) @ self.state[0:2].reshape(2, 1)
            dist_to_finish = T_pos_ac[1].item() - self._params_task.FINISH_LINE[1].item()

        # # relativ position of closest updraft
        # _, rel_updraft_pos_sorted = self.get_rel_updraft_positions()
        # closest_rel_updraft_pos = rel_updraft_pos_sorted[-1, :]
        # r_chi = self.get_azimuth_wrt_r_frame()
        # dist_to_closest_up = rel_updraft_pos_sorted[-1, 0]
        # dir_to_closest_up = self.wrap2pi(closest_rel_updraft_pos[1] + r_chi)

        # stack observation and normalize:
        # o = [time, height, d_finish_line]
        observation = (np.array([self.time, -self.state[2], dist_to_finish])
                       - self._params_agent.OBS_MEAN) / self._params_agent.OBS_STD
        return observation

    def get_updraft_positions(self):
        # assign updraft data
        updraft_count = int(self._wind_fun._wind_data['updraft_count'])
        updraft_position = self._wind_fun._wind_data['updraft_position']

        # horizontal track speed in local NE coordinates
        g_v_K = self.state[3:5].reshape(2, 1)

        # aircraft heading wrt g-frame
        g_chi = np.arctan2(g_v_K[1], g_v_K[0]).item()

        # rotation matrix from local NED-coordinates to k-coordinates
        k_T_g = self.getRotationMatrix(g_chi, 3)

        # set relative updraft positions (dist, dir)
        rel_updraft_pos = np.empty([updraft_count, 2])

        for k in range(0, updraft_count):
            # vector from aircraft to updraft in g-coordinates (i.e., line-of-sight)
            g_aircraft2updraft = updraft_position[:, k].reshape(2, 1) - self.state[0:2].reshape(2, 1)

            # updraft position in cartesian k-coordinates
            k_p = k_T_g[0:2, 0:2] @ g_aircraft2updraft

            # (negative) aircraft heading wrt. line-of-sight to updraft
            k_phi = np.arctan2(k_p[1], k_p[0]).item()

            # assign return values
            rel_updraft_pos[k, :] = np.array([np.linalg.norm(g_aircraft2updraft), k_phi])

        # sort the array in descending order wrt updraft distance (nearest updraft in last column)
        rel_updraft_pos_sorted = rel_updraft_pos[np.argsort(-rel_updraft_pos[:, 0]), :]

        # standardization
        rel_updraft_pos_normalized = (rel_updraft_pos_sorted[:, 0] - self._params_wind.UPD_MEAN) / \
                                     self._params_wind.UPD_STD
        rel_updraft_pos_normalized = np.stack((rel_updraft_pos_normalized, rel_updraft_pos_sorted[:, 1] / np.pi), 1)

        return rel_updraft_pos_normalized

    def get_azimuth_wrt_r_frame(self):
        previous_vertex = np.mod((self.active_vertex - 1), 3)
        g_previous2active = self._params_task.TRIANGLE[:, (self.active_vertex - 1)] \
                            - self._params_task.TRIANGLE[:, (previous_vertex - 1)]

        # polar angle of r-frame wrt g-frame
        g_phi_r = np.arctan2(g_previous2active[1], g_previous2active[0]).item()

        # rotation matrix from local NED-coordinates to reference coordinates
        r_T_g = self.getRotationMatrix(g_phi_r, 3)

        # tack speed in r-coordinates
        g_v_K = self.state[3:6].reshape(3, 1)
        r_v_K = r_T_g @ g_v_K

        # norm, azimuth wrt x_r-axis, and flight path angle of track speed vector
        r_chi = np.arctan2(r_v_K[1], r_v_K[0]).item()

        return r_chi

    def wrap2pi(self, angle):
        """
        This funtion wraps an input to the interval [-pi, pi].
        Matthi suggests: mod(angle + pi, 2*pi) - pi
        """
        if angle > np.pi:
            wrappend_angle = angle - (2 * np.pi)
        elif angle < (-np.pi):
            wrappend_angle = angle + (2 * np.pi)
        else:
            wrappend_angle = angle

        return wrappend_angle

    ########################################################################################################################

    def get_reward_and_done(self, timestep):

        # set active vertex
        old_vertex = self.active_vertex
        self.set_active_vertex()

        # set lap counter
        old_lap_counter = self.lap_counter
        self.set_lap_counter()

        # set flags relevant for done flag
        ground = (-self.state[2] <= 0)
        outofsight = (-self.state[2] > self._params_agent.HEIGHT_MAX)

        # assign reward
        if self.agent == 'vertex_tracker' and self.current_task == 'distance':
            reward = 200 / 3 if (self.active_vertex != old_vertex) else 0
            timeout = (self.time >= self._params_task.WORKING_TIME)
        elif self.agent == 'vertex_tracker' and self.current_task == 'speed':
            sys.exit("task not implemented, yet")
        elif self.agent == 'updraft_exploiter':
            reward = self.get_energy_reward()
            outofsight = False
            timeout = (self.time >= 400)
        elif self.agent == 'decision_maker':
            # reward = 200 / 3 if (self.active_vertex != old_vertex) else 0
            reward = 200 if (self.lap_counter > old_lap_counter) else 0
            # reward = reward - (self._params_task.WORKING_TIME - self.time) if (ground or outofsight) else reward
            timeout = (self.time >= self._params_task.WORKING_TIME)
            # reward = reward - (-self.state[2]) if timeout else reward
        else:
            sys.exit("not a valid agent passed for env setup")

        # set done flag
        done = (ground or timeout or outofsight or (not np.isfinite(self.state).all()))

        return reward, done

    def set_active_vertex(self):
        # get horizontal aircraft position in active-sector-coordinates
        sec_T_g = self.get_trafo_to_sector_coords()
        sec_pos_ac = sec_T_g @ (self.state[0:2].reshape(2, 1)
                                - self._params_task.TRIANGLE[:, (self.active_vertex - 1)].reshape(2, 1))

        # update active vertex if both active-sector-coordinates are positive
        if (sec_pos_ac >= 0).all():
            # print("hit active vertex no. {}" .format(self.active_vertex))
            self.vertex_counter += 1
            if (self.active_vertex + 1) > 3:
                self.active_vertex = 1
            else:
                self.active_vertex += 1

    def get_trafo_to_sector_coords(self):
        if self.active_vertex == 1:
            # rotation matrix from geodetic to sector-one-coordinates
            sec_T_g = (self._params_task.ONE_T_T @ np.transpose(self._params_task.G_T_T))
        elif self.active_vertex == 2:
            # rotation matrix from geodetic to sector-two-coordinates
            sec_T_g = (self._params_task.TWO_T_T @ np.transpose(self._params_task.G_T_T))
        elif self.active_vertex == 3:
            # rotation matrix from geodetic to sector-three-coordinates
            sec_T_g = (self._params_task.THREE_T_T @ np.transpose(self._params_task.G_T_T))
        else:
            sec_T_g = None
            print("active vertex no. {} is not a valid triangle vertex".format(self.active_vertex))

        return sec_T_g

    def set_lap_counter(self):
        T_pos_ac = np.transpose(self._params_task.G_T_T) @ self.state[0:2].reshape(2, 1)

        if self.vertex_counter == 3 and T_pos_ac[1] >= 0:
            self.lap_counter += 1
            self.vertex_counter = 0

    def get_energy_reward(self):

        g_v_W = self._wind_fun.get_current_wind(self.state[0:3])
        g_v_K = self.state[3:6].reshape(3, 1)
        g_v_A = g_v_K - g_v_W

        v_A_norm = np.maximum(np.linalg.norm(g_v_A), .1)
        gamma_a = -np.arcsin(np.clip((g_v_A[2] / v_A_norm), -1, 1))
        chi_a = np.arctan2(g_v_A[1], g_v_A[0])

        mu_a = self.control.item(0)
        alpha = self.control.item(1)

        cl = 2 * np.pi * (self._params_glider.ST / (self._params_glider.ST + 2)) * alpha
        cd = self._params_glider.CD0 + (1 / (np.pi * self._params_glider.ST * self._params_glider.OE)) * np.power(cl, 2)
        a_f_Ax = (self._params_physics.RHO / 2) * self._params_glider.S * np.power(v_A_norm, 2) * (-cd)

        a_T_g = self.getRotationMatrix(mu_a, 1) \
                @ self.getRotationMatrix(gamma_a.item(), 2) \
                @ self.getRotationMatrix(-chi_a.item(), 3)

        # energy-equivalent climb rate
        w = -g_v_K[2] + (1 / self._params_physics.G) * v_A_norm * ((1 / self._params_glider.M) * a_f_Ax
                                                                   + (a_T_g[0, :]
                                                                      @ np.array([[0], [0], [self._params_physics.G]])))

        # energy-equivalent climb rate, normalized by sink at best glide TODO: best endurance would be "correct"
        V_bestGlide, gamma_bestGlide = self.get_best_glide()
        w_normalized = w.item() - V_bestGlide * gamma_bestGlide

        # delta normalized energy-equivalent climb rate
        energyReward = w_normalized * self._params_agent.TIMESTEP_CTRL

        return energyReward

    def get_best_glide(self):
        alpha_bestGlide = ((self._params_glider.ST + 2)
                           * np.sqrt(self._params_glider.CD0 * self._params_glider.OE / self._params_glider.ST)) \
                          / (2 * np.sqrt(np.pi))
        cL_bestGlide = (2 * np.pi * alpha_bestGlide * self._params_glider.ST) / (self._params_glider.ST + 2)
        cD_bestGlide = self._params_glider.CD0 \
                       + (1 / (np.pi * self._params_glider.ST * self._params_glider.OE)) * np.power(cL_bestGlide, 2)

        V_bestGlide = np.sqrt((2 * self._params_glider.M * self._params_physics.G)
                              / (self._params_physics.RHO * self._params_glider.S * cL_bestGlide))
        gamma_bestGlide = -cD_bestGlide / cL_bestGlide

        return V_bestGlide, gamma_bestGlide

    def get_updraft_proximity_reward(self):
        # get distance to closest updraft
        _, rel_updraft_pos_sorted = self.get_rel_updraft_positions()
        shortest_distance = rel_updraft_pos_sorted[-1, 0]

        # compute simple if/else reward
        reward = 1 if (shortest_distance <= self._params_wind.DELTA) else 0

        # scale to max. return over episode
        reward *= self._params_sim.TIMESTEP

        return reward

    #########################################################################################################

    def getInfo(self):
        x = self.state
        info = {"t": self.time, "x": x[0], "y": x[1], "z": x[2], "u": x[3], "v": x[4], "w": x[5]}
        return info

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
