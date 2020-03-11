import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import scipy.integrate as integrate
import sys
import os
from ..utils.wind import Wind

sys.path.append(os.path.join("..", "..", ".."))
from params import *

class gliderEnv3D(gym.Env):

########################################################################################################################

    def __init__(self):

        # instantiate parameters
        self._params_glider = params_glider()
        self._params_physics = params_physics()
        self._params_task = params_task()
        self._params_sim = params_sim()
        self._params_wind = params_wind()

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
        self.lb             = np.min(self._params_task.ACTION_SPACE, 1)*(np.pi/180)
        self.ub             = np.max(self._params_task.ACTION_SPACE, 1)*(np.pi/180)
        self.state          = None
        self.time           = None
        self.control        = None
        self.activeVertex   = None
        self.vertexCounter  = None
        self.lapCounter     = None
        self.viewer         = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

########################################################################################################################

    def reset(self):
        if self._params_task.INIT_METHOD == 'random_normal':
            initState = self.np_random.multivariate_normal(self._params_task.INITIAL_STATE,
                                                           np.diag(np.square(self._params_task.INITIAL_STD)))
        elif self._params_task.INIT_METHOD == 'fix':
            initState = np.copy(self._params_task.INITIAL_STATE)

        if self._params_wind.ALWAYS_RESET:
            self._wind_fun.reset_wind()

        self.time = 0
        self.vertexCounter = 0
        self.lapCounter = 0
        self.activeVertex = 1
        self.state = np.copy(initState)
        return self.state

########################################################################################################################

    def step(self, action, timestep=None):
        timestep = self._params_sim.TIMESTEP if not timestep else timestep

        self.control = self.action2control(action)
        self.integrate(timestep)
        observation = self.state2observation()
        reward, done = self.getRewardAndDoneFlag(timestep)
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
                dt = np.minimum((t0 + timestep) - self.time, self._params_sim.TIMESTEP_INT)
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
        gamma_a = -np.arcsin(np.clip((g_v_A[2]/v_A_norm), -1, 1))
        chi_a = np.arctan2(g_v_A[1], g_v_A[0])
        
        # aerodynamic force in aerodynamic coordinates
        cl = 2*np.pi*(self._params_glider.ST/(self._params_glider.ST + 2))*alpha
        cd = self._params_glider.CD0 + (1/(np.pi*self._params_glider.ST*self._params_glider.OE))*np.power(cl, 2)
        a_f_A = (self._params_physics.RHO/2)*self._params_glider.S*np.power(v_A_norm, 2)*np.array([[-cd], [0], [-cl]])

        # aerodynamic force in local NED coordinates
        g_T_a = self.getRotationMatrix(-chi_a.item(), 3)\
                @ self.getRotationMatrix(-gamma_a.item(), 2)\
                @ self.getRotationMatrix(-mu_a, 1)
        g_f_A = g_T_a @ a_f_A

        # track acceleration in local NED coordinates
        g_a_K = (g_f_A/self._params_glider.M) + np.array([[0], [0], [self._params_physics.G]])

        # state derivative
        xp = np.append(g_v_K, g_a_K)

        return xp
        if np.isnan(xp).any():
            print("xp is not a number: {}".format(xp))

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
            print("not a valid rotation axis")

        return rotationMatrix
       
########################################################################################################################

    def state2observation(self):

        # vector from previous vertex to active vertex in g-coordinates (-> r-frame x-axis = g_ref/norm(g_ref))
        previousVertex = np.mod((self.activeVertex - 1), 3)
        g_previous2active = self._params_task.TRIANGLE[:, (self.activeVertex - 1)]\
                            - self._params_task.TRIANGLE[:, (previousVertex - 1)]

        # polar angle of r-frame wrt g-frame
        g_phi_r = np.arctan2(g_previous2active[1], g_previous2active[0]).item()

        # rotation matrix from local NED-coordinates to reference coordinates
        r_T_g = self.getRotationMatrix(g_phi_r, 3)

        # vector from origin of r-frame (i.e., previous vertex) to aircraft in g-coordinates
        g_previous2aircraft = self.state[0:2] - self._params_task.TRIANGLE[:, (previousVertex - 1)]

        # aircraft position in cartesian r-coordinates
        r_p = r_T_g @ np.append(g_previous2aircraft, self.state[2]).reshape(3, 1)

        # polar angle of aircraft position wrt x_r-axis
        r_phi = np.arctan2(r_p[1], r_p[0]).item()

        # vector from active vertex to aircraft in g-coordinates
        g_active2aircraft = self.state[0:2] - self._params_task.TRIANGLE[:, (self.activeVertex - 1)]

        # aircraft distance from active vertex
        dist2active = np.linalg.norm(g_active2aircraft)

        # tack speed in r-coordinates
        g_v_K = self.state[3:6].reshape(3, 1)
        r_v_K = r_T_g @ g_v_K

        # norm, azimuth wrt x_r-axis, and flight path angle of track speed vector
        V_K = np.linalg.norm(r_v_K)
        r_chi = np.arctan2(r_v_K[1], r_v_K[0]).item()
        gamma = -np.arcsin(r_v_K[2] / V_K).item()

        # get wind vector at current aircraft position
        g_v_W = self._wind_fun.get_current_wind(self.state[0:3])

        # stack observation and normalize:
        # o = [time, vertex counter,
        # distance to next vertex, polar angle wrt x_r, height,
        # v_K in polar coords. wrt r-frame, updraft]
        observation = (np.array([self.time, self.vertexCounter,
                                 dist2active, r_phi, -self.state[2],
                                 V_K, r_chi, gamma,
                                 -g_v_W[2].item()])
                       - self._params_task.OBS_MEAN) / self._params_task.OBS_STD

        return observation

########################################################################################################################

    def getRewardAndDoneFlag(self, timestep):

        vertexReward = self.setActiveVertex()
        lapReward = self.getLapReward()

        if self._params_task.TASK == 'distance':
            # reward = lapReward
            reward = vertexReward
        elif self._params_task.TASK == 'speed':
            # TBD
            print("task not implemented, yet")
        else:
            print("no such task")

        ground      = (-self.state[2] <= 0)
        timeout     = (self.time >= self._params_task.WORKING_TIME)
        # outofsight  = (np.linalg.norm(self.state[0:2]) > self._params_task.DISTANCE_MAX)
        outofsight = False

        done = (ground or timeout or outofsight or (not np.isfinite(self.state).all()))

        return reward, done

    def setActiveVertex(self):

        # get horizontal aircraft position in active-sector-coordinates
        sec_T_g = self.getTrafoToActiveSectorCoords()
        sec_pos_ac = sec_T_g @ (self.state[0:2].reshape(2, 1)
                                - self._params_task.TRIANGLE[:, (self.activeVertex - 1)].reshape(2, 1))

        # special reward for hitting vertices
        vertexReward = 0

        # update active vertex if both active-sector-coordinates are positive
        if (sec_pos_ac >= 0).all():
            # print("hit active vertex no. {}" .format(self.activeVertex))
            vertexReward = 200/3
            self.vertexCounter += 1
            if (self.activeVertex + 1) > 3:
                self.activeVertex = 1
            else:
                self.activeVertex += 1

        return vertexReward

    def getTrafoToActiveSectorCoords(self):

        if self.activeVertex == 1:
            # rotation matrix from geodetic to sector-one-coordinates
            sec_T_g = (self._params_task.ONE_T_T @ np.transpose(self._params_task.G_T_T))
        elif self.activeVertex == 2:
            # rotation matrix from geodetic to sector-two-coordinates
            sec_T_g = (self._params_task.TWO_T_T @ np.transpose(self._params_task.G_T_T))
        elif self.activeVertex == 3:
            # rotation matrix from geodetic to sector-three-coordinates
            sec_T_g = (self._params_task.THREE_T_T @ np.transpose(self._params_task.G_T_T))
        else:
            sec_T_g = None
            print("active vertex no. {} is not a valid triangle vertex".format(self.activeVertex))

        return sec_T_g

    def getLapReward(self):

        T_pos_ac = np.transpose(self._params_task.G_T_T) @ self.state[0:2].reshape(2, 1)
        lapReward = 0

        if self.vertexCounter == 3 and T_pos_ac[1] >= 0:
            self.lapCounter += 1
            self.vertexCounter = 0
            # lapReward = 170
            lapReward = 200

        return lapReward

#########################################################################################################

    def getInfo(self):
        x = self.state
        info = {"t": self.time, "x": x[0], "y": x[1], "z": x[2], "u": x[3], "v": x[4], "w": x[5]}
        return info

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None