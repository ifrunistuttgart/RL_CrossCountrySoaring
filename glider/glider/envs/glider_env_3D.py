import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import scipy.integrate as integrate
import sys
import os

sys.path.append(os.path.join("..", "..", ".."))

from params_3D import *

class gliderEnv3D(gym.Env):

#########################################################################################################

    def __init__(self):

        # instantiate parameters
        self._params_glider = params_glider()
        self._params_physics = params_physics()
        self._params_task = params_task()
        self._params_sim = params_sim()

        # set integrator
        if self._params_sim.USE_RK45:
            self._integrator = integrate.ode(self.buildDynamic3D).set_integrator('dopri5')
        else:
            self._integrator = 'euler'

        # set random seed
        self.seed()

        # initialize further member variables
        self._V_bestGlide   = self.getBestGlide()
        self.state          = None
        self.viewer         = None
        self.time           = None
        self.action         = None
        self.activeVertex   = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getBestGlide(self):
        alpha_bestGlide = ((self._params_glider.ST + 2)
                           * np.sqrt(self._params_glider.CD0*self._params_glider.OE/self._params_glider.ST))\
                           / (2*np.sqrt(np.pi))
        cL_bestGlide = (2*np.pi*alpha_bestGlide*self._params_glider.ST)/(self._params_glider.ST + 2)
        V_bestGlide = np.sqrt((2*self._params_glider.M*self._params_physics.G)
                              /(self._params_physics.RHO*self._params_glider.S*cL_bestGlide))
        return V_bestGlide

##########################################################################################################

    def step(self, action, timestep=None):
        if not timestep:
            timestep = self._params_sim.TIMESTEP

        self.action = self.scale_action(action)
        self.integrate(timestep)
        obs_stand = self.standardize_observations(self.state)
        reward, done = self.getRewardAndDoneFlag(timestep)
        self.setActiveVertex()
        info = self.getInfo()
        return np.array(obs_stand), reward, done, info

#########################################################################################################

    def reset(self):
        if self._params_task.INIT_METHOD == 'random_normal':
            initState = self.np_random.multivariate_normal(self._params_task.INITIAL_STATE,
                                                            np.diag(np.square(self._params_task.INITIAL_STD)))
        elif self._params_task.INIT_METHOD == 'fix':
            initState = np.copy(self._params_task.INITIAL_STATE)

        self.time = 0
        self.activeVertex = 1
        self.state = np.copy(np.append(self.time, initState))
        return self.state

#########################################################################################################

    def render(self, mode='human'):
        pass

########################################################################################################

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            
########################################################################################################

    def buildDynamic3D(self, t, x, wind=np.array([[0], [0], [0]])):
        # control variables assignment
        mu_a = self.action.item(0)
        alpha = self.action.item(1)

        # track speed in local NED coordinates
        g_v_K = x[3:6].reshape(3, 1)

        # airspeed in local NED coordinates
        g_v_A = g_v_K - wind

        # air-path angles
        v_A_norm = np.linalg.norm(g_v_A)
        gamma_a = -np.arcsin(g_v_A[2]/v_A_norm)
        chi_a = np.arctan2(g_v_A[1], g_v_A[0])
        
        # aerodynamic force in aerodynamic coordinates
        cl = 2*np.pi*(self._params_glider.ST/(self._params_glider.ST + 2))*alpha
        cd = self._params_glider.CD0 + (1/(np.pi*self._params_glider.ST*self._params_glider.OE))*np.power(cl, 2)
        a_f_A = (self._params_physics.RHO/2)*self._params_glider.S*np.power(v_A_norm, 2)*np.array([[-cd], [0], [-cl]])

        # aerodynamic force in local NED coordinates
        g_T_a = self.getRotationMatrix(-chi_a, 3)\
                @ self.getRotationMatrix(-gamma_a, 2)\
                @ self.getRotationMatrix(-mu_a, 1)
        g_f_A = g_T_a @ a_f_A

        # track acceleration in local NED coordinates
        g_a_k = (g_f_A/self._params_glider.M) + np.array([[0], [0], [self._params_physics.G]])

        # state derivative
        xp1 = float(g_v_K[0])
        xp2 = float(g_v_K[1])
        xp3 = float(g_v_K[2])
        xp4 = float(g_a_k[0])
        xp5 = float(g_a_k[1])
        xp6 = float(g_a_k[2])
        xp = np.array([[xp1], [xp2], [xp3], [xp4], [xp5], [xp6]])
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
            print("not a valid rotation axis")
        return rotationMatrix
       
###############################################################################################################

    def standardize_observations(self, state):  # TODO: check that method
        temp = np.copy(state)
        standardized_obs = (temp - self._params_task.OBS_MEAN)/self._params_task.OBS_STD
        return standardized_obs

    def scale_action(self, action):  # TODO: check that method (dim(a) = 2, now!)
        lb = np.min(self._params_task.ACTION_SPACE, 1)*(np.pi/180)
        ub = np.max(self._params_task.ACTION_SPACE, 1)*(np.pi/180)
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)  
        scaled_action = np.clip(scaled_action, lb, ub)
        return scaled_action

##############################################################################################################
        
    def integrate(self, timestep):
        if self._integrator == 'euler':
            x_dot = self.buildDynamic3D(self.time, self.state[-(len(self.state)-1):])  # TODO: Check indexing (fm state)
            self.state[0] += timestep
            for index in range(len(x_dot)):
                self.state[index+1] += float(timestep * x_dot[index])
            self.time += timestep
        else:
            r = self._integrator
            r.set_initial_value(self.state[-(len(self.state)-1):])
            r.integrate(timestep)
            self.time += r.t
            self.state = np.append(self.time, r.y)
        
##############################################################################################################

    def getRewardAndDoneFlag(self, timestep):
        # t, x, y, z, vx, vy, vz = self.state

        # compute approaching distance towards active vertex
        g_dir = self._params_task.TRIANGLE[(self.activeVertex-1), 0:2].reshape(2, 1) - self.state[1:3].reshape(2, 1)
        g_v_h = self.state[4:6].reshape(2, 1)
        approach = float(np.dot(np.transpose(g_v_h), (g_dir/np.linalg.norm(g_dir))))*timestep

        if self._params_task.TASK == 'distance':
            reward = approach  # TODO: only covering distance is rewarded so far (WP-type control)
        elif self._params_task.TASK == 'speed':
            # TBD
            print("task not implemented, yet")
        else:
            print("no such task")

        ground      = (-self.state[3] <= 0)
        timeout     = (self.state[0] >= self._params_task.WORKING_TIME)
        outofsight  = (np.linalg.norm(self.state[1:3]) > self._params_task.DISTANCE_MAX)
            
        done = (ground or timeout or outofsight)  # TODO: no final reward, at the moment

        return reward, done

    def setActiveVertex(self):  # TODO: Check that method
        # t, x, y, z, vx, vy, vz = self.state

        # horizontal aircraft position in local NE-coordinates
        g_pos_ac = self.state[1:3].reshape(2, 1)

        # horizontal aircraft position in active-sector-coordinates
        sec_pos_ac = None
        if self.activeVertex == 1:
            # horizontal aircraft position in sector-one-coordinates
            sec_pos_ac = (self._params_task.ONE_T_T
                          @ np.transpose(self._params_task.G_T_T))\
                         @ (g_pos_ac - self._params_task.TRIANGLE[:, 0].reshape(2, 1))
        elif self.activeVertex == 2:
            # horizontal aircraft position in sector-two-coordinates
            sec_pos_ac = (self._params_task.TWO_T_T
                          @ np.transpose(self._params_task.G_T_T))\
                         @ (g_pos_ac - self._params_task.TRIANGLE[:, 1].reshape(2, 1))
        elif self.activeVertex == 3:
            # horizontal aircraft position in sector-three-coordinates
            sec_pos_ac = (self._params_task.THREE_T_T
                          @ np.transpose(self._params_task.G_T_T))\
                         @ (g_pos_ac - self._params_task.TRIANGLE[:, 2].reshape(2, 1))
        else:
            print("not a triangle-vertex")

        # update active vertex if both active-sector-coordinates are positive
        if (sec_pos_ac >= 0).all():
            if (self.activeVertex + 1) > 3:
                self.activeVertex = 1
            else:
                self.activeVertex += 1

#########################################################################################################

    def getInfo(self):
        x = self.state
        info = {"t": x[0], "x": x[1], "y": x[2], "z": x[3], "u": x[4], "v": x[5], "w": x[6]}
        return info