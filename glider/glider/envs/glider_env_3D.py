import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import scipy.integrate as integrate
import sys
import os

sys.path.append(os.path.join("..", "..", ".."))

from params import *

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
            self._integrator = integrate.ode(self.buildDynamic2D).set_integrator('dopri5')
        else:
            self._integrator = 'euler'

        self.seed()
        self.state = None
        self.viewer = None

        self.time = None
        self.action = None

#########################################################################################################

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

##########################################################################################################

    def step(self, action):
        self.action = self.scale_action(action)
        self.integrate()
        obs_stand = self.standardize_observations(self.state)
        reward, done = self.getRewardAndDoneFlag()
        info = self.getInfo()
        return np.array(obs_stand), reward, done, info

#########################################################################################################

    def reset(self):
        if self._params_task.INIT_METHOD == 'random_uniform':
            self.state = self.np_random.uniform(low=self._params_task.INITIAL_RANGE[0, :],
                                                high=self._params_task.INITIAL_RANGE[1, :])
        elif self._params_task.INIT_METHOD == 'random_normal':
            self.state = self.np_random.multivariate_normal(self._params_task.INITIAL_STATE,
                                                            np.diag(np.square(self._params_task.INITIAL_STD)))
        elif self._params_task.INIT_METHOD == 'fix':
            self.state = np.copy(self._params_task.INITIAL_STATE)

        self.time = 0
        return np.array(self.state)

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
        v_A_norm = np.linalg.norm(g_V_A)
        gamma_a = -np.arcsin(g_v_A[2]/v_A_norm)
        chi_a = np.arctan2(g_v_A[1], g_v_A[0])

        # rotation matrix: aerodynamic frame of reference -> geodetic frame of reference
        g_T_a = np.array([[np.float(np.cos(gamma_a)), np.float(np.sin(gamma_a))],
                          [-np.float(np.sin(gamma_a)), np.float(np.cos(gamma_a))]])
        
        # specific aerodynamic force in aerodynamic coordinates
        cl = 2*np.pi*(self._params_glider.ST/(self._params_glider.ST + 2))*alpha
        cd = self._params_glider.CD0 + (1/(np.pi*self._params_glider.ST*self._params_glider.OE))*np.power(cl, 2)
        a_f_A = np.array([[-cd], [0], [-cl]])

        # specific aerodynamic force in local NED coordinates
        g_T_a = self.getRotationMatrix(-chi_a, 3)*self.getRotationMatrix(-gamma_a, 2)*self.getRotationMatrix(-mu_a, 1)
        g_f_A = (self._params_physics.RHO/2)*self._params_glider.S*np.power(v_A_norm, 2)*(g_T_a @ a_f_A)

        # track acceleration in local NED coordinates
        g_a_k = (g_f_A/self._params_glider.M) + np.array([[0], [0], [self._params_physics.G]])

        # state derivative
        xp1 = float(g_V_k[0])
        xp2 = float(g_V_k[1])
        xp3 = float(g_V_k[2])
        xp4 = float(g_a_k[0])
        xp5 = float(g_a_k[1])
        xp6 = float(g_a_k[2])
        xp = np.array([[xp1], [xp2], [xp3], [xp4], [xp5], [xp6]])
        return xp

########################################################################################################

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

    def standardize_observations(self, state):
        temp = np.copy(state)
        temp[0] = self._params_task.DISTANCE - temp[0]
        standardized_obs = (temp-self._params_task.OBS_MEAN)/self._params_task.OBS_STD
        return standardized_obs

    def scale_action(self, action):
        lb = min(self._params_task.ACTION_SPACE)*(np.pi/180)
        ub = max(self._params_task.ACTION_SPACE)*(np.pi/180)
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)  
        scaled_action = np.clip(scaled_action, lb, ub)
        return scaled_action

##############################################################################################################
        
    def integrate(self):
        if self._integrator == 'euler':
            x_dot = self.buildDynamic2D(self.time, self.state)
            for index in range(len(x_dot)):
                self.state[index] += float(self._params_sim.TIMESTEP * x_dot[index])
            self.time += self._params_sim.TIMESTEP
        else:
            r = self._integrator
            r.set_initial_value(self.state)
            r.integrate(self._params_sim.TIMESTEP)
            self.state = r.y
            self.time += r.t
        
##############################################################################################################

    def getRewardAndDoneFlag(self):
        x, z, vx, vz = self.state

        done = False
        reward = -self._params_sim.TIMESTEP

        height = -z
        
        # crash = (x < 0)
        ground = ((height <= 0) and (x < self._params_task.DISTANCE))
        goal = (x >= self._params_task.DISTANCE)

        # if crash:
        #     done = True
        #     reward -= (self._params_task.DISTANCE/4 + height)
            
        if ground:
            done = True
            reward -= (self._params_task.DISTANCE - x)
            
        if goal:
            done = True
            reward += self._params_task.DISTANCE/10

        return reward, done

#########################################################################################################
    
    def getInfo(self):
        x = self.state 
        info = {"x": x[0], "z": x[1], "u": x[2], "w": x[3]}
        return info