import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import scipy.integrate as integrate
import sys
import os

sys.path.append(os.path.join("..", "..", ".."))

from params import *

class gliderEnv(gym.Env):

#########################################################################################################

    def __init__(self):

        self._params_glider = params_glider()
        self._params_physics = params_physics()
        self._params_task = params_task()
        self._params_sim = params_sim()

        # observation and action space
        self.min_alpha = 0 * (np.pi/180)
        self.max_alpha = 12 * (np.pi/180)
        self.min_x = 0
        self.max_x = 1200
        self.min_z = -200
        self.max_z = 0
        self.min_u = -20
        self.max_u = 20
        self.min_w = -20
        self.max_w = 20        
        low_action = np.array([self.min_alpha])
        high_action = np.array([self.max_alpha])
        low_obs = np.array([self.min_x, self.min_z, self.min_u, self.min_w])
        high_obs = np.array([self.max_x, self.max_z, self.max_u, self.max_w])
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # standarization
        self._obs_mean = np.array([500., -50., 10., -2.])
        self._obs_std = np.array([300., 30., 5., 2.])

        # integrator
        if self._params_sim.USE_RK45:
            self._integrator = integrate.ode(self.buildDynamic2D).set_integrator('dopri5')
        else:
            self._integrator = 'euler'
        self.time = 0.0

        self.seed()
        self.state = None
        self.viewer = None

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
            low_obs = np.array([self.min_x, self.min_z, self.min_u, self.min_w])
            high_obs = np.array([self.max_x, self.max_z, self.max_u, self.max_w])
            self.state = self.np_random.uniform(low=low_obs, high=high_obs)
        elif self._params_task.INIT_METHOD == 'random_normal':
            low_eps = np.array([0, -1, -1, -0.1])
            high_eps = np.array([1, 1, 1, 0.1])
            self.state = np.copy(self._params_task.INITIAL_STATE) + self.np_random.uniform(low=low_eps, high=high_eps)
        elif self._params_task.INIT_METHOD == 'fix':
            self.state = np.copy(self._params_task.INITIAL_STATE)
            
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

    def buildDynamic2D(self, t, x):
        # control variable: angle of attack
        alpha = self.action[0]

        # track speed vector wrt geodetic frame of reference
        g_V_k = x[2:4].reshape(2, 1)

        # airspeed vector wrt geodetic frame of reference
        g_V_a = g_V_k

        # norm of airspeed
        V_A = np.linalg.norm(g_V_a)

        # air-path climb angle
        gamma_a = -np.arctan2(g_V_a[1], g_V_a[0])

        # rotation matrix: aerodynamic frame of reference -> geodetic frame of reference
        g_T_a = np.array([[np.float(np.cos(gamma_a)), np.float(np.sin(gamma_a))],
                          [-np.float(np.sin(gamma_a)), np.float(np.cos(gamma_a))]])
        
        # air-force wrt aerodynamic frame of reference
        cl = 2 * np.pi * (self._params_glider.ST / (self._params_glider.ST + 2)) * alpha
        cd = self._params_glider.CD0 + (1 / (np.pi * self._params_glider.ST * self._params_glider.OE)) * np.power(cl, 2)
        a_R_a = np.array([[-cd], [-cl]])

        # air-force wrt geodetic frame of reference
        g_R_a = (self._params_physics.RHO / 2) * self._params_glider.S * V_A**2 * (g_T_a @ a_R_a)

        # track acceleration wrt geodetic frame of reference
        g_a_k = (g_R_a / self._params_glider.M) + np.array([[0], [self._params_physics.G]])

        # state derivative
        xp1 = float(g_V_k[0])
        xp2 = float(g_V_k[1])
        xp3 = float(g_a_k[0])
        xp4 = float(g_a_k[1])
        xp = np.array([[xp1], [xp2], [xp3], [xp4]])
        return xp
       
###############################################################################################################

    def standardize_observations(self, state):
        temp = np.copy(state)
        temp[0] = self._params_task.DISTANCE - temp[0]
        standardized_obs = (temp-self._obs_mean)/self._obs_std
        return standardized_obs

    def scale_action(self, action):
        lb = self.action_space.low
        ub = self.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)  
        scaled_action = np.clip(scaled_action, lb, ub)
        return scaled_action

##############################################################################################################
        
    def integrate(self):
        if self._integrator == 'euler':
            x_dot = self.buildDynamic2D(self.time, self.state)
            for index in range(len(x_dot)):
                self.state[index] += float(self._params_sim.TIMESTEP * x_dot[index])
            self.time = self.time + self._params_sim.TIMESTEP
        else:
            r = self._integrator
            r.set_initial_value(self.state)
            r.integrate(self._params_sim.TIMESTEP)
            self.state = r.y
            self.time = r.t
        
##############################################################################################################

    def getRewardAndDoneFlag(self):
        x, z, vx, vz = self.state

        done = False
        reward = -self._params_sim.TIMESTEP

        height = -z
        
        crash = (x < 0)            
        ground = ((height <= 0) and (x < self._params_task.DISTANCE))
        goal = (x >= self._params_task.DISTANCE)

        if crash:
            done = True
            reward = reward - (self._params_task.DISTANCE/4 + height)
            
        if ground:
            done = True
            reward = reward - (self._params_task.DISTANCE - x)
            
        if goal:
            done = True
            reward = reward + self._params_task.DISTANCE/10

        return reward, done

#########################################################################################################
    
    def getInfo(self):
        x = self.state 
        info = {"x": x[0], "z": x[1], "u": x[2], "w": x[3]}
        return info

