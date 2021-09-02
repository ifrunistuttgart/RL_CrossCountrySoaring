"""
########################
########################
########################
#TODO: Fliegt vermutlich raus
########################
########################
########################
"""



import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import scipy.integrate as integrate
import sys
import os

sys.path.append(os.path.join("..", "..", ".."))

from params_2D import *

class gliderEnv2D(gym.Env):

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

        # set random seed
        self.seed()

        # initialize further member variables
        self._V_bestGlide = self.get_best_glide()
        self.state = None
        self.viewer = None
        self.time = None
        self.action = None

#########################################################################################################

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

##########################################################################################################

    def get_best_glide(self):
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
        reward, done = self.get_reward_and_done(timestep)
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

    def buildDynamic2D(self, t, x):
        # control variable: angle of attack
        alpha = self.action.item(0)

        # track speed vector wrt geodetic frame of reference
        g_V_k = x[2:4].reshape(2, 1)

        # airspeed vector wrt geodetic frame of reference
        g_V_a = g_V_k

        # norm of airspeed
        V_A = np.linalg.norm(g_V_a)

        # air-path climb angle
        gamma_a = -np.arcsin(g_v_a[1]/V_A)  # TODO: validate gamma_a = -np.arcsin(g_v_a[1]/V_A)

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
        standardized_obs = (temp-self._params_task.OBS_MEAN)/self._params_task.OBS_STD
        return standardized_obs

    def scale_action(self, action):
        lb = min(self._params_task.ACTION_SPACE)*(np.pi/180)
        ub = max(self._params_task.ACTION_SPACE)*(np.pi/180)
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)  
        scaled_action = np.clip(scaled_action, lb, ub)
        return scaled_action

##############################################################################################################
        
    def integrate(self, timestep):
        if self._integrator == 'euler':
            x_dot = self.buildDynamic2D(self.time, self.state)
            for index in range(len(x_dot)):
                self.state[index] += float(timestep * x_dot[index])
            self.time += timestep
        else:
            r = self._integrator
            r.set_initial_value(self.state)
            r.integrate(timestep)
            self.state = r.y
            self.time += r.t
        
##############################################################################################################

    def get_reward_and_done(self, timestep):
        x, z, vx, vz = self.state

        done = False
        # reward = -timestep
        reward = 0  # as entire trajectories are rolled out, final reward only should be fine
        # reward = vx/self._V_bestGlide
        # reward = (vx*timestep)/self._params_task.DISTANCE  # approaching distance to target
        # reward = (vx*timestep)/10  # approaching distance to target

        height = -z
        ground = ((height <= 0) and (x < self._params_task.DISTANCE))
        goal = (x >= self._params_task.DISTANCE)

        if ground:
            done = True
            # reward -= ((self._params_task.DISTANCE - x)/self._params_task.DISTANCE)*1000
            reward -= (self._params_task.DISTANCE - x)
            # reward = x

        if goal:
            done = True
            # reward += (self._params_task.DISTANCE/10)

            # if self.time < (self._params_task.DISTANCE/self._V_bestGlide):
            # reward += (self._params_task.DISTANCE/self._V_bestGlide - self.time)*self._params_task.DISTANCE
            reward += ((1/self.time) - (0.8*self._V_bestGlide/self._params_task.DISTANCE))\
                      *(self._params_task.DISTANCE/(0.8*self._V_bestGlide))*self._params_task.DISTANCE  # flying fast

        return reward, done

#########################################################################################################
    
    def getInfo(self):
        x = self.state 
        info = {"x": x[0], "z": x[1], "u": x[2], "w": x[3]}
        return info