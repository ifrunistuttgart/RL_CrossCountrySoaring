import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
#import scipy.integrate as integrate
#from os import path

class gliderEnv(gym.Env):
    
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

#########################################################################################################

    def __init__(self):
        # glider physical parameterss
        self.m      =   3.366   # [kg] Mass
        self.wl     =   6.177   # [kg/m^2]
        self.s      =   .568    # [m^2] Bezugflügelfläche
        self.st     =   10.2    # [m] streckung
        self.oe     =   .9      # oswald factor
        self.cd0    =   .015    # zero lift drag coefficient
        self.rho    =   1.225   # densitiy
        self.g      =   9.81    # g-acceleration

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

        # initialisation method
        self.initMethod = 'random_normal' #'fix' #'random_uniform'
        self.startState = np.array([0., -100., 10., 1.])

        # standarization
        self._obs_mean = np.array([500., -50., 10., -2.])
        self._obs_std = np.array([300., 30., 5., 2.])

        # absolut time, integration time step and integration method
        self.integrator = 'euler'
        self.time = 0.0
        self.action_dt = 0.01

        # reward fuction variables
        self.distance = 1000

        self.seed()
        self.state = None
        self.viewer = None

#########################################################################################################

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

##########################################################################################################

    def step(self, action):
        #r = integrate.ode(self.buildDynamic2D).set_integrator('dopri5', rtol=1e-2, atol=1e-4)
        #r.set_initial_value()
        scaled_action = self.scale_action(action)
        x_dot = self.buildDynamic2D(scaled_action)
        self._integrate(x_dot)
        obs_stand = self.standardize_observations(self.state)
        reward, done = self.getRewardAndDoneFlag()
        info = self.getInfo()
        self.time = self.time + self.action_dt
        return np.array(obs_stand), reward, done, info

#########################################################################################################

    def reset(self):
        if self.initMethod == 'random_uniform':
            low_obs = np.array([self.min_x, self.min_z, self.min_u, self.min_w])
            high_obs = np.array([self.max_x, self.max_z, self.max_u, self.max_w])
            self.state = self.np_random.uniform(low=low_obs, high=high_obs)
        elif self.initMethod == 'random_normal':
            low_eps = np.array([0, -1, -1, -0.1])
            high_eps = np.array([1, 1, 1, 0.1])
            self.state = np.copy(self.startState) + self.np_random.uniform(low=low_eps, high=high_eps)
        elif self.initMethod == 'fix':
            self.state = np.copy(self.startState)
            
        return np.array(self.state)

#########################################################################################################

    def render(self, mode='human'):
        pass

########################################################################################################

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            
###############################################################################################################    
    
    def buildDynamic2D(self, action):
        # state
        x = self.state
        # action
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action)) 
        alpha = action[0]
	
        # track speed geodetic
        g_V_k = x[2:4].reshape(2, 1)

        # airspeed geodetic
        g_V_a = g_V_k 
        # norm of airspeed
        V_A_norm = np.linalg.norm(g_V_a)
        # wind angle of attack
        gamma_A = -np.arctan2(g_V_a[1], g_V_a[0])

        # rotation from aerodynamic frame of reference to geodetic frame of reference
        g_T_a = np.array([[np.float(np.cos(gamma_A)), np.float(np.sin(gamma_A))], [-np.float(np.sin(gamma_A)), np.float(np.cos(gamma_A))]])  
        
        # air force in aerodynamic frame of reference
        cl = 2 * np.pi * (self.st / (self.st + 2)) * alpha
        cd = self.cd0 + (1 / (np.pi * self.st * self.oe)) * np.power(cl, 2)
        g_R_a = np.array([[-cd], [-cl]])
        # air force geodetic
        g_R = (self.rho / 2) * self.s * V_A_norm**2 * (g_T_a @ g_R_a)
        # track acceleration in geodetic frame of reference
        g_vp_k = (g_R / self.m) + np.array([[0], [self.g]])

        # state derivative
        xp1 = float(g_V_k[0])
        xp2 = float(g_V_k[1])
        xp3 = float(g_vp_k[0])
        xp4 = float(g_vp_k[1])
        xp = np.array([[xp1], [xp2], [xp3], [xp4]])
        return xp
       
###############################################################################################################

    def standardize_observations(self, state):
        temp = np.copy(state)
        temp[0] = self.distance - temp[0];
        standardized_obs = (temp-self._obs_mean)/self._obs_std
        return standardized_obs

    def scale_action(self, action):
        lb = self.action_space.low
        ub = self.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)  
        scaled_action = np.clip(scaled_action, lb, ub)
        return scaled_action

##############################################################################################################
        
    def _integrate(self, x_dot):
        if self.integrator == 'euler':
            for index in range(len(x_dot)):
                self.state[index] += float(self.action_dt * x_dot[index])

        else: # andere Integrationsmethode
            pass

        
##############################################################################################################

    def getRewardAndDoneFlag(self):
        x, z, vx, vz = self.state

        done = False
        reward = -self.action_dt

        height = -z
        
        crash = (x < 0)            
        ground = ((height <= 0) and (x < self.distance))
        goal = (x >= self.distance)

        if crash:
            done = True
            reward = reward - (self.distance / 4 + height)
            
        if ground:
        	done = True
        	reward = reward - (self.distance - x)
            
        if goal:
        	done = True
        	reward = reward + self.distance / 10

        return reward, done

#########################################################################################################
    
    def getInfo(self):
        x = self.state 
        info = {"x": x[0], "z": x[1], "u": x[2], "w": x[3]}
        return info

