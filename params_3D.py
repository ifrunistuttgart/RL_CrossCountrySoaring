import numpy as np

class params_rl:
    def __init__(self):
        self.N_EPOCH            = 1e3             # total number of episodes to be evaluated
        self.N_EPPERITER        = 5               # number of episodes to be evaluated within a policy update
        self.LEARNING_RATE      = 1e-4
        self.GAMMA              = 0.99
        self.LAMBDA             = 0.96
        self.EPS_CLIP           = 0.2
        self.N_UPDATE           = 3                 # number of policy updates for batch of N_EPPERITER episodes
        self.SIGMA              = .2                # std-deviation for exploration (0.1 -> 0.6 deg after scaling)
        self.AUTO_EXPLORATION   = False              # exploration driven by NN output (self.SIGMA obsolete if true)
        self.SEED               = None              # manual specification of random seed (Fabian: 42)

class params_model:
    def __init__(self):
        self.DIM_IN         = 2                     # dimension of observation-space (time + state + wind) TODO: 10 including wind
        self.DIM_OUT        = 1                     # dimension of action-space
        self.DIM_HIDDEN     = 64
        self.NUM_HIDDEN     = 2

class params_task:
    def __init__(self):
        # triangle orientation: alpha - pi/2 ('alpha' referred to in GPS triangle regulatioself.TRIANGLEn code)
        self.ORIENTATION    = 0
        # transformation matrix from triangle-coordinates to local NE-coordinates
        self.G_T_T          = np.transpose(np.array([[np.cos(self.ORIENTATION), np.sin(self.ORIENTATION)],
                                                     [-np.sin(self.ORIENTATION), np.cos(self.ORIENTATION)]]))
        # local NE-coordinates of the triangle vertices (2 x 3)
        self.TRIANGLE       = self.G_T_T @ np.array([[0., 350, 0.],
                                                     [350, 0., -350]])

        # rotation matrix from triangle-coordinates to sector-one-coordinates (translation needed for transformation)
        self.ONE_T_T        = np.array([[np.cos(3 * np.pi / 8), np.sin(3 * np.pi / 8)],
                                        [-np.sin(3 * np.pi / 8), np.cos(3 * np.pi / 8)]])
        # rotation matrix from triangle-coordinates to sector-two-coordinates (translation needed for transformation)
        self.TWO_T_T        = np.array([[np.cos(-np.pi / 4), np.sin(-np.pi / 4)],
                                        [-np.sin(-np.pi / 4), np.cos(-np.pi / 4)]])
        # rotation matrix from triangle-coordinates to sector-three-coordinates (translation needed for transformation)
        self.THREE_T_T      = np.array([[np.cos(9 * np.pi / 8), np.sin(9 * np.pi / 8)],
                                        [-np.sin(9 * np.pi / 8), np.cos(9 * np.pi / 8)]])

        # task (alternatives: 'distance', 'speed')
        self.TASK           = 'distance'
        # working time to have the task done (relevant for 'distance' only)
        self.WORKING_TIME   = 60*30
        # maximum horizontal distance allowed from origin (ensures that a safety pilot could interact)
        self.DISTANCE_MAX   = 1e3

        # initialization method (alternatives: 'fix', 'random_normal')
        self.INIT_METHOD    = 'fix'
        # initial state value (s = g_[p_north, p_east, p_down, v_north, v_east, v_down])
        self.INITIAL_STATE  = np.append(np.array([0., 0., -400.]),
                                        np.append(self.G_T_T @ np.array([[0.], [15.]]), 1))
        # std-dev. for init method: random_normal
        self.INITIAL_STD    = np.append(np.append(self.G_T_T @ np.array([[20.], [0.]]), 20.),
                                        np.array([2., 2., .5]))

        # bank angle and AoA constraint to [MIN MAX]   (deg)
        # self.ACTION_SPACE   = np.array([[-45, 45],
        #                                 [0, 12]])
        self.ACTION_SPACE = np.array([[-45, 45]])

        # # mean values for observation standardization (o = [t, p_dist, p_chi, -height, v_norm, v_chi, v_down])
        # self.OBS_MEAN       = np.array([self.WORKING_TIME/2,
        #                                 350, 0., self.INITIAL_STATE[2]/2,
        #                                 15., 0., 1.])
        # # spread parameters for observation standardization (o = [t, p_dist, p_chi, -height, v_norm, v_chi, v_down])
        # self.OBS_STD        = np.array([self.WORKING_TIME/2,
        #                                 350, np.pi, abs(self.INITIAL_STATE[2])/2,
        #                                 10., np.pi, 2.])

        # mean values for observation standardization (o = [v_norm, r_chi_v)])
        self.OBS_MEAN = np.array([15., 0.])
        # spread parameters for observation standardization (o = [v_norm, r_chi_v)])
        self.OBS_STD = np.array([10., np.pi])


class params_sim:
    def __init__(self):
        self.TIMESTEP       = 0.2                   # const. u simulation time-step (s)
        self.USE_RK45       = True                  # Runge-Kutta 45 integration, False -> Euler forward  # TODO: Check Euler


class params_glider:
    def __init__(self):
        self.M              = 3.366                 # aircraft mass                 (kg)
        self.S              = .568                  # reference area                (m2)
        self.ST             = 10.2                  # aspect ratio                  (-)
        self.OE             = .9                    # oswald factor                 (-)
        self.CD0            = .015                  # zero lift drag coefficient    (-)

class params_physics:
    def __init__(self):
        self.RHO            = 1.225                 # air density                   (kg/m3)
        self.G              = 9.81                  # gravitational acceleration    (m/s2)

class params_logging:
    def __init__(self):
        self.PRINT_INTERVAL = 10
        self.SAVE_INTERVAL  = 100