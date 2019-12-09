import numpy as np

class params_rl:
    def __init__(self):
        self.N_EPOCH            = 40e3              # total number of episodes to be evaluated
        self.N_EPPERITER        = 25               # number of episodes to be evaluated within a policy update
        self.LEARNING_RATE      = 1e-4
        self.GAMMA              = 1
        self.LAMBDA             = 0.96
        self.EPS_CLIP           = 0.2
        self.N_UPDATE           = 3                 # number of policy updates for batch of I_EPOCH episodes
        self.SIGMA              = .2                # std-deviation for exploration (0.1 -> 0.6 deg after scaling)
        self.AUTO_EXPLORATION   = True              # exploration driven by NN output (self.SIGMA obsolete if true)

class params_model:
    def __init__(self):
        self.DIM_IN         = 4                     # dimension of observation-space
        self.DIM_OUT        = 1                     # dimension of action-space
        self.DIM_HIDDEN     = 64
        self.NUM_HIDDEN     = 2

class params_task:
    def __init__(self):
        self.DISTANCE       = 2000                  # distance to be covered        (m)
        self.INIT_METHOD    = 'fix'                 # alternatives: 'fix', 'random_normal', 'random_uniform'

        # initial state value
        self.INITIAL_STATE  = np.array([0., -100., 10., 1.])
        # std-dev. for init method: random_normal
        self.INITIAL_STD    = np.array([.5, 5., 2., .5])
        # range [MIN MAX] for init method: random_uniform
        self.INITIAL_RANGE  = np.array([[0, -100, 10, -10], [self.DISTANCE, 0, 30, 20]])
        # AoA constraint to [MIN MAX]   (deg)
        self.ACTION_SPACE   = np.array([0, 12])
        # mean values for observation standardization
        self.OBS_MEAN       = np.array([self.DISTANCE/2, self.INITIAL_STATE[1]/2, 15, 2])
        # std-deviation values for observation standardization
        self.OBS_STD        = np.array([self.DISTANCE/2, abs(self.INITIAL_STATE[1])/2, 5, 2])

class params_sim:
    def __init__(self):
        self.TIMESTEP       = 0.2                   # const. u simulation time-step (s)
        self.USE_RK45       = True                  # Runge-Kutta 45 integration, False -> Euler forward


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
        self.PRINT_INTERVAL = 100
        self.SAVE_INTERVAL  = 1e3