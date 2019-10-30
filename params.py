import numpy as np

class params_rl:
    def __init__(self):
        self.N_EPOCH            = 10e3
        self.LEARNING_RATE      = 1e-5
        self.GAMMA              = 0.99
        self.LMBDA              = 1.0
        self.EPS_CLIP           = 0.2
        self.K_EPOCH            = 3
        self.SIGMA              = .5                # std-deviation for exploration (0.1 -> 0.6 deg after scaling)

class params_model:
    def __init__(self):
        self.DIM_IN         = 4
        self.DIM_OUT        = 1                     # dimension of action-space
        self.DIM_HIDDEN     = 64
        self.NUM_HIDDEN     = 2

class params_task:
    def __init__(self):
        self.DISTANCE       = 2000                  # distance to be covered        (m)
        self.INIT_METHOD    = 'random_normal'       # alternatives: 'fix', 'random_normal', 'random_uniform'

        # initial state value
        self.INITIAL_STATE  = np.array([0., -100., 10., 1.])
        # std-dev. for init method: random_normal (caution: first value must no be large, otherwise done might be true)
        self.INITIAL_STD    = np.array([.5, 5., 2., .5])
        # range [MIN MAX] for init method: random_uniform
        self.INITIAL_RANGE  = np.array([[0, -100, 10, -10], [self.DISTANCE, 0, 30, 20]])
        # AoA constraint to [MIN MAX]   (deg)
        self.ACTION_SPACE   = np.array([1e-3, 12])
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
        self.WL             = 6.177                 # wing loading                  (kg/m2)
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