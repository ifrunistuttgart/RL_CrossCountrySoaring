import numpy as np

class params_rl:
    def __init__(self):
        self.N_EPISODES         = 2e3           # total number of episodes to be evaluated
        self.BATCHSIZE          = 2048          # size of bach (i.e, length of rollout) before policy update
        self.MINIBATCHSIZE      = 32            # size of mini-batch for SGD
        self.K_EPOCH            = 10            # number of policy updates on single batch
        self.LEARNING_RATE      = 5e-5          # learning rate for optimizer
        self.GAMMA              = 0.99          # discount factor for advantage estimation
        self.LAMBDA             = 0.96          # only relevant if GAE is implemented
        self.EPS_CLIP           = 0.2           # PPO clipping value
        self.SIGMA              = .2            # std-deviation for exploration (0.1 -> 0.6 deg after scaling)
        self.AUTO_EXPLORATION   = False         # exploration driven by NN output (self.SIGMA obsolete if true)
        self.SEED               = None          # manual specification of random seed (Fabian: 42)

class params_model:
    def __init__(self):
        self.DIM_IN         = 9                 # dimension of observation-space
        self.DIM_OUT        = 2                 # dimension of action-space
        self.DIM_HIDDEN     = 64
        self.NUM_HIDDEN     = 2

class params_task:
    def __init__(self):
        # triangle orientation: alpha - pi/2 ('alpha' referred to in GPS triangle regulations)
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

        # task (alternatives: 'distance', 'speed', 'loiter')
        self.TASK           = 'loiter'
        # working time to have the task done (relevant for 'distance' only)
        self.WORKING_TIME   = 60*30
        # maximum horizontal distance allowed from origin (ensures that a safety pilot could interact)
        self.DISTANCE_MAX   = 1e3
        # maximum height allowed
        self.HEIGHT_MAX = 500

        # initialization method (alternatives: 'fix', 'random_normal')
        self.INIT_METHOD    = 'random_normal'
        # initial state value (s = g_[p_north, p_east, p_down, v_north, v_east, v_down])
        self.INITIAL_STATE  = np.append(np.array([0., 0., -400.]),
                                        np.append(self.G_T_T @ np.array([[0.], [15.]]), 1.))
        # std-dev. for init method: random_normal
        self.INITIAL_STD    = np.append(np.append(self.G_T_T @ np.array([[20.], [0.]]), 5.),
                                        np.array([2., 2., .5]))

        # bank angle and AoA constraint to [MIN MAX]   (deg)
        self.ACTION_SPACE   = np.array([[-45, 45],
                                        [0, 12]])

        # mean values for observation standardization
        self.OBS_MEAN       = np.array([self.WORKING_TIME/2, 1.5,               # time, vertex counter
                                        350, 0., -self.INITIAL_STATE[2]/2,      # distance, polar angle, height
                                        15., 0., 0.,                            # vel. wrt r-frame in polar coords.
                                        2.])                                    # updraft at current aircraft position
        # spread parameters for observation standardization
        self.OBS_STD        = np.array([self.WORKING_TIME/2, 1.5,
                                        350, np.pi, abs(self.INITIAL_STATE[2])/2,
                                        10., np.pi, np.pi/6,
                                        2.])

class params_wind:
    def __init__(self):
        self.APPLY_HEADWIND = False                 # apply or suppress horizontal wind
        self.HEADWIND_HIGH = np.array([5., np.pi])  # max. values for headwind velocity and direction
        self.HEADWIND_LOW = np.array([0., -np.pi])  # min. values for headwind velocity and direction

        self.APPLY_UPDRAFTS = True                  # apply or suppress occurrence of updrafts
        self.UPCOUNT_MAX    = 7                     # max. # updrafts within DISTANCE_MAX from origin
        self.UPCOUNT_MIN    = 5                     # max. # updrafts within DISTANCE_MAX from origin
        self.WSTAR          = 2.0611                # updraft strength scale factor (m/s) - June, 3 pm
        self.ZI             = 1.4965e3              # convective layer height (m) - June, 3 pm
        self.RGAIN_STD      = 0.0                   # std-dev. for perturbations from wstar
        self.WGAIN_STD      = 0.0                   # std-dev. for updraft radius perturbations
        self.APPLY_SINK     = True                  # sink outside of thermals
        self.DIST_MIN       = 100                   # minimal distance between individual updrafts

        self.ALWAYS_RESET   = True                  # choose whether wind is reset every episode or remains constant

        self.UPD_MEAN       = 500                   # mean value for updraft distance standardization (LSTM encoding)
        self.UPD_STD        = 500                   # spread value for updraft distance standardization (LSTM encoding)

        self.EPSILON        = 20                    # epsilon neighborhood for updraft proximity reward
        self.DELTA          = 100                   # delta neighborhood for updraft proximity reward, DELTA > EPSILON!


class params_sim:
    def __init__(self):
        self.TIMESTEP       = 1.0                   # const. u simulation time-step (s), with constant wind
        self.TIMESTEP_INT   = 0.02                  # internal integrator time-step (s) (obsolete if USE_RK45 = true)
        self.USE_RK45       = False                 # Runge-Kutta 45 integration, False -> Euler forward

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
        self.PRINT_INTERVAL = 10                    # interval wrt # epi to print score and save avg. return to file
        self.SAVE_INTERVAL  = 200                   # interval wrt # epi to save actor/critic net and make a plot
