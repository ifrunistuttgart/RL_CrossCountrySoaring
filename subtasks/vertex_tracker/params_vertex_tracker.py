import numpy as np

from parameters.params_triangle_soaring import params_task

class params_rl:
    def __init__(self):
        self.N_EPISODES = 10e3                      # total number of episodes to be evaluated
        self.BATCHSIZE = 4096                       # size of bach (i.e, length of rollout) before policy update
        self.MINIBATCHSIZE = 64                     # size of mini-batch for SGD
        self.K_EPOCH = 10                           # number of policy updates on single batch
        self.LEARNING_RATE = 5e-5                   # learning rate for optimizer
        self.GAMMA = 0.99                           # discount factor for advantage estimation
        self.LAMBDA = 0.96                          # only relevant if GAE is implemented
        self.EPS_CLIP = 0.2                         # PPO clipping value
        self.SIGMA = .2                             # std-deviation for exploration (0.1 -> 0.6 deg after scaling)
        self.AUTO_EXPLORATION = False               # exploration driven by NN output (self.SIGMA obsolete if true)
        self.SEED = None                             # manual specification of random seed (Fabian: 42)


class params_model:
    def __init__(self):
        self.DIM_IN = 6                             # dimension of observation-space
        self.DIM_OUT = 2                            # dimension of action-space
        self.DIM_HIDDEN = 64
        self.NUM_HIDDEN = 2


class params_agent:
    def __init__(self):
        # instantiate params_task object
        _params_task = params_task()

        # control update time-step (s)
        self.TIMESTEP_CRTL = 1

        # initial state space [MIN MAX]
        self.INITIAL_SPACE = np.array([[-500, 500],
                                       [-500, 500],
                                       [-405, -395],
                                       [-10, 10],
                                       [-10, 10],
                                       [0.5, 1.5]])

        # bank angle and AoA constraint to [MIN MAX]   (deg)
        self.ACTION_SPACE = np.array([[-45, 45],
                                      [0, 12]])

        # mean values for observation standardization
        self.OBS_MEAN = np.array([350, 0., 400 / 2,  # distance, polar angle, height
                                  15., 0., 0.])  # vel. wrt r-frame in polar coords.

        # spread parameters for observation standardization
        self.OBS_STD = np.array([350, np.pi, 400 / 2,
                                 10., np.pi, np.pi / 6])

        # maximum horizontal distance allowed from origin (ensures that a safety pilot could interact)
        self.DISTANCE_MAX = 1e3

        # maximum height allowed
        self.HEIGHT_MAX = 500


class params_logging:
    def __init__(self):
        self.PRINT_INTERVAL = 10  # interval wrt # epi to print score and save avg. return to file
        self.SAVE_INTERVAL = 500  # interval wrt # epi to save actor/critic net and make a plot