import numpy as np

from parameters.params_triangle_soaring import params_task


class params_rl:
    def __init__(self):
        self.N_EPISODES         = 2e2            # total number of episodes to be evaluated
        self.BATCHSIZE          = 4096          # size of bach (i.e, length of rollout) before policy update
        self.MINIBATCHSIZE      = 64            # size of mini-batch for SGD
        self.K_EPOCH            = 10            # number of policy updates on single batch
        self.LEARNING_RATE      = 5e-5          # learning rate for optimizer
        self.GAMMA              = 0.99          # discount factor for advantage estimation
        self.LAMBDA             = 0.96          # only relevant if GAE is implemented
        self.EPS_CLIP           = 0.2           # PPO clipping value
        self.SIGMA              = .2            # std-deviation for exploration (0.1 -> 0.6 deg after scaling)
        self.AUTO_EXPLORATION   = False         # exploration driven by NN output (self.SIGMA obsolete if true)
        self.SEED               = 42            # manual specification of random seed (Fabian: 42)

class params_model:
    def __init__(self):
        self.DIM_IN         = 4                 # dimension of observation-space
        self.DIM_OUT        = 2                 # dimension of action-space - for decision maker: probs for subtasks
        self.DIM_HIDDEN     = 64
        self.NUM_HIDDEN     = 2

class params_agent:
    def __init__(self):
        # instantiate params_task object
        _params_task = params_task()

        # control update time-step (s)
        self.TIMESTEP_CRTL  = 1.0

        # initial mean state value: s = g_[p_north, p_east, p_down, v_north, v_east, v_down] ('distance', 'speed')
        self.INITIAL_STATE = np.append(np.array([0., 0., -400.]),
                                       np.append(_params_task.G_T_T @ np.array([[0.], [15.]]), 1.))

        # std-dev. for state initialization ('distance', 'speed')
        self.INITIAL_STD = np.append(np.append(_params_task.G_T_T @ np.array([[20.], [0.]]), 5.),
                                     np.array([2., 2., .5]))

        # bank angle and AoA constraint to; CAUTION: Needs to be consistent to vertex_tracker & updraft_exploiter!
        self.ACTION_SPACE   = np.array([[-45, 45],
                                        [0, 12]])

        # mean values for observation standardization
        self.OBS_MEAN       = np.array([_params_task.WORKING_TIME/2,    # time
                                        -self.INITIAL_STATE[2]/2,       # height
                                        1e3, 200])                      # distance to finish line, nearest updraft

        # spread parameters for observation standardization
        self.OBS_STD        = np.array([_params_task.WORKING_TIME/2,
                                        abs(self.INITIAL_STATE[2])/2,
                                        1e3, 200])

        # maximum horizontal distance allowed from origin (ensures that a safety pilot could interact)
        self.DISTANCE_MAX = 1e3

        # maximum height allowed
        self.HEIGHT_MAX = 500

class params_logging:
    def __init__(self):
        self.PRINT_INTERVAL = 10                    # interval wrt # epi to print score and save avg. return to file
        self.SAVE_INTERVAL  = 500                   # interval wrt # epi to save actor/critic net and make a plot
