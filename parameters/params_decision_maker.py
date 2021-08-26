""" This module contains parameters for the decision maker

    Classes
    -------

    LearningParameters
         Hyperparameters for model training

    ModelParameters
        Architecture of ANN

    AgentParameters
        Parameters for glider initialization and simulation

    LoggingParameters
        Print and save interval for data logger


#TODO Was ist GAE?
"""

import numpy as np
from parameters.params_triangle_soaring import TaskParameters


class LearningParameters:
    """ Hyperparameters for training actor-critic model

    Attributes
    ----------

    N_ITERATIONS: float
        total number of policy iterations during training

    BATCHSIZE: int
        size of bach (i.e, length of rollout) before policy update

    SEQ_LEN: int
        size of mini-batch for SGD (evaluated as a sequence, due to LSTM)

    OVERLAP: int
        shift tuples for overlapping sequences (shift = seq_len - overlap)

    SEQ_LEN_MIN: int
        use first steps of each sequence to "burn in" hidden state

    N_BURN_IN: int
        use first steps of each sequence to "burn in" hidden state

    K_EPOCH: int
        number of policy updates on single batch

    LEARNING_RATE_PI: float
        learning rate for actor optimizer

    LEARNING_RATE_VF: float
        learning rate for critic optimizer

    GAMMA: float
        discount factor for advantage estimation

    LAMBDA: float
        only relevant if GAE is implemented

    EPS_CLIP: float
        PPO clipping value

    SIGMA: float
        std-deviation for exploration (0.1 -> 0.6 deg after scaling)

    AUTO_EXPLORATION: bool
        exploration driven by NN output (self.SIGMA obsolete if true)

    SEED: None
        manual specification of random seed (Fabian: 42)
"""

    def __init__(self):
        self.N_ITERATIONS = 4e3
        self.BATCHSIZE = 4096
        self.SEQ_LEN = 256
        self.OVERLAP = 128
        self.SEQ_LEN_MIN = 64
        self.N_BURN_IN = 16
        self.K_EPOCH = 10
        self.LEARNING_RATE_PI = 1e-5
        self.LEARNING_RATE_VF = 1e-4
        self.GAMMA = 0.997
        self.LAMBDA = 0.96
        self.EPS_CLIP = 0.2
        self.SIGMA = .2
        self.AUTO_EXPLORATION = False
        self.SEED = None


class ModelParameters:
    """ Parameters which describe ANN architecture

    Attributes
    ----------

    DIM_IN: int
        Dimension of input layer (observation-space)

    DIM_OUT: int
        Dimension of output layer (action-space). For decision maker: prob. for subtask

    DIM_HIDDEN: int
        Dimension of hidden layer

    DIM_LSTM: int
        Dimension of LSTM layer
    """

    def __init__(self):
        self.DIM_IN = 3
        self.DIM_OUT = 1
        self.DIM_HIDDEN = 32
        self.DIM_LSTM = 32


class AgentParameters:
    """ Parameters for initializing and simulation agent (glider)

    Attributes
    ----------

    TIMESTEP_CTRL: float
        Control update time-step [s]

    INITIAL_STATE: ndarray
        Initial mean state value in NED coordinates [p_north, p_east, p_down, v_north, v_east, v_down]

    INITIAL_STD: ndarray
        Standard deviation for initial state

    ACTION_SPACE: ndarray
        Max. bank angle and AoA constraint

    OBS_MEAN: ndarray
        Mean values for observation standardization

    OBS_STD: ndarray
        Spread parameters for observation standardization

    DISTANCE_MAX: float
        Maximum horizontal distance allowed from origin (ensures that a safety pilot could interact)

    HEIGHT_MAX: int
        Maximum altitude above ground
    """

    def __init__(self):
        _params_task = TaskParameters()  # get task parameters
        self.TIMESTEP_CTRL = 1.0
        self.INITIAL_STATE = np.append(np.array([0., 0., -400.]),
                                       np.append(_params_task.G_T_T @ np.array([[0.], [15.]]), 1.))

        self.INITIAL_STD = np.append(np.append(_params_task.G_T_T @ np.array([[2.], [0.]]), 5.),
                                     np.array([2., 2., .5]))

        self.ACTION_SPACE = np.array([[-45, 45],
                                      [0, 12]])  # CAUTION: Has to be consistent to vertex_tracker & updraft_exploiter!

        self.OBS_MEAN = np.array([_params_task.WORKING_TIME / 2,  # time
                                  -self.INITIAL_STATE[2] / 2,  # height
                                  350 + np.sqrt(2) * 350])  # distance to finish line (0.5*circumference)

        self.OBS_STD = np.array([_params_task.WORKING_TIME / 2,
                                 abs(self.INITIAL_STATE[2]) / 2,
                                 350 + np.sqrt(2) * 350])
        self.DISTANCE_MAX = 1e3
        self.HEIGHT_MAX = 500


class LoggingParameters:
    """ Intervals for saving and logging

    Attributes
    ----------

    PRINT_INTERVAL: int
        Interval wrt # epi to print score and save avg. return to file
    SAVE_INTERVAL: int
        Interval wrt # epi to save actor/critic net and make a plot
    """

    def __init__(self):
        self.PRINT_INTERVAL = 10
        self.SAVE_INTERVAL = 250
