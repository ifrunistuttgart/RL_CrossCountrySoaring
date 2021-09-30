import numpy as np
from numpy.core._multiarray_umath import ndarray


class AgentParameters:
    """ In case of classical waypoint control, these params are only used
        for evaluation plots.

     Attributes
    ----------

    TIMESTEP_CRTL: int
        Control update time-step (s)

    ACTION_SPACE: ndarray
        Bank angle and AoA constraint to [MIN MAX]   (deg)

    DISTANCE_MAX: float
        Maximum horizontal distance allowed from origin (ensures that a safety pilot could interact)

    HEIGHT_MAX: int
        Maximum height allowed

    """

    def __init__(self):

        self.TIMESTEP_CRTL = 1
        self.ACTION_SPACE = np.array([[-45, 45],
                                      [0, 12]])
        self.DISTANCE_MAX = 1e3
        self.HEIGHT_MAX = 500


class ControlParameters:
    """ Parameters for path tracker

    Attributes
    ----------
    STRETCH: float
        Stretching the triangle slightly to ensure sectors are hit

    K_CHI: int
        Lateral controller parameter

    PHI_MAX: float
        Maximum roll angle

    AOA_MIN: int
        Minimum angle-of-attack

    AOA_MAX: float
        Maximum angle-of-attack
    """
    def __init__(self):

        self.STRETCH = 1.05
        self.K_CHI = 5
        self.PHI_MAX = 45 * np.pi/180
        self.AOA_MIN = 0
        self.AOA_MAX = 12 * np.pi/180
