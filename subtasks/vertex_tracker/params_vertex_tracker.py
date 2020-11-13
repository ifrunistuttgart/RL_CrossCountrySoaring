import numpy as np

class params_agent:
    def __init__(self):
        """
        In case of classical waypoint control, these params are only used for evaluation plots.
        """

        # control update time-step (s)
        self.TIMESTEP_CRTL = 1

        # bank angle and AoA constraint to [MIN MAX]   (deg)
        self.ACTION_SPACE = np.array([[-45, 45],
                                      [0, 12]])
        # maximum horizontal distance allowed from origin (ensures that a safety pilot could interact)
        self.DISTANCE_MAX = 1e3

        # maximum height allowed
        self.HEIGHT_MAX = 500

class params_control:
    def __init__(self):

        self.STRETCH = 1.05             # stretching the triangle slightly to ensure sectors are hit
        self.K_CHI = 5                  # lateral controller parameter
        self.PHI_MAX = 45 * np.pi/180
        self.AOA_MIN = 0
        self.AOA_MAX = 12 * np.pi/180