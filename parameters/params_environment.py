import numpy as np

class params_wind:
    def __init__(self):
        self.APPLY_HEADWIND = False                 # apply or suppress horizontal wind
        self.HEADWIND_HIGH = np.array([5., np.pi])  # max. values for headwind velocity and direction
        self.HEADWIND_LOW = np.array([0., -np.pi])  # min. values for headwind velocity and direction

        self.APPLY_UPDRAFTS = True                  # apply or suppress occurrence of updrafts
        self.UPCOUNT_MAX    = 7                     # max. # updrafts within DISTANCE_MAX from origin
        self.UPCOUNT_MIN    = 3                     # max. # updrafts within DISTANCE_MAX from origin
        self.WSTAR          = 2.0611                # updraft strength scale factor (m/s) - June, 3 pm
        self.ZI             = 1.4965e3              # convective layer height (m) - June, 3 pm
        self.RGAIN_STD      = 0.0                   # std-dev. for perturbations from wstar
        self.WGAIN_STD      = 0.0                   # std-dev. for updraft radius perturbations
        self.APPLY_SINK     = True                  # sink outside of thermals
        self.RADIUS         = 1000                  # radius for area of interest
        self.DIST_MIN       = 100                   # minimal distance between individual updrafts

        self.ALWAYS_RESET   = True                  # choose whether wind is reset every episode or remains constant

        self.UPD_MEAN       = 500                   # mean value for updraft distance standardization (LSTM encoding)
        self.UPD_STD        = 500                   # spread value for updraft distance standardization (LSTM encoding)

        self.DELTA          = 100                   # delta-neighborhood to indicate updraft proximity

class params_glider:
    def __init__(self):
        self.M              = 4.200                 # aircraft mass                 (kg)
        self.S              = .468                  # reference area                (m2)
        self.ST             = 10.2                  # aspect ratio                  (-)
        self.OE             = .9                    # oswald factor                 (-)
        self.CD0            = .015                  # zero lift drag coefficient    (-)
        self.Z_ALPHA       = -155                  # derivative                    (m/s2)

class params_physics:
    def __init__(self):
        self.RHO            = 1.225                 # air density                   (kg/m3)
        self.G              = 9.81                  # gravitational acceleration    (m/s2)

class params_sim:
    def __init__(self):
        self.TIMESTEP_SIM = 0.02                    # internal integrator time-step (s) (obsolete if USE_RK45 = true)
        self.USE_RK45 = False                       # Runge-Kutta 45 integration, False -> Euler forward