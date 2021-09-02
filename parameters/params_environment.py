""" This module contains parameters for the simulation of the glider environment.

Classes
    -------

    WindParameters
        Parameters for the updraft model. For updraft model check
        "Updraft Model for Development of Autonomous Soaring Uninhabited Air Vehicles" by Michael Allen
        https://arc.aiaa.org/doi/10.2514/6.2006-1510

    GliderParameters
        Parameters for glider (especially aerodynamics)

    PhysicsParameters
        Physical parameters of environment

    SimulationParameters
        Parameters for numerical simulation

"""
import numpy as np


class WindParameters:
    """ Parameters for the updraft model. For updraft model check
        "Updraft Model for Development of Autonomous Soaring Uninhabited Air Vehicles" by Michael Allen
        https://arc.aiaa.org/doi/10.2514/6.2006-1510

    Attributes
    ----------

    APPLY_HEADWIND: bool
        Apply or suppress horizontal wind

    HEADWIND_HIGH: ndarray
        Max. values for headwind velocity and direction

    HEADWIND_LOW: ndarray
        Min. values for headwind velocity and direction

    APPLY_UPDRAFTS: bool
        Apply or suppress occurrence of updrafts

    UPCOUNT_MAX: int
        Max. # updrafts within DISTANCE_MAX from origin

    UPCOUNT_MIN: int
        Max. # updrafts within DISTANCE_MAX from origin

    WSTAR: float
        Updraft strength scale factor (m/s) - June, 3 pm

    ZI: float
        Convective layer height (m) - June, 3 pm

    RGAIN_STD: float
        Std-dev. for perturbations from wstar

    WGAIN_STD: float
        Std-dev. for updraft radius perturbations

    APPLY_SINK: bool
        Sink outside of thermals

    RADIUS: int
        Radius for area of interest

    DIST_MIN: int
        Minimal distance between individual updrafts

    ALWAYS_RESET: bool
        Choose whether wind is reset every episode or remains constant

    UPD_MEAN: int
        Mean value for updraft distance standardization (LSTM encoding)

    UPD_STD: int
        Spread value for updraft distance standardization (LSTM encoding)

    DELTA: int
        Delta-neighborhood to indicate updraft proximity
    """

    def __init__(self):
        self.APPLY_HEADWIND = False
        self.HEADWIND_HIGH = np.array([5., np.pi])
        self.HEADWIND_LOW = np.array([0., -np.pi])

        self.APPLY_UPDRAFTS = True
        self.UPCOUNT_MAX = 7
        self.UPCOUNT_MIN = 3
        self.WSTAR = 2.0611
        self.ZI = 1.4965e3
        self.RGAIN_STD = 0.0
        self.WGAIN_STD = 0.0
        self.APPLY_SINK = True
        self.RADIUS = 1000
        self.DIST_MIN = 100
        self.ALWAYS_RESET = True
        self.UPD_MEAN = 500
        self.UPD_STD = 500
        self.DELTA = 100


class GliderParameters:
    """ Parameters describing mass and aerodynamic properties of glider

    Attributes
    ----------

    M: float
        Aircraft mass

    S: float
        Reference area

    ST: float
        Aspect ratio

    OE: float
        Oswald factor

    CD0: float
        Zero lift drag coefficient

    Z_ALPHA: int
        Derivative
    """

    def __init__(self):
        """

        Returns
        -------
        object
        """
        self.M = 4.200
        self.S = .468
        self.ST = 10.2
        self.OE = .9
        self.CD0 = .015
        self.Z_ALPHA = -155


class PhysicsParameters:
    """ Parameters describing environment

    Attributes
    ----------

    RHO: float
        Air density (kg/m3)

    G: float
        Gravitational acceleration (m/s2)
    """

    def __init__(self):
        self.RHO = 1.225
        self.G = 9.81


class SimulationParameters:
    """ Parameters for simulation

        Attributes
        ----------

        TIMESTEP_SIM: float
            Internal integrator time-step (s) (obsolete if USE_RK45 = true)

        USE_RK45: bool
            Runge-Kutta 45 integration, False -> Euler forward
        """

    def __init__(self):
        self.TIMESTEP_SIM = 0.02
        self.USE_RK45 = False
