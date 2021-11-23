""" Implements a waypoint controller, which controls lateral and longitudinal position of glider. Also adds a
    controller wrapper, which wraps control parameters, environment and controller into one object
"""
import numpy as np

from parameters import params_triangle_soaring, params_environment
from parameters.params_triangle_soaring import TaskParameters
from subtasks.vertex_tracker import params_vertex_tracker
from subtasks.vertex_tracker.params_vertex_tracker import ControlParameters


class ControllerWrapper:
    """ Controller wrapper for control parameters, environment and controller.

         Attributes
        ----------
        waypoint_controller : WaypointController
            Contains longitudinal and lateral controller

        env : object
            OpenAI gym training environment

        _params_agent : AgentParameters
            Waypoint controller parameters
    """

    def __init__(self, environment):
        self.waypoint_controller = WaypointController()
        self.env = environment
        self._params_agent = params_vertex_tracker.AgentParameters()

    def select_action(self):
        """

        Returns
        -------
        action : ndarray
            Command for roll angle and angle-of-attack
        """
        phi_cmd, alpha_cmd = self.waypoint_controller.get_control(self.env.state, self.env.active_vertex)
        action = np.array([self.wrap_to_interval(phi_cmd, self._params_agent.ACTION_SPACE[0, :] * np.pi / 180),
                           self.wrap_to_interval(alpha_cmd, self._params_agent.ACTION_SPACE[1, :] * np.pi / 180)])
        return action

    @staticmethod
    def wrap_to_interval(value, source_interval, target_interval=np.array([-1, 1])):
        """ Maps source_interval to target_interval

        Parameters
        ----------
        value : float
            Control value

        source_interval : ndarray
            Original interval of value

        target_interval :
            Target interval of value

        Returns
        -------
        wrapped_value : float
            Control command, mapped from source_interval to target_interval
        """

        wrapped_value = np.interp(value, (source_interval.min(), source_interval.max()),
                                  (target_interval.min(), target_interval.max()))
        return wrapped_value


class WaypointController:
    """ Controller wrapper for control parameters, environment and controller.

    Attributes
    ----------

    _params_task : TaskParameters
        Parameters which describe soaring task

    _params_glider : object
        Parameters which describe physical properties of glider

    _params_physics : PhysicsParameters
        Physical constants like gravity and air density

    _params_control : ControlParameters
        Parameters for controller like maximum control parameters
    """

    def __init__(self):
        self._params_task = params_triangle_soaring.TaskParameters()
        self._params_glider = params_environment.GliderParameters()
        self._params_physics = params_environment.PhysicsParameters()
        self._params_control = params_vertex_tracker.ControlParameters()

    def get_control(self, state, active_vertex_id):
        """ Calculates roll angle command with lateral controller. Calculates angle of attack with longitudinal
            controller

        Parameters
        ----------
        state : ndarray
            State of glider which contains position and velocity

        active_vertex_id : int
            ID of current target vertex

        Returns
        -------
        phi_cmd : float
            Commanded roll angle

        alpha_cmd : float
            Commanded angle-of-attack
        """

        phi_cmd = self.controller_lat(state, active_vertex_id)
        alpha_cmd = self.controller_lon(phi_cmd)

        return phi_cmd, alpha_cmd

    def controller_lat(self, state, active_vertex_id):
        """ Controls lateral movement of glider with roll angle to hit target vertex sector

        Parameters
        ----------
        state : ndarray
            State of glider which contains position and velocity

        active_vertex_id : int
            ID of current target vertex

        Returns
        -------
        phi_cmd : float
            Commanded roll angle
        """

        position = state[0:3]
        velocity = state[3:6]

        chi = np.arctan2(velocity[1], velocity[0])
        chi_cmd = self.guidance(position, active_vertex_id)

        chi_error = chi_cmd - chi

        if chi_error > np.pi:
            chi_error -= (2 * np.pi)
        elif chi_error < -np.pi:
            chi_error += (2 * np.pi)

        speed = np.linalg.norm(velocity)

        if speed < 10:
            phi_cmd = chi_error * self._params_control.K_CHI / 10
        else:
            phi_cmd = chi_error * self._params_control.K_CHI / speed

        phi_cmd = np.clip(phi_cmd, -self._params_control.PHI_MAX, self._params_control.PHI_MAX)

        return phi_cmd

    def guidance(self, position, active_vertex_id):
        """ Calculates chi command for lateral controller

        Parameters
        ----------
        position : ndarray
            Position of glider

        active_vertex_id : int
            ID of current target vertex

        Returns
        -------
        chi_cmd : float
            Commanded azimuth to hit target vertex
        """

        # stretching triangle slightly ensures hitting the sectors (especially vertex #2)
        waypoint = self._params_control.STRETCH * self._params_task.TRIANGLE[:, (active_vertex_id - 1)]
        g_los = waypoint - position[0:2]
        chi_cmd = np.arctan2(g_los[1], g_los[0])

        return chi_cmd

    def controller_lon(self, phi):
        """ Controls lateral movement of glider with angle-of-attack to fly at point of best glide

        Parameters
        ----------
        phi : float
            Commanded roll angle

        Returns
        -------
        alpha_cmd : float
            Commanded angle-of-attack

        """
        alpha_bestGlide = ((self._params_glider.ST + 2)
                           * np.sqrt(self._params_glider.CD0 * self._params_glider.OE / self._params_glider.ST)) \
                          / (2 * np.sqrt(np.pi))

        # add turn compensation
        alpha_turn = -(self._params_physics.G / self._params_glider.Z_ALPHA) * (1 / np.cos(phi) - 1)
        alpha_cmd = alpha_bestGlide + alpha_turn

        # limit control command
        alpha_cmd = np.clip(alpha_cmd, -self._params_control.AOA_MIN, self._params_control.AOA_MAX)

        return alpha_cmd
