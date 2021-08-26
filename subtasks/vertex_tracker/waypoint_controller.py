import numpy as np

from parameters import params_triangle_soaring, params_environment
from subtasks.vertex_tracker import params_vertex_tracker

class Controller_Wrapper():
    def __init__(self, environment):
        self.waypoint_controller = Waypoint_Controller()
        self.env = environment
        self._params_agent = params_vertex_tracker.params_agent()

    def select_action(self, obserervation, memory=None, validation_mask=False):
        phi_cmd, alpha_cmd = self.waypoint_controller.get_control(self.env.state, self.env.active_vertex)
        action = np.array([self.wrap_to_interval(phi_cmd, self._params_agent.ACTION_SPACE[0, :] * np.pi/180),
                           self.wrap_to_interval(alpha_cmd, self._params_agent.ACTION_SPACE[1, :] * np.pi/180)])
        return action

    def wrap_to_interval(self, value, source_interval, target_interval=np.array([-1, 1])):
        wrapped_value = np.interp(value, (source_interval.min(), source_interval.max()),
                                  (target_interval.min(), target_interval.max()))
        return wrapped_value

class Waypoint_Controller():
    def __init__(self):
        self._params_task = params_triangle_soaring.TaskParameters()
        self._params_glider = params_environment.params_glider()
        self._params_physics = params_environment.params_physics()
        self._params_control = params_vertex_tracker.params_control()


    def get_control(self, state, active_vertex_id):
        phi_cmd     = self.controller_lat(state, active_vertex_id)
        alpha_cmd   = self.controller_lon(phi_cmd)
        return phi_cmd, alpha_cmd

    def controller_lat(self, state, active_vertex_id):
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
        # stretching triangle slightly ensures hitting the sectors (especially vertex #2)
        waypoint = self._params_control.STRETCH * self._params_task.TRIANGLE[:, (active_vertex_id - 1)]
        g_los = waypoint - position[0:2]
        chi_cmd = np.arctan2(g_los[1], g_los[0])
        return chi_cmd

    def controller_lon(self, phi):
        alpha_bestGlide = ((self._params_glider.ST + 2)
                           * np.sqrt(self._params_glider.CD0 * self._params_glider.OE / self._params_glider.ST)) \
                          / (2 * np.sqrt(np.pi))
        alpha_turn = -(self._params_physics.G/self._params_glider.Z_ALPHA) * (1/np.cos(phi) - 1)
        alpha_cmd = alpha_bestGlide + alpha_turn

        alpha_cmd = np.clip(alpha_cmd, -self._params_control.AOA_MIN, self._params_control.AOA_MAX)

        return alpha_cmd