import numpy as np

from parameters import params_triangle_soaring, params_environment

class Waypoint_Controller():
    def __init__(self):
        self._params_task = params_triangle_soaring.params_task()
        self._params_glider = params_environment.params_glider()
        self._params_physics = params_environment.params_physics()

        self._K_CHI = 5 * 2
        self._Z_ALPHA = -155

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
            phi_cmd = chi_error * self._K_CHI / 10
        else:
            phi_cmd = chi_error * self._K_CHI / speed

        return phi_cmd

    def guidance(self, position, active_vertex_id):
        g_aircraft2waypoint = self._params_task.TRIANGLE[:, (active_vertex_id - 1)] - position[0:2]
        chi_cmd = np.arctan2(g_aircraft2waypoint[1], g_aircraft2waypoint[0])
        return chi_cmd

    def controller_lon(self, phi):
        alpha_bestGlide = ((self._params_glider.ST + 2)
                           * np.sqrt(self._params_glider.CD0 * self._params_glider.OE / self._params_glider.ST)) \
                          / (2 * np.sqrt(np.pi))
        alpha_turn = 0
        # alpha_turn = (self._params_physics.G/self._Z_ALPHA) * (1/np.cos(phi) - 1)
        alpha_cmd = alpha_bestGlide + alpha_turn
        return alpha_cmd
