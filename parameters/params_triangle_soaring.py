import numpy as np


class params_task:
    def __init__(self):
        # triangle orientation: alpha - pi/2 ('alpha' referred to in GPS triangle regulations)
        self.ORIENTATION    = 0
        # transformation matrix from triangle-coordinates to local NE-coordinates
        self.G_T_T          = np.transpose(np.array([[np.cos(self.ORIENTATION), np.sin(self.ORIENTATION)],
                                                     [-np.sin(self.ORIENTATION), np.cos(self.ORIENTATION)]]))
        # local NE-coordinates of the triangle vertices (2 x 3)
        self.TRIANGLE       = self.G_T_T @ np.array([[0., 350, 0.],
                                                     [350, 0., -350]])
        # local NE-coordinates of the triangle origin, i.e. finish line (2 x 1)
        self.FINISH_LINE    =  self.TRIANGLE[:, 2] + (self.TRIANGLE[:, 0] - self.TRIANGLE[:, 2]) / 2

        # rotation matrix from triangle-coordinates to sector-one-coordinates (translation needed for transformation)
        self.ONE_T_T        = np.array([[np.cos(3 * np.pi / 8), np.sin(3 * np.pi / 8)],
                                        [-np.sin(3 * np.pi / 8), np.cos(3 * np.pi / 8)]])
        # rotation matrix from triangle-coordinates to sector-two-coordinates (translation needed for transformation)
        self.TWO_T_T        = np.array([[np.cos(-np.pi / 4), np.sin(-np.pi / 4)],
                                        [-np.sin(-np.pi / 4), np.cos(-np.pi / 4)]])
        # rotation matrix from triangle-coordinates to sector-three-coordinates (translation needed for transformation)
        self.THREE_T_T      = np.array([[np.cos(9 * np.pi / 8), np.sin(9 * np.pi / 8)],
                                        [-np.sin(9 * np.pi / 8), np.cos(9 * np.pi / 8)]])

        # task (alternatives: 'distance', 'speed')
        self.TASK           = 'distance'
        # working time to have the task done (relevant for 'distance' only)
        self.WORKING_TIME   = 60*30




