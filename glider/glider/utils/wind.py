import numpy as np
import sys
import os

sys.path.append(os.path.join("..", "..", ".."))
from params_3D import *

class Wind(object):
    def __init__(self):

        # instantiate parameters
        self._params_task = params_task()
        self._params_wind = params_wind()

        # set wind data
        self._wind_data = np.nan
        self.reset_wind()

        # set updraft shape factors
        self._r1r2shape = np.array([0.1400, 0.2500, 0.3600, 0.4700, 0.5800, 0.6900, 0.8000])
        self._kShape    = np.array([[1.5352, 2.5826, -0.0113, 0.0008],
                                   [1.5265, 3.6054, -0.0176, 0.0005],
                                   [1.4866, 4.8356, -0.0320, 0.0001],
                                   [1.2042, 7.7904, 0.0848, 0.0001],
                                   [0.8816, 13.9720, 0.3404, 0.0001],
                                   [0.7067, 23.9940, 0.5689, 0.0002],
                                   [0.6189, 42.7965, 0.7157, 0.0001]])

    @staticmethod  # TODO: check that method
    def _init_headwind(params_wind):

        if params_wind.APPLY_HEADWIND:
            headwind = np.random.uniform(params_wind.HEADWIND_LOW, params_wind.HEADWIND_HIGH)
            headwind_data = dict(
                headwind_velocity=headwind[0],
                headwind_direction=headwind[1]
            )
        else:
            headwind_data = dict(
                headwind_velocity=0.,
                headwind_direction=np.nan
            )

        return headwind_data

    @staticmethod  # TODO: check
    def _init_updrafts(params_wind, params_task):
        """ init updraft model according to M. Allen:
        Updraft Model for Development of Autonomous Soaring Uninhabited Air Vehicles """

        if params_wind.APPLY_UPDRAFTS:
            # Sample number of updrafts
            updraft_count = int(np.random.uniform(0, params_wind.UPCOUNT_MAX + 1))

            # set updraft positions (north, east) TODO: make sure that updrafts do no overlap
            updraft_position = np.zeros((2, updraft_count))
            for i in range(0, updraft_count):
                distance = np.random.uniform(0., params_task.DISTANCE_MAX)
                polarAngle = np.random.uniform(-np.pi, np.pi)
                updraft_position[:, i] = distance*np.array([np.cos(polarAngle), np.sin(polarAngle)])

            # set perturbation multipliers
            updraft_wgain = np.random.normal(np.ones(updraft_count), params_wind.WGAIN_STD * np.ones(updraft_count))
            updraft_rgain = np.random.normal(np.ones(updraft_count), params_wind.RGAIN_STD * np.ones(updraft_count))

            updraft_data = dict(
                updraft_count=updraft_count,
                updraft_position=updraft_position,
                updraft_wgain=updraft_wgain,
                updraft_rgain=updraft_rgain
            )
        else:
            updraft_data = dict(
                updraft_count=0,
                updraft_position=np.nan,
                updraft_wgain=np.nan,
                updraft_rgain=np.nan
            )

        return updraft_data

########################################################################################################################
    """ Setter and Getter:"""
    def reset_wind(self):
        headwind_data = self._init_headwind(self._params_wind)
        updraft_data = self._init_updrafts(self._params_wind, self._params_task)
        self._wind_data = {**headwind_data, **updraft_data}

    def get_wind_data(self):
        return self._wind_data

    def set_wind_data(self, wind_data):
        self._wind_data = wind_data

    def store_wind_data(self, wind_data):
        self._stored_wind_data = wind_data

    def get_current_wind(self, position):
        wind = np.zeros(3).reshape(3, 1)
        position = position.reshape(3, 1)
        if self._params_wind.APPLY_HEADWIND:
            wind[0:2] = self._wind_data['headwind_velocity'] *\
                        np.array([[np.cos(self._wind_data['headwind_direction'])],
                                  [np.sin(self._wind_data['headwind_direction'])]])
        if self._params_wind.APPLY_UPDRAFTS and self._wind_data['updraft_count'] and (-position[2] > 0.1):
            wind[2] = -self.get_current_updraft(position)

        return wind

    def get_current_updraft(self, position=None):
        """ get current updraft according to M. Allen's updraft model:
                Updraft Model for Development of Autonomous Soaring Uninhabited Air Vehicles """

        # assign updraft data
        updraft_count = int(self._wind_data['updraft_count'])
        updraft_position = self._wind_data['updraft_position']
        updraft_wgain = self._wind_data['updraft_wgain']
        updraft_rgain = self._wind_data['updraft_rgain']

        # calculate distance to each updraft
        dist = []
        for k in range(0, updraft_count):
            r_k = updraft_position[:, k].reshape(2, 1) - position[0:2]
            dist = np.append(dist, np.linalg.norm(r_k))

        # calculate average updraft size
        zzi = -position[2] / self._params_wind.ZI
        rbar = (.102 * np.power(zzi, (1 / 3))) * (1 - (.25 * zzi)) * self._params_wind.ZI

        # calculate average updraft strength
        wtbar = np.power(zzi, (1 / 3)) * (1 - 1.1 * zzi) * self._params_wind.WSTAR

        # find nearest updraft
        upused = np.argmin(dist)

        # multiply by random perturbation gain
        r2 = rbar * updraft_rgain[upused]

        # calculate inner and outer radios of rotated trapezoid updraft
        if r2 < 10:
            r2 = 10
        if r2 < 600:
            r1r2 = .0011 * r2 + .14
        else:
            r1r2 = .8
        r1 = r1r2 * r2

        # multiply average updraft strength by wgain for this updraft
        wt = wtbar * updraft_wgain[upused]

        # calculate strength at center of rotated trapezoid updraft
        wc = (3 * wt * (np.power(r2, 3) - np.power(r2, 2) * r1)) / (np.power(r2, 3) - np.power(r1, 3))

        # calculate updraft velocity
        r = dist[upused]
        rr2 = r / r2
        if -position[2] < self._params_wind.ZI:
            ka, kb, kc, kd = self.get_updraft_shape(rr2)  # get shape coefficients
            ws = 1. / (1 + np.power(ka * np.abs(rr2 + kc), kb))\
                 + kd * rr2  # calculate smooth vertical velocity distribution
            ws = np.maximum(0, ws)  # no negative updrafts
        else:
            ws = 0

        # calculate downdraft velocity at the edge of the updraft
        if dist[upused] > r1 and rr2 < 2:
            w1 = (np.pi / 6) * np.sin(np.pi * rr2)  # downdraft, positive up
        else:
            w1 = 0
        if .5 < zzi <= .9:
            swd = 2.5 * (zzi - .5)  # scale factor for wd for zzi, used again later
            wd = swd * w1
            wd = np.minimum(wd, 0)
        else:
            swd = 0
            wd = 0
        w2 = ws * wc + wd * wt  # scale updraft to actual velocity

        # calculate environment sink velocity
        At = np.pi * (rbar ** 2) * updraft_count  # total area taken by updrafts
        A = np.pi * (self._params_task.DISTANCE_MAX ** 2)  # total area of interest
        if At > A:
            print('Area of test space is too small')
            raise ArithmeticError

        we = -(At * wtbar * (1 - swd)) / (A - At)  # environment sink rate
        we = np.minimum(we, 0)  # don't allow positive sink

        # stretch updraft to blend with sink at edge
        if dist[upused] > r1:  # if you are outside the core
            w = w2 * (1 - we / wc) + we  # stretch
        else:
            w = w2

        return w

    def get_updraft_shape(self, r1r2=None):
        r1r2shape = self._r1r2shape
        kShape = self._kShape

        if r1r2 < .5 * (r1r2shape[0] + r1r2shape[1]):  # pick shape
            ka = kShape[0, 0]
            kb = kShape[0, 1]
            kc = kShape[0, 2]
            kd = kShape[0, 3]
        elif r1r2 < .5 * (r1r2shape[1] + r1r2shape[2]):
            ka = kShape[1, 0]
            kb = kShape[1, 1]
            kc = kShape[1, 2]
            kd = kShape[1, 3]
        elif r1r2 < .5 * (r1r2shape[2] + r1r2shape[3]):
            ka = kShape[2, 0]
            kb = kShape[2, 1]
            kc = kShape[2, 2]
            kd = kShape[2, 3]
        elif r1r2 < .5 * (r1r2shape[3] + r1r2shape[4]):
            ka = kShape[3, 0]
            kb = kShape[3, 1]
            kc = kShape[3, 2]
            kd = kShape[3, 3]
        elif r1r2 < .5 * (r1r2shape[4] + r1r2shape[5]):
            ka = kShape[4, 0]
            kb = kShape[4, 1]
            kc = kShape[4, 2]
            kd = kShape[4, 3]
        elif r1r2 < .5 * (r1r2shape[5] + r1r2shape[6]):
            ka = kShape[5, 0]
            kb = kShape[5, 1]
            kc = kShape[5, 2]
            kd = kShape[5, 3]
        else:
            ka = kShape[6, 0]
            kb = kShape[6, 1]
            kc = kShape[6, 2]
            kd = kShape[6, 3]

        return ka, kb, kc, kd