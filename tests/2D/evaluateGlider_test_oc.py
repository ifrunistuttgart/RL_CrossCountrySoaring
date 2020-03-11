import gym
import glider
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import normal
import matplotlib.pyplot as plt
import datetime
import os
import sys

import scipy.io

from params import *
# from trainGlider_ppo import Memory

def main():
    env = gym.make('glider2D-v1')

    state = env.reset()
    # observation = env.standardize_observations(state)
    done = False
    time = 0
    ret = 0
    pos_list = [[state[0], -state[1]]]
    vel_list = [[state[2], state[3]]]
    act_list = []

    _params_rl = params_rl()
    _params_task = params_task()

    ocp = scipy.io.loadmat('opti_19_11_21_22_26.mat')
    opti = ocp['opti']
    totalTime = opti[0]
    dt = totalTime / (opti.size - 2)
    index = 1

    while (time < totalTime) and not done:

        # evaluate and apply policy
        action = rescale_action(min(_params_task.ACTION_SPACE)*(np.pi/180),
                                max(_params_task.ACTION_SPACE)*(np.pi/180), torch.tensor(opti[index]))

        observation, r, done, info = env.step(action[0].numpy(), dt)

        # scale to actual state
        state = observation * env._params_task.OBS_STD + env._params_task.OBS_MEAN
        state[0] = env._params_task.DISTANCE - state[0]
        pos_list.append([state[0], -state[1]])
        vel_list.append([state[2], state[3]])

        # scale to actual AoA
        action = env.scale_action(action.detach().numpy())
        act_list.append([action])

        time += float(dt)
        ret += r
        index += 1

    # plot position and control trajectory
    plt.figure("resultant trajectory")
    plt.subplot(211)
    plt.plot(np.array(pos_list)[:, 0], np.array(pos_list)[:, 1])
    plt.plot(np.array(pos_list)[:, 0], np.array(vel_list)[:, 0])
    plt.xlim(0, _params_task.DISTANCE)
    plt.ylabel("height (m), vx (m/s)")
    # plt.ylim(0, (_params_task.INITIAL_STATE[1]*(-1) + _params_task.INITIAL_STD[1]))
    plt.title("Episode ended after {:.2f} seconds".format(time))
    plt.grid(True)

    plt.subplot(212)
    plt.plot(np.array(pos_list)[0:-1, 0], (180/np.pi)*np.array(act_list).reshape(len(act_list), 1))
    plt.xlabel("distance (m)")
    plt.xlim(0, _params_task.DISTANCE)
    plt.ylabel("alpha (deg)")
    plt.ylim(_params_task.ACTION_SPACE[0], _params_task.ACTION_SPACE[1])
    plt.grid(True)

    plt.savefig("ocp_solution.png")
    plt.show()

    print("return: {}".format(ret))
    env.close()

def rescale_action(lb, ub, action):
    rescaled_action = (2*action - ub - lb)/(ub - lb)
    return rescaled_action

if __name__ == '__main__':
    main()