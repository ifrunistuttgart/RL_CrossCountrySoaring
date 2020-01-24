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

from params_3D import *

def main(env, policy, n_epi, validation=False):

    policy.actor.Net.train()

    state = env.reset()
    observation = env.state2observation()
    done = False
    ret = 0

    pos_list = [[state[0], state[1], state[2]]]
    ctrl_list = []

    _params_rl = params_rl()
    _params_task = params_task()
    _params_sim = params_sim()

    while not done:

        # evaluate and apply policy
        action, _ = policy.act(torch.from_numpy(observation).float(), validation)
        action = action.cpu().data.numpy().flatten()
        observation, r, done, info = env.step(action)
        ret += r

        # write to lists
        pos_list.append([env.state[0], env.state[1], env.state[2]])
        control = env.action2control(action)
        ctrl_list.append([control[0], control[1]])


    time = env.time

    # plot position and control trajectory
    fig, (ax1, ax2, ax3l) = plt.subplots(3, 1)

    ax1.plot(np.append(_params_task.TRIANGLE[1, :], _params_task.TRIANGLE[1, 0]),
             np.append(_params_task.TRIANGLE[0, :], _params_task.TRIANGLE[0, 0]), 'r-')
    ax1.plot(np.array(pos_list)[:, 1], np.array(pos_list)[:, 0])
    ax1.set_xlim(-_params_task.DISTANCE_MAX, _params_task.DISTANCE_MAX)
    ax1.set_ylim(-_params_task.DISTANCE_MAX, _params_task.DISTANCE_MAX)
    ax1.axis('equal')
    ax1.set_xlabel("east (m)")
    ax1.set_ylabel("north (m)")
    ax1.grid(True)

    fig.suptitle("Sample after {} episodes of training: \n Vertices hit: {}, Score {:.1f}"
                 .format(n_epi, env.vertexCounter, ret))

    ax2.plot(np.linspace(_params_sim.TIMESTEP, time, len(pos_list)), -np.array(pos_list)[:, 2])
    ax2.set_xlim(0, _params_task.WORKING_TIME/3)
    ax2.set_ylim(0, (_params_task.INITIAL_STATE[2]*(-1) + _params_task.INITIAL_STD[2]))
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("height (m)")
    ax2.grid(True)

    ax3r = ax3l.twinx()
    ax3r.plot(np.linspace(_params_sim.TIMESTEP, time, len(ctrl_list)), (180 / np.pi) * np.array(ctrl_list)[:, 1], 'g-')
    ax3l.plot(np.linspace(_params_sim.TIMESTEP, time, len(ctrl_list)), (180 / np.pi) * np.array(ctrl_list)[:, 0], 'b-')
    ax3l.set_xlim(0, _params_task.WORKING_TIME/2)
    ax3r.set_ylim(_params_task.ACTION_SPACE[1, 0], _params_task.ACTION_SPACE[1, 1])
    ax3l.set_ylim(_params_task.ACTION_SPACE[0, 0], _params_task.ACTION_SPACE[0, 1])
    ax3l.set_xlabel("time (s)")
    ax3r.set_ylabel("alpha (deg)", color='g')
    ax3l.set_ylabel("mu (deg)", color='b')
    ax3l.grid(True)

    plt.savefig("resultant_trajectory_episode_{}".format(n_epi) + ".png")
    plt.show()

    env.close()

if __name__ == '__main__':
    main(env, model, n_epi, validation)

