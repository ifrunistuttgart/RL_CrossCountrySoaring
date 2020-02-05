import gym
import glider
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import normal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
import os
import sys
import warnings

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
    fig = plt.figure()
    fig.set_size_inches(11.69, 8.27)  # DinA4
    fig.suptitle("Sample after {} episodes of training\nVertices hit: {}, Score: {:.1f}"
                 .format(n_epi, (env.lapCounter*3 + env.vertexCounter), ret))

    grid = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)
    ax1 = fig.add_subplot(grid[0:2, 0])
    ax2 = fig.add_subplot(grid[-1, :])
    ax3 = fig.add_subplot(grid[0, 1])
    ax4 = fig.add_subplot(grid[1, 1])

    ax1.plot(np.append(_params_task.TRIANGLE[1, :], _params_task.TRIANGLE[1, 0]),
             np.append(_params_task.TRIANGLE[0, :], _params_task.TRIANGLE[0, 0]), 'r-')
    updraft_position = env._wind_fun._wind_data['updraft_position']
    ax1.plot(updraft_position[1, :], updraft_position[0, :], 'b+')
    ax1.plot(np.array(pos_list)[:, 1], np.array(pos_list)[:, 0])
    ax1.set_xlim(-_params_task.DISTANCE_MAX, _params_task.DISTANCE_MAX)
    ax1.set_ylim(-_params_task.DISTANCE_MAX, _params_task.DISTANCE_MAX)
    ax1.set(aspect='equal')
    ax1.set_xlabel("east (m)")
    ax1.set_ylabel("north (m)")
    ax1.grid(True)

    ax2.plot(np.linspace(_params_sim.TIMESTEP, time, len(pos_list)), -np.array(pos_list)[:, 2])
    ax2.set_xlim(0, _params_task.WORKING_TIME/2)
    ax2.set_ylim(0, (_params_task.INITIAL_STATE[2]*(-1) + 5*_params_task.INITIAL_STD[2]))
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("height (m)")
    ax2.grid(True)

    ax3.plot(np.linspace(_params_sim.TIMESTEP, time, len(ctrl_list)), (180 / np.pi) * np.array(ctrl_list)[:, 0])
    ax3.set_xlim(0, _params_task.WORKING_TIME/2)
    ax3.set_ylim(_params_task.ACTION_SPACE[0, 0], _params_task.ACTION_SPACE[0, 1])
    ax3.set_xticklabels([])
    ax3.set_ylabel("mu (deg)")
    ax3.grid(True)

    ax4.plot(np.linspace(_params_sim.TIMESTEP, time, len(ctrl_list)), (180 / np.pi) * np.array(ctrl_list)[:, 1])
    ax4.set_xlim(0, _params_task.WORKING_TIME / 2)
    ax4.set_ylim(_params_task.ACTION_SPACE[1, 0], _params_task.ACTION_SPACE[1, 1])
    ax4.set_xlabel("time (s)")
    ax4.set_ylabel("alpha (deg)")
    ax4.grid(True)
    ax4.get_shared_x_axes().join(ax4, ax3)

    warnings.filterwarnings("ignore", category=UserWarning, module="backend_interagg")
    grid.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    plt.savefig("resultant_trajectory_episode_{}".format(n_epi) + ".png", dpi=400)
    plt.show()

    env.close()

if __name__ == '__main__':
    main(env, model, n_epi, validation)

