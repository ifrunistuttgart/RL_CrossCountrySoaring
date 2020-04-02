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
from matplotlib import cm

import datetime
import os
import sys
import warnings

from params import *

def main(env, ppo, n_epi, validation=False):

    ppo.policy.eval()

    state = env.reset(p_loiter=0)
    observation = env.state2observation()
    rel_updraft_pos, _ = env.get_rel_updraft_positions()
    done = False
    ret = 0

    pos_list = [[state[0], state[1], state[2]]]
    ctrl_list = []

    _params_rl = params_rl()
    _params_task = params_task()
    _params_sim = params_sim()
    _params_wind = params_wind()

    while not done:

        # evaluate and apply policy
        action = ppo.select_action(observation, rel_updraft_pos)
        observation, r, done, info = env.step(action)
        rel_updraft_pos, _ = env.get_rel_updraft_positions()
        ret += r

        # write to lists
        pos_list.append([env.state[0], env.state[1], env.state[2]])
        control = env.action2control(action)
        ctrl_list.append([control[0], control[1]])

    time = env.time

    # plot position and control trajectory
    fig = plt.figure()
    fig.set_size_inches(11.69, 8.27)  # DinA4
    fig.suptitle("Sample after {} episodes of training:\nVertices hit: {}, Return: {:.1f}, Score: {}"
                 .format(n_epi, (env.lapCounter*3 + env.vertexCounter), ret, env.lapCounter*200))

    grid = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)
    ax1 = fig.add_subplot(grid[0:2, 0])
    ax2 = fig.add_subplot(grid[-1, :])
    ax3 = fig.add_subplot(grid[0, 1])
    ax4 = fig.add_subplot(grid[1, 1])

    timeVec= np.linspace(_params_sim.TIMESTEP, time, len(pos_list))
    colormap = cm.spring(timeVec / timeVec.max())

    ax1.plot(np.append(_params_task.TRIANGLE[1, :], _params_task.TRIANGLE[1, 0]),
             np.append(_params_task.TRIANGLE[0, :], _params_task.TRIANGLE[0, 0]), 'r-')

    # plot updrafts
    updraft_position = env._wind_fun._wind_data['updraft_position']
    if not np.isnan(updraft_position).any():
        ax1.plot(updraft_position[1, :], updraft_position[0, :], 'b+')
        for k in range(len(updraft_position[0])):
            updraft_outline = plt.Circle((updraft_position[1, k], updraft_position[0, k]), _params_wind.DELTA,
                                         color='b', fill=False)
            ax1.add_artist(updraft_outline)

    # plot solid north-east trajectory with color gradient
    for k in range(len(colormap)):
        x_segm = np.array(pos_list)[k:(k + 2), 1]
        y_segm = np.array(pos_list)[k:(k + 2), 0]
        c_segm = colormap[k]
        ax1.plot(x_segm, y_segm, c=c_segm)

    ax1.set_xlim(-_params_task.DISTANCE_MAX, _params_task.DISTANCE_MAX)
    ax1.set_ylim(-_params_task.DISTANCE_MAX, _params_task.DISTANCE_MAX)
    ax1.set_xlabel("east (m)")
    ax1.set_ylabel("north (m)")
    ax1.grid(True)

    # plot solid height trajectory with color gradient
    for k in range(len(colormap)):
        x_segm = timeVec[k:(k + 2)]
        y_segm = -np.array(pos_list)[k:(k + 2), 2]
        c_segm = colormap[k]
        ax2.plot(x_segm, y_segm, c=c_segm)

    ax2.set_xlim(0, _params_task.WORKING_TIME)
    ax2.set_ylim(0, _params_task.HEIGHT_MAX)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("height (m)")
    ax2.grid(True)

    ax3.scatter(timeVec[1:], (180 / np.pi) * np.array(ctrl_list)[:, 0],  s=2, c=colormap[1:], edgecolor='none')
    ax3.set_xlim(0, _params_task.WORKING_TIME)
    ax3.set_ylim(_params_task.ACTION_SPACE[0, 0], _params_task.ACTION_SPACE[0, 1])
    ax3.set_xticklabels([])
    ax3.set_ylabel("mu (deg)")
    ax3.grid(True)

    ax4.scatter(timeVec[1:], (180 / np.pi) * np.array(ctrl_list)[:, 1], s=2, c=colormap[1:], edgecolor='none')
    ax4.set_xlim(0, _params_task.WORKING_TIME)
    ax4.set_ylim(_params_task.ACTION_SPACE[1, 0], _params_task.ACTION_SPACE[1, 1])
    ax4.set_xlabel("time (s)")
    ax4.set_ylabel("alpha (deg)")
    ax4.grid(True)
    ax4.get_shared_x_axes().join(ax4, ax3)

    warnings.filterwarnings("ignore", category=UserWarning, module="backend_interagg")
    grid.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    plt.savefig("resultant_trajectory_episode_{}".format(n_epi) + ".png", dpi=400)
    plt.show()

    plt.close(fig)
    env.close()

if __name__ == '__main__':
    main(env, model, n_epi, validation)
