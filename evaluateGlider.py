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

from params import *

def main(env, policy, n_epi, validation=False):

    policy.actor.Net.train()

    state = env.reset()
    observation = env.standardize_observations(state)
    done = False

    pos_list = [[state[1], state[2], state[3]]]
    act_list = []

    _params_rl = params_rl()
    _params_task = params_task()
    _params_sim = params_sim()

    while not done:

        # evaluate and apply policy
        action, _ = policy.act(torch.from_numpy(observation).float(), validation)
        action = action.cpu().data.numpy().flatten()
        observation, r, done, info = env.step(action)

        # scale to actual state
        state = observation * env._params_task.OBS_STD + env._params_task.OBS_MEAN
        # state[0] = env._params_task.DISTANCE - state[0]
        pos_list.append([state[1], state[2], state[3]])

        # scale to actual AoA
        action = env.scale_action(action)
        act_list.append([action[0], action[1]])

    time = state[0]

    # plot position and control trajectory
    plt.figure("resultant trajectory")
    plt.subplot(211)
    plt.plot(np.array(pos_list)[:, 1], np.array(pos_list)[:, 0])
    # plt.xlim(0, _params_task.DISTANCE)
    plt.xlabel("east (m)")
    plt.ylabel("north (m)")
    # plt.ylim(0, (_params_task.INITIAL_STATE[1]*(-1) + _params_task.INITIAL_STD[1]))
    plt.title("Sample after {} episodes of training: \n Episode ended after {:.2f} seconds"
              .format(n_epi, time))
    plt.grid(True)

    ax1 = plt.subplot(212)
    ax2 = ax1.twinx()
    ax1.plot(np.linspace[_params_sim.TIMESTEP, time], (180/np.pi)*np.array(act_list[:, 0]))
    ax2.plot(np.linspace[_params_sim.TIMESTEP, time], (180 / np.pi) * np.array(act_list[:, 1]))

    plt.xlim(0, _params_task.WORKING_TIME)
    ax1.ylim(_params_task.ACTION_SPACE[0, 0], _params_task.ACTION_SPACE[0, 1])
    ax2.ylim(_params_task.ACTION_SPACE[1, 0], _params_task.ACTION_SPACE[1, 1])

    plt.xlabel("time (s)")
    ax1.ylabel("mu (deg)")
    ax2.ylabel("alpha (deg)")
    plt.grid(True)


    plt.savefig("resultant_trajectory_episode_{}".format(n_epi) + ".png")
    plt.show()

    env.close()

if __name__ == '__main__':
    main(env, model, n_epi, validation)

