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
from trainGlider_ppo import Memory

def main(env, policy, n_epi, validation=False):
# def main():
#     env = gym.make('glider2D-v1')
    memory = Memory()
    # model = torch.load("./experiments/PPO_2D_noWind_25102019/actor_critic_network_final_25-October-2019_08-55.pt")
    policy.eval()

    state = env.reset()
    observation = env.standardize_observations(state)
    done = False
    time = 0
    pos_list = [[state[0], -state[1]]]
    act_list = []

    _params_rl = params_rl()
    _params_task = params_task()

    while not done:

        # print(state)

        # evaluate and apply policy
        action, _ = policy.act(torch.from_numpy(observation).float(), validation)
        observation, r, done, info = env.step(action[0].numpy())

        # scale to actual state
        state = observation * env._params_task.OBS_STD + env._params_task.OBS_MEAN
        state[0] = env._params_task.DISTANCE - state[0]
        pos_list.append([state[0], -state[1]])

        # scale to actual AoA
        action = env.scale_action(action.detach().numpy())
        act_list.append([action])

        time += env._params_sim.TIMESTEP

    # plot position and control trajectory
    plt.figure("resultant trajectory")
    plt.subplot(211)
    plt.plot(np.array(pos_list)[:, 0], np.array(pos_list)[:, 1])
    plt.xlim(0, _params_task.DISTANCE)
    plt.ylabel("height (m)")
    plt.ylim(0, (_params_task.INITIAL_STATE[1]*(-1) + _params_task.INITIAL_STD[1]))
    plt.title("after {} episodes of training: \n test finished after {:.2f} seconds"
              .format(n_epi, time))
    plt.grid(True)

    plt.subplot(212)
    plt.plot(np.array(pos_list)[0:-1, 0], (180/np.pi)*np.array(act_list).reshape(len(act_list), 1))
    plt.xlabel("distance (m)")
    plt.xlim(0, _params_task.DISTANCE)
    plt.ylabel("alpha (deg)")
    plt.ylim(_params_task.ACTION_SPACE[0], _params_task.ACTION_SPACE[1])
    plt.grid(True)

    plt.savefig("resultant_trajectory_episode_{}".format(n_epi) + ".png")

    plt.show()

    env.close()

if __name__ == '__main__':
    main(env, model, n_epi, validation)
    # main()
