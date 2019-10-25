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

# from trainGlider_ppo import PPO
def main(model):
# def main():
    env = gym.make('glider-v0')
    # model = torch.load("./experiments/PPO_2D_noWind_25102019/actor_critic_network_final_25-October-2019_08-55.pt")
    model.eval()

    state = env.reset()
    observation = env.standardize_observations(state)
    done = False
    time = 0
    pos_list = [[state[0], -state[1]]]
    act_list = []

    _params_rl = params_rl()

    while not done:

        # print(state)

        # evaluate and apply policy
        pi = model.actor(torch.from_numpy(observation).float())

        pdf = normal.Normal(pi[0], 0)
        # pdf = normal.Normal(pi[0], _params_rl.SIGMA)
        # pdf = normal.Normal(pi[0], torch.exp(pi[1]))
        action = pdf.sample()
        observation, reward, done, info = env.step(action.numpy())

        # scale to actual state
        state = observation * env._params_task.OBS_STD + env._params_task.OBS_MEAN
        state[0] = env._params_task.DISTANCE - state[0]
        pos_list.append([state[0], -state[1]])

        # scale to actual AoA
        action = env.scale_action(action.detach().numpy())
        act_list.append([action])

        time += env._params_sim.TIMESTEP

    print("Episode completed after {:.2f} seconds".format(time))

    # plot position and control trajectory
    plt.figure("resultant trajectory")
    plt.subplot(211)
    plt.plot(np.array(pos_list)[:, 0], np.array(pos_list)[:, 1])
    plt.ylabel("height (m)")
    plt.title("episode finished after {:.2f} seconds".format(time))
    plt.grid(True)
    plt.subplot(212)
    plt.plot(np.array(pos_list)[0:-1, 0], (180/np.pi)*np.array(act_list).reshape(len(act_list), 1))
    plt.xlabel("distance (m)")
    plt.ylabel("alpha (deg)")
    plt.grid(True)

    now = datetime.datetime.now()
    plt.savefig("resultant_trajectory" + now.strftime("%d-%B-%Y_%H-%M") + ".png")

    plt.show()

    env.close()

if __name__ == '__main__':
    main(model)
    # main()
