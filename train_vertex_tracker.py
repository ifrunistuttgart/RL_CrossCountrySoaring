import datetime
import time
import os
import shutil
import gym
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset

import evaluate_vertex_tracker
from parameters import params_environment, params_triangle_soaring
from subtasks.vertex_tracker import params_vertex_tracker
from subtasks.vertex_tracker import model_vertex_tracker

device = torch.device("cuda:0")
# device = torch.device("cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.observations[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class PPO:
    def __init__(self):
        # instantiate parameters
        self._params_rl = params_vertex_tracker.params_rl()

        # instantiate actor-critic model
        self.policy = model_vertex_tracker.ActorCritic().to(device)
        # self.policy.load_state_dict(torch.load("actor_critic_network_final_01-April-2020_14-58.pt"))
        self.policy_old = model_vertex_tracker.ActorCritic().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # setup optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=self._params_rl.LEARNING_RATE, betas=(0.9, 0.999))
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, observation, memory=None, validation_mask=False):
        return self.policy_old.act(torch.FloatTensor(observation).to(device), memory, validation_mask).cpu().data.numpy().flatten()

    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        returns_to_go = []
        # discounted_reward = 0
        discounted_reward = self.policy_old.critic(memory.observations[-1])
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self._params_rl.GAMMA * discounted_reward)
            returns_to_go.insert(0, discounted_reward)

        # Normalizing the rewards:
        returns_to_go = torch.tensor(returns_to_go).to(device)
        returns_to_go = (returns_to_go - returns_to_go.mean()) / (returns_to_go.std() + 1e-5)
        returns_to_go = returns_to_go.unsqueeze(1)
        
        # convert list to tensor
        old_actions = torch.squeeze(torch.stack(memory.actions), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).detach()
        old_observations = torch.squeeze(torch.stack(memory.observations), 1).detach()


        # put batch to data_loader
        train_loader = DataLoader(dataset=TensorDataset(old_actions, old_logprobs, old_observations, returns_to_go),
                                  batch_size=self._params_rl.MINIBATCHSIZE, shuffle=True)
        
        # Optimize policy for K epochs:
        for _ in range(self._params_rl.K_EPOCH):
            for mini_batch in train_loader:
                # get sampled data in mini-batch and send them to device
                old_actions, old_logprobs, old_observations, returns_to_go = mini_batch

                # evaluate old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_observations, old_actions)

                # ppo ratio
                ratios = torch.exp(logprobs - old_logprobs)

                # surrogate loss
                advantages = returns_to_go - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self._params_rl.EPS_CLIP, 1 + self._params_rl.EPS_CLIP) * advantages
                loss = -torch.min(surr1, surr2).mean()\
                       + 0.5*self.MseLoss(state_values, returns_to_go)\
                       - 0.01*dist_entropy

                # gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
            
        # transfer weights
        self.policy_old.load_state_dict(self.policy.state_dict())

def main():
    # set up training
    tstart = time.time()
    env = gym.make('glider3D-v0', agent='vertex_tracker')
    ppo = PPO()
    memory = Memory()

    # load parameters
    _params_rl      = params_vertex_tracker.params_rl()
    _params_agent   = params_vertex_tracker.params_agent()
    _params_logging = params_vertex_tracker.params_logging()

    # create folder to store data for the experiment running
    experimentID = 1
    dirName = "{}_vertex_tracker_experiment_running".format(experimentID)
    while os.path.exists(dirName):
        experimentID += 1
        dirName = "{}_".format(experimentID) + dirName.split('_', 1)[1]
    os.mkdir(dirName)
    shutil.copytree(os.getcwd(), os.path.join(dirName, "Sources_unzipped"),
                    ignore=shutil.ignore_patterns('*experiment*', 'archive', 'tests', '.git*', 'rl_ccs_experiments',
                                                  '.idea', '__pycache__', 'README*'))
    os.chdir(dirName)
    shutil.make_archive("Sources", 'zip', "Sources_unzipped")
    shutil.rmtree("Sources_unzipped")
    print("Directory for running experiment no. {} created".format(experimentID))

    # save parameters to file
    parameterFile = open("parameterFile.txt", "w")
    parameterFile.write(
        format(vars(_params_rl)) + "\n" +
        format(vars(params_vertex_tracker.params_model())) + "\n" +
        format(vars(_params_agent)) + "\n" +
        format(vars(_params_logging)) + "\n\n" +
        format(vars(params_triangle_soaring.params_task())) + "\n\n" +
        format(vars(params_environment.params_sim())) + "\n" +
        format(vars(params_environment.params_glider())) + "\n" +
        format(vars(params_environment.params_physics())) + "\n" +
        format(vars(params_environment.params_wind())))
    parameterFile.close()

    # set random seed
    if _params_rl.SEED:
        print("Random Seed: {}".format(_params_rl.SEED))
        torch.manual_seed(_params_rl.SEED)
        env.seed(_params_rl.SEED)
        np.random.seed(_params_rl.SEED)

    # set up file to save average returns
    returnFile = open("returnFile_running.dat", "w")
    returnFile.write("episodes, policy iterations, avg. return over {} episodes\n"
                     .format(_params_logging.PRINT_INTERVAL))
    returnFile.close()

    # initialize logging variables
    returns = []
    average_returns = []
    policy_iterations = 0
    timestep_counter = 0

    # showcase behavior before training takes place
    evaluate_vertex_tracker.main(env, ppo, 0, _params_agent, validation_mask=False)
    
    # training loop
    for n_epi in range(1, int(_params_rl.N_EPISODES + 1)):
        env.reset()
        done = False
        ret = 0
        observation = env.get_observation()

        while not done:
            timestep_counter += 1

            # run policy_old
            action = ppo.select_action(observation, memory)
            _, reward, done, _ = env.step(action)
            observation = env.get_observation()
            ret += reward

            # store reward and done flag:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update policy every N_TIMESTEPS
            if timestep_counter % _params_rl.BATCHSIZE == 0:
                ppo.policy.train()
                ppo.update(memory)
                ppo.policy.eval()
                memory.clear_memory()
                timestep_counter = 0
                policy_iterations += 1

            # stop rollout if episode is completed
            if done:
                returns.append(ret)
                break

        n_mean = _params_logging.PRINT_INTERVAL if len(returns) >= _params_logging.PRINT_INTERVAL else len(returns)
        average_returns.append(np.convolve(returns[-n_mean:], np.ones((n_mean,)) / n_mean, mode='valid')[0])

        if n_epi % _params_logging.PRINT_INTERVAL == 0:
            print("# episode: {}/{} \t\t avg return over last {} episodes: {:.1f}"
                  .format(n_epi, int(_params_rl.N_EPISODES),
                          _params_logging.PRINT_INTERVAL, average_returns[-1]))

            with open("returnFile_running.dat", "a+") as returnFile:
                returnFile.write(format(n_epi) + "," + format(policy_iterations) + ","
                                 + '{:.1f}'.format(average_returns[-1]) + "\n")

        if n_epi % _params_logging.SAVE_INTERVAL == 0:
            torch.save(ppo.policy.state_dict(), "updraft_exploiter_actor_critic_episode_{}".format(n_epi) + ".pt")
            evaluate_vertex_tracker.main(env, ppo, n_epi, _params_agent)

    # display results
    print('Duration: %.2f s' % (time.time() - tstart))
    evaluate_vertex_tracker.main(env, ppo, "final", _params_agent, validation_mask=True)
    now = datetime.datetime.now()
    plt.figure("average returns")
    plt.plot(average_returns)
    plt.ylabel("average returns (-)")
    plt.xlabel("episodes (-)")
    plt.grid(True)
    plt.savefig("average_returns_" + now.strftime("%d-%B-%Y_%H-%M") + ".png")
    plt.show()

    # save actor-critic
    torch.save(ppo.policy.state_dict(),
               "updraft_exploiter_actor_critic_final_" + now.strftime("%d-%B-%Y_%H-%M") + ".pt")

    # rename parameter file consistently
    os.rename(parameterFile.name, "parameters_" + now.strftime("%d-%B-%Y_%H-%M") + ".txt")

    # rename return file consistently
    returnFile.close()
    os.rename(returnFile.name, "average_returns_" + now.strftime("%d-%B-%Y_%H-%M") + ".dat")

    env.close()

if __name__ == '__main__':
    main()