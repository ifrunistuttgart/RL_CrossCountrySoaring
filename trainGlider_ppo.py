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

import evaluateGlider
from params import *

device = torch.device("cuda:0")
# device = torch.device("cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.up_pos = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.up_pos[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()

        # instantiate parameters
        self._params_model = params_model()
        self._params_rl = params_rl()

        # set model elements
        self.lstm           = nn.LSTM(input_size=2,
                                      hidden_size=(self._params_model.DIM_HIDDEN - self._params_model.DIM_IN),
                                      batch_first=True)
        self.hidden_layer   = nn.Linear(self._params_model.DIM_HIDDEN, self._params_model.DIM_HIDDEN)
        self.out_actor      = nn.Linear(self._params_model.DIM_HIDDEN, self._params_model.DIM_OUT)
        self.out_critic     = nn.Linear(self._params_model.DIM_HIDDEN, 1)

        # set variance for action-sampling
        self.action_var = torch.full((self._params_model.DIM_OUT,),
                                     self._params_rl.SIGMA * self._params_rl.SIGMA).to(device)

    def actor(self, observation, updraft_obs):
        # evaluate lstm
        _, (h_n, c_n) = self.lstm(updraft_obs)

        # forward lstm hidden state for t = seq_len
        x = torch.tanh(h_n[-1])

        # concatenate std. observation & lstm output
        z = torch.cat((observation, x), dim=-1)

        # evaluate feedforward net
        z = self.hidden_layer(z)
        z = torch.tanh(z)
        z = self.out_actor(z)
        z = torch.tanh(z)
        return z

    def critic(self, observation, updraft_obs):
        # evaluate lstm
        _, (h_n, c_n) = self.lstm(updraft_obs)

        # forward lstm hidden state for t = seq_len
        x = torch.tanh(h_n[-1])

        # concatenate std. observation & lstm output
        z = torch.cat((observation, x), dim=-1)

        # evaluate feedforward net
        z = self.hidden_layer(z)
        z = torch.tanh(z)
        z = self.out_critic(z)
        z = torch.tanh(z)
        return z
    
    def act(self, state, rel_updraft_pos, memory):
        action_mean = self.actor(state, rel_updraft_pos)        # get mean action value by evaluating network
        dist = Normal(action_mean, self._params_rl.SIGMA)       # normal distribution around mean action value
        action = dist.sample()                                  # sample from probability distribution
        action_logprob = dist.log_prob(action)

        if memory is not None:
            memory.states.append(state)
            memory.up_pos.append(rel_updraft_pos)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, rel_updraft_pos, action):
        action_mean = self.actor(state, rel_updraft_pos)

        dist = Normal(action_mean, self._params_rl.SIGMA)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state, rel_updraft_pos)

        return action_logprobs, state_value, dist_entropy

class PPO:
    def __init__(self):
        self._params_model = params_model()
        self._params_rl = params_rl()
        
        self.policy = ActorCritic().to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=self._params_rl.LEARNING_RATE, betas=(0.9, 0.999))

        self.policy_old = ActorCritic().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, rel_updraft_pos, memory=None):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)                      # timesteps x observations
        rel_updraft_pos = torch.FloatTensor(rel_updraft_pos).view(1, -1, 2).to(device)  # timesteps x updrafts x features
        return self.policy_old.act(state, rel_updraft_pos, memory).cpu().data.numpy().flatten()

    def pad_updraft_observations(self, up_obs_lst):
        # init tensors for padding
        up_obs_padded = torch.zeros(len(up_obs_lst), max([len(row) for batch in up_obs_lst for row in batch]), 2)
        seq_length = torch.zeros(len(up_obs_lst), dtype=torch.int)

        # loop over batch TODO: surely can be done without looping
        k = 0
        for up_obs in up_obs_lst:
            seq_length[k] = up_obs.size(1)
            up_obs_padded[k, 0:int(seq_length[k]), :] = up_obs
            k += 1

        return up_obs_padded.to(device), seq_length.to(device)
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        returns_to_go = []
        # discounted_reward = 0
        discounted_reward = self.policy_old.critic(memory.states[-1], memory.up_pos[-1])
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
        old_states = torch.squeeze(torch.stack(memory.states)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).detach()

        # pad updraft observations (due to sequences potentially of different length)
        old_up_obs_padded, seq_length = self.pad_updraft_observations(memory.up_pos)

        # put batch to data_loader
        train_loader = DataLoader(dataset=TensorDataset(old_states, old_actions, old_logprobs,
                                                        old_up_obs_padded, seq_length, returns_to_go),
                                  batch_size=self._params_rl.MINIBATCHSIZE, shuffle=True)
        
        # Optimize policy for K epochs:
        for _ in range(self._params_rl.K_EPOCH):
            for mini_batch in train_loader:
                # get sampled data in mini-batch and send them to device
                old_states, old_actions, old_logprobs, old_up_obs_padded, seq_length, returns_to_go = mini_batch

                # pack padded sequence
                old_up_obs_packed = pack_padded_sequence(old_up_obs_padded, seq_length,
                                                         batch_first=True, enforce_sorted=False)

                # evaluate old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_up_obs_packed, old_actions)

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
    env = gym.make('glider3D-v0')
    ppo = PPO()
    memory = Memory()

    # load parameters
    _params_rl = params_rl()
    _params_model = params_model()
    _params_task = params_task()
    _params_sim = params_sim()
    _params_glider = params_glider()
    _params_physics = params_physics()
    _params_logging = params_logging()
    _params_wind = params_wind()
    
    # create folder to store data for the experiment running
    experimentID = 1
    dirName = "experiment_running_{}".format(experimentID)
    while os.path.exists(dirName):
        experimentID += 1
        dirName = dirName.rsplit('_', 1)[0] + "_{}".format(experimentID)
    os.mkdir(dirName)
    shutil.copytree(os.getcwd(), os.path.join(dirName, "Sources_unzipped"),
                    ignore=shutil.ignore_patterns('experiment*', 'archive', 'tests', '.git*', 'rl_ccs_experiments',
                                                  '.idea', '__pycache__', 'README*'))
    os.chdir(dirName)
    shutil.make_archive("Sources", 'zip', "Sources_unzipped")
    shutil.rmtree("Sources_unzipped")
    print("Directory for running experiment no. {} created".format(experimentID))

    # save parameters to file
    parameterFile = open("parameterFile.txt", "w")
    parameterFile.write(
        format(vars(_params_rl)) + "\n" + format(vars(_params_model)) + "\n" + format(vars(_params_task)) + "\n" +
        format(vars(_params_wind)) + "\n" + format(vars(_params_sim)) + "\n" + format(vars(_params_glider)) + "\n" +
        format(vars(_params_physics)) + "\n" + format(vars(_params_logging)))
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
    evaluateGlider.main(env, ppo, 0)
    
    # training loop
    for n_epi in range(1, int(_params_rl.N_EPISODES + 1)):
        p_loiter = 0.9 - (.8 / _params_rl.N_EPISODES) * (n_epi - 1)
        env.reset(p_loiter=p_loiter)
        done = False
        ret = 0
        state = env.state2observation()
        rel_updraft_pos, _ = env.get_rel_updraft_positions()

        while not done:
            timestep_counter += 1

            # run policy_old
            action = ppo.select_action(state, rel_updraft_pos, memory)
            state, reward, done, _ = env.step(action)
            rel_updraft_pos, _ = env.get_rel_updraft_positions()
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
            print("# episode: {} \t\t vertices hit: {} \t avg return over last {} episodes: {:.1f}"
                  .format(n_epi, (env.lapCounter * 3 + env.vertexCounter),
                          _params_logging.PRINT_INTERVAL, average_returns[-1]))

            with open("returnFile_running.dat", "a+") as returnFile:
                returnFile.write(format(n_epi) + "," + format(policy_iterations) + ","
                                 + '{:.1f}'.format(average_returns[-1]) + "\n")

        if n_epi % _params_logging.SAVE_INTERVAL == 0:
            torch.save(ppo.policy.state_dict(), "actor_critic_network_episode_{}".format(n_epi) + ".pt")
            evaluateGlider.main(env, ppo, n_epi)

    # display results
    print('Duration: %.2f s' % (time.time() - tstart))
    evaluateGlider.main(env, ppo, "final", True)
    now = datetime.datetime.now()
    plt.figure("average returns")
    plt.plot(average_returns)
    plt.ylabel("average returns (-)")
    plt.xlabel("episodes (-)")
    plt.grid(True)
    plt.savefig("average_returns_" + now.strftime("%d-%B-%Y_%H-%M") + ".png")
    plt.show()

    # save actor-critic
    torch.save(ppo.policy.state_dict(), "actor_critic_network_final_" + now.strftime("%d-%B-%Y_%H-%M") + ".pt")

    # rename parameter file consistently
    os.rename(parameterFile.name, "parameters_" + now.strftime("%d-%B-%Y_%H-%M") + ".txt")

    # rename return file consistently
    returnFile.close()
    os.rename(returnFile.name, "average_returns_" + now.strftime("%d-%B-%Y_%H-%M") + ".dat")

    env.close()

if __name__ == '__main__':
    main()