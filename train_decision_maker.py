import datetime
import time
import os
import shutil
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal, Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset

import evaluate_glider
from parameters import params_environment, params_triangle_soaring, params_decision_maker
from subtasks.updraft_exploiter import model_updraft_exploiter
from subtasks.vertex_tracker.waypoint_controller.waypoint_controller import Waypoint_Controller, Controller_Wrapper


device = torch.device("cuda:0")
# device = torch.device("cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.h_old = []
        self.c_old = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.h_old[:]
        del self.c_old[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()

        # instantiate parameters
        self._params_model = params_decision_maker.params_model()
        self._params_rl = params_decision_maker.params_rl()

        # set model elements: lstm net
        self.lstm           = nn.LSTM(input_size=self._params_model.DIM_IN, hidden_size=self._params_model.DIM_HIDDEN,
                                      batch_first=True)
        self.h_old          = torch.zeros(1, 1, self._params_model.DIM_HIDDEN)
        self.c_old          = torch.zeros(1, 1, self._params_model.DIM_HIDDEN)

        # set model elements: ffwd layers
        self.out_actor      = nn.Linear(self._params_model.DIM_HIDDEN, self._params_model.DIM_OUT)
        self.out_critic     = nn.Linear(self._params_model.DIM_HIDDEN, 1)

    def actor(self, observation, h_old_in=None, c_old_in=None):
        # initialize lstm
        h_old = self.h_old.to(device) if h_old_in is None else h_old_in
        c_old = self.c_old.to(device) if c_old_in is None else c_old_in

        # evaluate lstm
        _, (h_n, c_n) = self.lstm(observation, (h_old, c_old))

        # update lstm state (for rollout, only)
        self.h_old = h_n if h_old_in is None else self.h_old
        self.c_old = c_n if c_old_in is None else self.c_old

        # forward lstm hidden state for t = seq_len
        x = torch.tanh(h_n[-1])

        # evaluate feedforward layer
        z = self.out_actor(x)
        probs = F.softmax(z, dim=1)

        return probs, (h_old, c_old)

    def critic(self, observation, h_old=None, c_old=None):
        # initialize lstm
        h_old = self.h_old.to(device) if h_old is None else h_old
        c_old = self.c_old.to(device) if c_old is None else c_old

        # evaluate lstm
        _, (h_n, c_n) = self.lstm(observation, (h_old, c_old))

        # forward lstm hidden state for t = seq_len
        x = torch.tanh(h_n[-1])

        # evaluate feedforward layer
        state_value = self.out_critic(x)

        return state_value
    
    def act(self, state, memory, validation_mask):
        # evaluate current actor to sample action for rollout
        probs, lstm_state = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        # store state, action, logprob (not for evaluate_glider())
        if memory is not None:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
            memory.h_old.append(lstm_state[0])
            memory.c_old.append(lstm_state[1])
        
        return action.detach()
    
    def evaluate(self, state_old, action_old, h_old, c_old):
        # evaluate actor for sampled states
        probs, _ = self.actor(state_old, h_old, c_old)
        dist = Categorical(probs)

        # get logprobs & entropy for distribution subject to current actor evaluated for sampled actions
        action_logprobs = dist.log_prob(action_old)
        dist_entropy = dist.entropy()

        # evaluate critic for sampled states
        state_values = self.critic(state_old, h_old, c_old)

        return action_logprobs, state_values, dist_entropy

    def reset_lstm(self, batchsize=1):
        self.h_old = torch.zeros(1, batchsize, self._params_model.DIM_HIDDEN)
        self.c_old = torch.zeros(1, batchsize, self._params_model.DIM_HIDDEN)

class PPO:
    def __init__(self, vertex_tracker, updraft_exploiter, environment):
        # instantiate parameters
        self._params_rl = params_decision_maker.params_rl()

        # instantiate actor-critic model
        self.policy = ActorCritic().to(device)
        # self.policy.load_state_dict(torch.load("actor_critic_network_final_01-April-2020_14-58.pt"))
        self.policy_old = ActorCritic().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # setup optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=self._params_rl.LEARNING_RATE, betas=(0.9, 0.999))
        self.MseLoss = nn.MSELoss()

        # instantiate vertex tracker and updraft exploiter
        self.vertex_tracker = vertex_tracker
        self.updraft_exploiter = updraft_exploiter
        self.env = environment
    
    def select_action(self, state, memory=None, validation_mask=False):
        state = torch.FloatTensor(state.reshape(1, 1, -1)).to(device)  # batch=1 x sequence=1 x observations
        decision = self.policy.act(state, memory, validation_mask).cpu().data.numpy().flatten()
        if decision == 0:
            action = self.vertex_tracker.select_action(self.env.get_full_observation)
        elif decision == 1:
            observation = torch.FloatTensor(self.env.get_rel_updraft_positions()).view(1, -1, 2).to(device)
            action = self.updraft_exploiter.act(observation, None, True).cpu().data.numpy().flatten()
        return action
    
    def update(self, memory):
        # estimate returns (Monte Carlo) TODO: GAE
        returns_to_go = []
        discounted_reward = self.policy_old.critic(memory.states[-1], memory.h_old[-1], memory.c_old[-1])
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self._params_rl.GAMMA * discounted_reward)
            returns_to_go.insert(0, discounted_reward)

        # normalize the rewards
        returns_to_go = torch.tensor(returns_to_go).to(device)
        returns_to_go = (returns_to_go - returns_to_go.mean()) / (returns_to_go.std() + 1e-5)
        returns_to_go = returns_to_go.unsqueeze(1)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).detach()
        old_lstm_hiddens = torch.squeeze(torch.stack(memory.h_old), 1).detach()
        old_lstm_states = torch.squeeze(torch.stack(memory.c_old), 1).detach()

        # put batch to data_loader
        train_loader = DataLoader(dataset=TensorDataset(old_states, old_actions, old_logprobs,
                                                        old_lstm_hiddens, old_lstm_states, returns_to_go),
                                  batch_size=self._params_rl.MINIBATCHSIZE, shuffle=True)
        
        # Optimize policy for K epochs:
        for _ in range(self._params_rl.K_EPOCH):
            for mini_batch in train_loader:
                # get sampled data in mini-batch and send them to device
                old_states, old_actions, old_logprobs, old_lstm_hiddens, old_lstm_states, returns_to_go = mini_batch

                # re-arrange h_old, c_old (necessary in spite of batch_first = True): layers x batch_size x hidden_size
                old_lstm_hiddens = old_lstm_hiddens.permute(1, 0, 2)
                old_lstm_states = old_lstm_states.permute(1, 0, 2)

                # evaluate policy for sampled states and actions
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions,
                                                                            old_lstm_hiddens, old_lstm_states)

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
    env = gym.make('glider3D-v0', agent='decision_maker')

    # instantiate vertex tracker and updraft exploiter
    waypoint_controller = Controller_Wrapper(env)
    updraft_exploiter = model_updraft_exploiter.ActorCritic().to(device)
    updraft_exploiter.load_state_dict(torch.load("updraft_exploiter_actor_critic_episode_1500.pt"))

    # instantiate agent
    ppo = PPO(waypoint_controller, updraft_exploiter, env)
    memory = Memory()

    # load parameters
    _params_rl = params_decision_maker.params_rl()
    _params_agent = params_decision_maker.params_agent()
    _params_logging = params_decision_maker.params_logging()
    
    # create folder to store data for the experiment running
    experimentID = 1
    dirName = "decision_maker_experiment_running_{}".format(experimentID)
    while os.path.exists(dirName):
        experimentID += 1
        dirName = dirName.rsplit('_', 1)[0] + "_{}".format(experimentID)
    os.mkdir(dirName)
    shutil.copytree(os.getcwd(), os.path.join(dirName, "Sources_unzipped"),
                    ignore=shutil.ignore_patterns('experiment*', 'archive', 'tests', '.git*', 'rl_ccs_experiments',
                                                  '.idea', '__pycache__', 'README*', 'updraft*', 'vertex*', 'decision*'))
    os.chdir(dirName)
    shutil.make_archive("Sources", 'zip', "Sources_unzipped")
    shutil.rmtree("Sources_unzipped")
    print("Directory for running experiment no. {} created".format(experimentID))

    # save parameters to file
    parameterFile = open("parameterFile.txt", "w")
    parameterFile.write(
        format(vars(_params_rl)) + "\n" +
        format(vars(params_decision_maker.params_model())) + "\n" +
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
    evaluate_glider.main(env, ppo, 0, _params_agent, validation_mask=False)
    
    # training loop
    for n_epi in range(1, int(_params_rl.N_EPISODES + 1)):
        env.reset()
        ppo.policy.reset_lstm()
        done = False
        ret = 0
        observation = env.get_observation()

        while not done:
            timestep_counter += 1

            # run policy
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
            print("# episode: {}/{} \t\t vertices hit: {} \t avg return over last {} episodes: {:.1f}"
                  .format(n_epi, int(_params_rl.N_EPISODES),  (env.lap_counter * 3 + env.vertex_counter),
                          _params_logging.PRINT_INTERVAL, average_returns[-1]))

            with open("returnFile_running.dat", "a+") as returnFile:
                returnFile.write(format(n_epi) + "," + format(policy_iterations) + ","
                                 + '{:.1f}'.format(average_returns[-1]) + "\n")

        if n_epi % _params_logging.SAVE_INTERVAL == 0:
            torch.save(ppo.policy.state_dict(), "decision_maker_actor_critic_episode_{}".format(n_epi) + ".pt")
            evaluate_glider.main(env, ppo, n_epi, _params_agent, validation_mask=False)

    # display results
    print('Duration: %.2f s' % (time.time() - tstart))
    evaluate_glider.main(env, ppo, n_epi, _params_agent, validation_mask=True)
    now = datetime.datetime.now()
    plt.figure("average returns")
    plt.plot(average_returns)
    plt.ylabel("average returns (-)")
    plt.xlabel("episodes (-)")
    plt.grid(True)
    plt.savefig("average_returns_" + now.strftime("%d-%B-%Y_%H-%M") + ".png")
    plt.show()

    # save actor-critic
    torch.save(ppo.policy.state_dict(), "decision_maker_actor_critic_final_" + now.strftime("%d-%B-%Y_%H-%M") + ".pt")

    # rename parameter file consistently
    os.rename(parameterFile.name, "parameters_" + now.strftime("%d-%B-%Y_%H-%M") + ".txt")

    # rename return file consistently
    returnFile.close()
    os.rename(returnFile.name, "average_returns_" + now.strftime("%d-%B-%Y_%H-%M") + ".dat")

    env.close()


if __name__ == '__main__':
    main()