import datetime
import os
import shutil
import gym
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, TensorDataset
import collections

import evaluate_decision_maker
from parameters import params_environment, params_triangle_soaring, params_decision_maker
from subtasks.updraft_exploiter import model_updraft_exploiter
from subtasks.vertex_tracker.waypoint_controller import Controller_Wrapper
import utils.core as core

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        # instantiate parameters
        self._params_model = params_decision_maker.params_model()
        self._params_rl = params_decision_maker.params_rl()

        # setup ANN
        self.actor = LSTMActor(obs_dim=self._params_model.DIM_IN, act_dim=self._params_model.DIM_OUT,
                               hidden_size=self._params_model.DIM_HIDDEN, lstm_size=self._params_model.DIM_LSTM)
        self.critic = Critic(obs_dim=self._params_model.DIM_IN, hidden_size=self._params_model.DIM_HIDDEN)

    def act(self, state, lstm_hidden, validation_mask=False):
        # evaluate current actor to sample action for rollout
        action_mean, lstm_hidden = self.actor.forward(state, lstm_hidden)
        dist = Normal(action_mean, self._params_rl.SIGMA * (not validation_mask))
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action, action_logprob, lstm_hidden

    def evaluate_actor(self, sampled_state, sampled_action, sampled_lstm_hidden):
        # evaluate actor for sampled states
        action_mean, _ = self.actor.forward(sampled_state, sampled_lstm_hidden)
        dist = Normal(torch.flatten(action_mean, 1), self._params_rl.SIGMA)

        # get logprobs for distribution subject to current actor, evaluated for sampled actions
        action_logprobs = dist.log_prob(sampled_action)

        return action_logprobs.flatten()

    def reset_lstm(self):
        return torch.zeros(1, 1, self._params_model.DIM_LSTM, device=device), \
               torch.zeros(1, 1, self._params_model.DIM_LSTM, device=device)

class LSTMActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size, lstm_size):
        super().__init__()

        self.input_layer    = nn.Linear(obs_dim, hidden_size)
        self.lstm           = nn.LSTM(hidden_size, lstm_size)
        self.output_layer   = nn.Linear(lstm_size, act_dim)

        self._obs_dim       = obs_dim
        self._act_dim       = act_dim
        self._hidden_size   = hidden_size
        self._lstm_size     = lstm_size

    def forward(self, observation, lstm_hidden):
        # evaluate input
        x = observation.reshape(-1, self._obs_dim).to(device)  # seq_len x  input_size
        x = torch.tanh(self.input_layer(x))

        # evaluate lstm
        x = x.reshape(-1, 1, self._hidden_size)  # seq_len x batch_size x  lstm_in_size
        x, lstm_hidden = self.lstm(x, (lstm_hidden[0].reshape(1, 1, self._hidden_size),
                                       lstm_hidden[1].reshape(1, 1, self._hidden_size)))

        # evaluate actor output layer
        x = x.reshape(-1, self._lstm_size)  # seq_len x lstm_out_size
        action = self.output_layer(x)

        # # map output to [0, 1]
        # p_exploit = 0.5 * (torch.tanh(z) + 1)

        return action, lstm_hidden


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_size):
        super().__init__()

        self.input_layer    = nn.Linear(obs_dim, hidden_size)
        self.hidden_layer   = nn.Linear(hidden_size, hidden_size)
        self.output_layer   = nn.Linear(hidden_size, 1)

        self._obs_dim       = obs_dim

    def forward(self, observation):
        # evaluate input
        x = observation.reshape(-1, self._obs_dim).to(device)  # batch_size x  input_size
        x = torch.tanh(self.input_layer(x))

        # evaluate hidden layer
        x = torch.tanh(self.hidden_layer(x))

        # evaluate critic output layer
        value = self.output_layer(x)

        return value


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment.
    """
    def __init__(self, obs_dim, act_dim, batch_size, lstm_hidden_size, gamma=0.99, lam=0.95):
        self.obs_buf        = torch.zeros(batch_size, obs_dim, dtype=torch.float32, device=device)
        self.act_buf        = torch.zeros(batch_size, act_dim, dtype=torch.float32, device=device)
        self.adv_buf        = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.rew_buf        = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.ret_buf        = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.val_buf        = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.logp_buf       = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.lstm_h_in_buf  = torch.zeros(batch_size, lstm_hidden_size, dtype=torch.float32, device=device)
        self.lstm_c_in_buf  = torch.zeros(batch_size, lstm_hidden_size, dtype=torch.float32, device=device)
        self.done_buf       = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, batch_size

    def store(self, obs, act, rew, val, logp, lstm_h_in, lstm_c_in, done):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr]          = obs
        self.act_buf[self.ptr]          = act
        self.rew_buf[self.ptr]          = rew
        self.val_buf[self.ptr]          = val
        self.logp_buf[self.ptr]         = logp
        self.lstm_h_in_buf[self.ptr]    = lstm_h_in
        self.lstm_c_in_buf[self.ptr]    = lstm_c_in
        self.done_buf[self.ptr]         = done
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = torch.cat((self.rew_buf[path_slice], last_val))
        vals = torch.cat((self.val_buf[path_slice], last_val))

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = torch.FloatTensor(
            (core.discount_cumsum(deltas.cpu().numpy(), self.gamma * self.lam)).copy()).to(device)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = torch.FloatTensor(
            (core.discount_cumsum(rews.cpu().numpy(), self.gamma)[:-1]).copy()).to(device)

        self.path_start_idx = self.ptr

    def get(self,):
        """
        Gets all of the data from the buffer and resets pointer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        # the next line implements the advantage normalization trick
        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std() + 1e-5)

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf,
                    lstm_h_in=self.lstm_h_in_buf, lstm_c_in=self.lstm_c_in_buf, done=self.done_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in data.items()}


class MyDataset(Dataset):
    def __init__(self, data, window):
        self.data = data
        self.window = window

    def __getitem__(self, index):
        x = self.data[index:index + self.window]
        return x

    def __len__(self):
        return len(self.data) - self.window


class PPO:
    def __init__(self, vertex_tracker, updraft_exploiter, environment):
        # instantiate parameters
        self._params_rl = params_decision_maker.params_rl()
        self._params_model = params_decision_maker.params_model()

        # instantiate buffer for rollout
        self.buffer = PPOBuffer(self._params_model.DIM_IN, self._params_model.DIM_OUT, self._params_rl.BATCHSIZE,
                                self._params_model.DIM_LSTM, gamma=self._params_rl.GAMMA, lam=self._params_rl.LAMBDA)

        # instantiate actor-critic model
        self.model = ActorCritic().to(device)
        # self.model.actor.load_state_dict(torch.load("decision_maker_actor_final_03-November-2020_15-17.pt"))
        # self.model.critic.load_state_dict(torch.load("decision_maker_critic_final_03-November-2020_15-17.pt"))

        # setup optimizers
        self.pi_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=self._params_rl.LEARNING_RATE_PI)
        self.vf_optimizer = torch.optim.Adam(self.model.critic.parameters(), lr=self._params_rl.LEARNING_RATE_VF)

        # instantiate vertex tracker and updraft exploiter
        self.vertex_tracker = vertex_tracker
        self.updraft_exploiter = updraft_exploiter
        self.env = environment

    def select_action(self, state, lstm_hidden, validation_mask=False):
        # evaluate decision maker
        action_agent, action_agent_logprob, lstm_hidden = self.model.act(state, lstm_hidden,
                                                                          validation_mask=validation_mask)

        # induce bias: height=0 leads to bias->1, initial_height (400 m) leads to bias->0
        inductive_bias = .5 * (1 - torch.tanh(state[1]))
        p_exploit = np.clip(action_agent.item() + inductive_bias.item(), 0, 1)

        # evaluate vertex tracker
        action_vertex_tracker = self.vertex_tracker.select_action(self.env.get_full_observation)

        # evaluate updraft exploiter
        normalized_updraft_positions = self.env.get_updraft_positions()
        observation = torch.FloatTensor(normalized_updraft_positions).view(1, -1, 2).to(device)
        action_updraft_exploiter = self.updraft_exploiter.act(observation,
                                                              memory=None,
                                                              validation_mask=True).cpu().data.numpy().flatten()

        # mix the actions
        action_env = p_exploit * action_updraft_exploiter + (1 - p_exploit) * action_vertex_tracker

        # evaluate critic
        state_value = self.model.critic(state)

        return action_env, action_agent, action_agent_logprob, state_value, lstm_hidden

    def update(self):

        # get sampled data
        data = self.buffer.get()
        obs, act, ret, adv, logp, lstm_h_in, lstm_c_in, done \
            = data['obs'], data['act'], data['ret'], data['adv'], data['logp'],\
              data['lstm_h_in'], data['lstm_c_in'], data['done']

        # put batch to sliding-window data_loader
        data_set = MyDataset(TensorDataset(obs, act, ret, adv, logp, lstm_h_in, lstm_c_in, done),
                             self._params_rl.SEQ_LEN)

        # Optimize policy for K epochs:
        for _ in range(self._params_rl.K_EPOCH):
            for start_index in range(0, (data_set.__len__()) + 1,
                                     int(self._params_rl.SEQ_LEN - self._params_rl.OVERLAP)):
                # get sampled sequence/mini-batch
                obs_seq, act_seq, ret_seq, adv_seq, logp_seq, lstm_h_in_seq, lstm_c_in_seq, done_seq = \
                    data_set.__getitem__(start_index)

                if any(done_seq == 1):
                    #  never evaluate sequences that cross episode boundaries
                    done_index = done_seq.nonzero()[0].item()  # index of first done flag in sequence
                    if done_index > (self._params_rl.SEQ_LEN_MIN - 1):
                        obs_seq         = obs_seq[slice(0, done_index + 1)]
                        act_seq         = act_seq[slice(0, done_index + 1)]
                        ret_seq         = ret_seq[slice(0, done_index + 1)]
                        adv_seq         = adv_seq[slice(0, done_index + 1)]
                        logp_seq        = logp_seq[slice(0, done_index + 1)]
                        lstm_h_in_seq   = lstm_h_in_seq[slice(0, done_index + 1)]
                        lstm_c_in_seq   = lstm_c_in_seq[slice(0, done_index + 1)]
                    else:
                        break

                # "burn in" lstm hidden state (cf. "R2D2")
                with torch.no_grad():
                    _, (lstm_h_burned_in, lstm_c_burned_in) =\
                        self.model.actor(obs_seq[0:self._params_rl.N_BURNIN, :], (lstm_h_in_seq[0], lstm_c_in_seq[0]))

                # evaluate policy for remainder sampled sequence of states and actions
                logp_eval = self.model.evaluate_actor(obs_seq[self._params_rl.N_BURNIN:, :],
                                                      act_seq[self._params_rl.N_BURNIN:],
                                                      (lstm_h_burned_in, lstm_c_burned_in))

                # ppo ratio
                ratios = torch.exp(logp_eval - logp_seq[self._params_rl.N_BURNIN:])

                # surrogate loss (PPO)
                surr1 = ratios * adv_seq[self._params_rl.N_BURNIN:]
                surr2 = torch.clamp(ratios, 1 - self._params_rl.EPS_CLIP, 1 + self._params_rl.EPS_CLIP)\
                        * adv_seq[self._params_rl.N_BURNIN:]
                loss_pi = -torch.min(surr1, surr2).mean()

                # policy gradient step
                self.pi_optimizer.zero_grad()
                loss_pi.backward()
                self.pi_optimizer.step()

                # value function gradient step
                loss_vf = ((self.model.critic(obs_seq) - ret_seq)**2).mean()
                self.vf_optimizer.zero_grad()
                loss_vf.backward()
                self.vf_optimizer.step()

def main():
    # set up training
    env = gym.make('glider3D-v0', agent='decision_maker')

    # instantiate vertex-tracker and updraft-exploiter
    waypoint_controller = Controller_Wrapper(env)
    updraft_exploiter = model_updraft_exploiter.ActorCritic().to(device)
    updraft_exploiter.load_state_dict(torch.load("updraft_exploiter_actor_critic_final_17-December-2020_11-06.pt"))

    # instantiate agent
    ppo = PPO(waypoint_controller, updraft_exploiter, env)

    # load parameters
    _params_rl = params_decision_maker.params_rl()
    _params_agent = params_decision_maker.params_agent()
    _params_logging = params_decision_maker.params_logging()

    # create folder to store data for the experiment running
    experimentID = 1
    dirName = "{}_decision_maker_experiment".format(experimentID)
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

    # set up file to save average returns and scores
    return_file = open("returnFile_running.dat", "w")
    return_file.write("iterations,episodes,avg_returns,avg_scores\n")
    return_file.close()

    # initialize logging variables
    returns = collections.deque(maxlen=10)
    scores = collections.deque(maxlen=10)
    average_returns = []
    average_scores = []
    policy_iterations = 0
    episodes = 0
    interactions = 0
    ret = 0

    # create sample of untrained system behavior
    evaluate_decision_maker.main(env, ppo, policy_iterations, _params_agent, validation_mask=True)

    # (re-)set env and model
    env.reset()
    obs = env.get_observation()
    lstm_in = ppo.model.reset_lstm()

    # training loop
    while policy_iterations < (int(_params_rl.N_ITERATIONS)):

        # rollout s.t. current policy
        with torch.no_grad():
            action_env, action_agent, action_agent_logprob, state_value, lstm_out = \
                ppo.select_action(torch.FloatTensor(obs), lstm_in)
            next_obs, reward, done, _ = env.step(action_env)

        # store data in ppo.buffer
        ppo.buffer.store(obs=torch.FloatTensor(obs).to(device), act=action_agent.flatten(),
                         rew=torch.FloatTensor([reward/100]), val=torch.FloatTensor([state_value]),
                         logp=action_agent_logprob.flatten(), lstm_h_in=lstm_in[0].flatten(),
                         lstm_c_in=lstm_in[1].flatten(), done=torch.FloatTensor([done]))

        # update variables each interaction
        obs = next_obs
        lstm_in = lstm_out
        ret += reward
        interactions += 1

        # store results and reset experiment if episode is completed
        if done:
            ppo.buffer.finish_path(torch.FloatTensor([0]).to(device))
            returns.append(ret)
            scores.append(env.lap_counter * 200)
            episodes += 1
            ret = 0
            env.reset()
            lstm_in = ppo.model.reset_lstm()
            obs = env.get_observation()

            n_mean = _params_logging.PRINT_INTERVAL if len(returns) >= _params_logging.PRINT_INTERVAL else len(returns)
            average_returns.append(np.convolve(list(returns)[-n_mean:], np.ones((n_mean,)) / n_mean, mode='valid')[0])
            average_scores.append(np.convolve(list(scores)[-n_mean:], np.ones((n_mean,)) / n_mean, mode='valid')[0])

        # update policy every BATCHSIZE interactions
        if interactions != 0 and interactions % _params_rl.BATCHSIZE == 0:
            next_value = ppo.model.critic(torch.FloatTensor(obs)).detach()
            ppo.buffer.finish_path(next_value.flatten())
            ppo.model.train()
            ppo.update()
            ppo.model.eval()
            interactions = 0
            policy_iterations += 1

            # print results to display and log them to file
            if len(average_returns) and policy_iterations % _params_logging.PRINT_INTERVAL == 0:
                print("# policy iteration: {}/{} \t\t"
                      "avg. return over last 10 episodes: {:06.1f} \t\t"
                      "avg. score over last 10 episodes: {:06.1f}"
                      .format(policy_iterations, int(_params_rl.N_ITERATIONS), average_returns[-1], average_scores[-1]))

                with open("returnFile_running.dat", "a+") as return_file:
                    return_file.write(format(policy_iterations) + "," + format(episodes) + "," +
                                     '{:.1f}'.format(average_returns[-1]) + "," + '{:.1f}'.format(average_scores[-1]) +
                                     "\n")

            # save model and create sample of system behavior
            if policy_iterations % _params_logging.SAVE_INTERVAL == 0:
                torch.save(ppo.model.actor.state_dict(),
                           "decision_maker_actor_iter_{}".format(policy_iterations) + ".pt")
                torch.save(ppo.model.critic.state_dict(),
                           "decision_maker_critic_iter_{}".format(policy_iterations) + ".pt")
                evaluate_decision_maker.main(env, ppo, policy_iterations, _params_agent, validation_mask=True)

    # display final results
    now = datetime.datetime.now()
    returns_to_plot = pd.read_csv('returnFile_running.dat')
    returns_to_plot.plot(x='iterations', y='avg_returns')
    plt.title("evolution of average returns")
    plt.xlabel("policy iterations (-)")
    plt.ylabel("average returns (-)")
    plt.grid(True)
    plt.savefig("average_returns_" + now.strftime("%d-%B-%Y_%H-%M") + ".png")
    plt.show()

    # save final model
    torch.save(ppo.model.actor.state_dict(), "decision_maker_actor_final_" + now.strftime("%d-%B-%Y_%H-%M") + ".pt")
    torch.save(ppo.model.critic.state_dict(), "decision_maker_critic_final_" + now.strftime("%d-%B-%Y_%H-%M") + ".pt")

    # rename parameter file consistently
    os.rename(parameterFile.name, "parameters_" + now.strftime("%d-%B-%Y_%H-%M") + ".txt")

    # rename return file consistently
    return_file.close()
    os.rename(return_file.name, "average_returns_" + now.strftime("%d-%B-%Y_%H-%M") + ".dat")

    env.close()


if __name__ == '__main__':
    main()
