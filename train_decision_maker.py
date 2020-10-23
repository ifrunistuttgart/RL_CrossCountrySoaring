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

import evaluate_decision_maker
from parameters import params_environment, params_triangle_soaring, params_decision_maker
from subtasks.updraft_exploiter import model_updraft_exploiter
from subtasks.vertex_tracker.waypoint_controller.waypoint_controller import Waypoint_Controller, Controller_Wrapper
import utils.core as core

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()

        # instantiate parameters
        self._params_model = params_decision_maker.params_model()
        self._params_rl = params_decision_maker.params_rl()

        # setup ANN
        self.input = nn.Linear(self._params_model.DIM_IN, self._params_model.DIM_HIDDEN)
        self.lstm = nn.LSTM(self._params_model.DIM_HIDDEN, self._params_model.DIM_LSTM)
        self.out_actor = nn.Linear(self._params_model.DIM_LSTM, self._params_model.DIM_OUT)
        self.out_critic = nn.Linear(self._params_model.DIM_LSTM, 1)

    def actor(self, observation, lstm_hidden):
        # evalutate input
        x = observation.reshape(-1, self._params_model.DIM_IN).to(device)  # seq_len x  input_size
        x = torch.tanh(self.input(x))

        # evaluate lstm
        x = x.reshape(-1, 1, self._params_model.DIM_HIDDEN)  # seq_len x batch_size x  lstm_in_size
        x, lstm_hidden = self.lstm(x, (lstm_hidden[0].reshape(1, 1, self._params_model.DIM_HIDDEN),
                                       lstm_hidden[1].reshape(1, 1, self._params_model.DIM_HIDDEN)))

        # evaluate actor ouput layer
        x = x.reshape(-1, self._params_model.DIM_LSTM)  # seq_len x lstm_out_size
        z = self.out_actor(x)

        # map output to [0, 1]
        p_exploit = 0.5 * (torch.tanh(z) + 1)

        return p_exploit, lstm_hidden

    def critic(self, observation, lstm_hidden):
        # evalutate input
        x = observation.reshape(-1, self._params_model.DIM_IN).to(device)  # seq_len x  input_size
        x = torch.tanh(self.input(x))

        # evaluate lstm
        x = x.reshape(-1, 1, self._params_model.DIM_HIDDEN)  # seq_len x batch_size x  lstm_in_size
        x, lstm_hidden = self.lstm(x, (lstm_hidden[0].reshape(1, 1, self._params_model.DIM_HIDDEN),
                                       lstm_hidden[1].reshape(1, 1, self._params_model.DIM_HIDDEN)))

        # evaluate critic ouput layer
        x = x.reshape(-1, self._params_model.DIM_LSTM)  # seq_len x lstm_out_size
        state_value = self.out_critic(x)

        return state_value

    def act(self, state, lstm_hidden, validation_mask=False):
        # evaluate current actor to sample action for rollout
        action_mean, lstm_hidden = self.actor(state, lstm_hidden)
        dist = Normal(action_mean, self._params_rl.SIGMA * (not validation_mask))
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action, action_logprob, lstm_hidden

    def evaluate(self, sampled_state, sampled_action, sampled_lstm_hidden):
        # evaluate actor for sampled states
        action_mean, _ = self.actor(sampled_state, sampled_lstm_hidden)
        dist = Normal(torch.flatten(action_mean, 1), self._params_rl.SIGMA)

        # get logprobs & entropy for distribution subject to current actor evaluated for sampled actions
        action_logprobs = dist.log_prob(sampled_action)
        entropy_bonus = dist.entropy()

        # evaluate critic for sampled states
        state_values = self.critic(sampled_state, sampled_lstm_hidden)

        return action_logprobs.flatten(), state_values.flatten(), entropy_bonus

    def reset_lstm(self):
        return torch.zeros(1, 1, self._params_model.DIM_LSTM, device=device), \
               torch.zeros(1, 1, self._params_model.DIM_LSTM, device=device)


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment.
    """
    def __init__(self, obs_dim, act_dim, batch_size, lstm_hidden_size):
        self.obs_buf        = torch.zeros(batch_size, obs_dim, dtype=torch.float32, device=device)
        self.act_buf        = torch.zeros(batch_size, act_dim, dtype=torch.float32, device=device)
        self.rew_buf        = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.next_obs_buf   = torch.zeros(batch_size, obs_dim, dtype=torch.float32, device=device)
        self.logp_buf       = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.lstm_h_in_buf  = torch.zeros(batch_size, lstm_hidden_size, dtype=torch.float32, device=device)
        self.lstm_c_in_buf  = torch.zeros(batch_size, lstm_hidden_size, dtype=torch.float32, device=device)
        self.done_mask_buf  = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.ptr, self.max_size = 0, batch_size

    def store(self, obs, act, rew, next_obs, logp, lstm_h_in, lstm_c_in, done_mask):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr]          = obs
        self.act_buf[self.ptr]          = act
        self.rew_buf[self.ptr]          = rew
        self.next_obs_buf[self.ptr]     = next_obs
        self.logp_buf[self.ptr]         = logp
        self.lstm_h_in_buf[self.ptr]    = lstm_h_in
        self.lstm_c_in_buf[self.ptr]    = lstm_c_in
        self.done_mask_buf[self.ptr]     = done_mask
        self.ptr += 1


    def get(self):
        """
        Gets all of the data from the buffer and resets pointer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr = 0

        data = dict(obs=self.obs_buf, act=self.act_buf, rew=self.rew_buf, next_obs=self.next_obs_buf,
                    logp=self.logp_buf, lstm_h_in=self.lstm_h_in_buf, lstm_c_in=self.lstm_c_in_buf,
                    done_mask=self.done_mask_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in data.items()}


class PPO:
    def __init__(self, vertex_tracker, updraft_exploiter, environment):
        # instantiate parameters
        self._params_rl = params_decision_maker.params_rl()
        self._params_model = params_decision_maker.params_model()

        # instantiate buffer for rollout
        self.buffer = PPOBuffer(self._params_model.DIM_IN, self._params_model.DIM_OUT, self._params_rl.BATCHSIZE,
                                self._params_model.DIM_LSTM)

        # instantiate actor-critic model
        self.policy = ActorCritic().to(device)
        # self.policy.load_state_dict(torch.load("decision_maker_actor_critic_final_17-August-2020_14-27.pt"))

        # setup optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._params_rl.LEARNING_RATE)

        # instantiate vertex tracker and updraft exploiter
        self.vertex_tracker = vertex_tracker
        self.updraft_exploiter = updraft_exploiter
        self.env = environment

    def select_action(self, state, lstm_hidden, validation_mask=False):
        # evaluate decision maker
        action_agent, action_agent_logprob, lstm_hidden = self.policy.act(state, lstm_hidden,
                                                                          validation_mask=validation_mask)
        p_exploit = np.clip(action_agent.item(), 0, 1)

        # evaluate vertex tracker
        action_vertex_tracker = self.vertex_tracker.select_action(self.env.get_full_observation)

        # evaluate updraft exploiter
        normalized_updraft_positions, _ = self.env.get_rel_updraft_positions()
        observation = torch.FloatTensor(normalized_updraft_positions).view(1, -1, 2).to(device)
        action_updraft_exploiter = self.updraft_exploiter.act(observation,
                                                              memory=None,
                                                              validation_mask=True).cpu().data.numpy().flatten()

        # mix the actions
        action_env = p_exploit * action_updraft_exploiter + (1 - p_exploit) * action_vertex_tracker

        return action_env, action_agent, action_agent_logprob, lstm_hidden

    def update(self):

        # get sampled data
        data = self.buffer.get()
        obs, act, rew, next_obs, logp, lstm_h_in, lstm_c_in, done_mask \
            = data['obs'], data['act'], data['rew'], data['next_obs'], data['logp'], data['lstm_h_in'],\
              data['lstm_c_in'], data['done_mask']

        # put batch to data_loader
        train_loader = DataLoader(dataset=TensorDataset(obs, act, rew, next_obs, logp, lstm_h_in, lstm_c_in, done_mask),
                                  batch_size=self._params_rl.MINIBATCHSIZE, shuffle=False)

        # Optimize policy for K epochs:
        for _ in range(self._params_rl.K_EPOCH):
            for sequence in train_loader:
                # get sampled data in mini-batch (treated as a sequence) and send them to device
                obs_seq, act_seq, rew_seq, next_obs_seq, logp_seq, lstm_h_in_seq, lstm_c_in_seq, done_mask_seq = sequence

                # "burn in" lstm hidden state (cf. "R2D2")
                with torch.no_grad():
                    _, (lstm_h_burned_in, lstm_c_burned_in) =\
                        self.policy.actor(obs_seq[0:self._params_rl.N_BURNIN, :], (lstm_h_in_seq[0], lstm_c_in_seq[0]))
                    _, (next_lstm_h_burned_in, next_lstm_c_burned_in) =\
                        self.policy.actor(next_obs_seq[0:self._params_rl.N_BURNIN, :],
                                          (lstm_h_in_seq[1], lstm_c_in_seq[1]))  # TODO: Does this really matter

                # evaluate policy for remainder sampled sequence of states and actions
                logp_eval, v_eval, _ = self.policy.evaluate(obs_seq[self._params_rl.N_BURNIN:, :],
                                                            act_seq[self._params_rl.N_BURNIN:],
                                                            (lstm_h_burned_in, lstm_c_burned_in))

                # GAE
                v_eval_prime = self.policy.critic(next_obs_seq[self._params_rl.N_BURNIN:],
                                                  (next_lstm_h_burned_in, next_lstm_c_burned_in))
                td_target = rew_seq[self._params_rl.N_BURNIN:] +\
                            self._params_rl.GAMMA * v_eval_prime.flatten() * done_mask_seq[self._params_rl.N_BURNIN:]  # TODO: normalize!?
                delta = td_target - v_eval
                delta = delta.detach().cpu().numpy()

                advantage_lst = []  # TODO use trick instead (spinup style)
                advantage = 0.0
                for item in delta[::-1]:
                    advantage = self._params_rl.GAMMA * self._params_rl.LAMBDA * advantage + item
                    advantage_lst.append([advantage])
                advantage_lst.reverse()
                advantages = torch.tensor(advantage_lst, dtype=torch.float).flatten().to(device)

                # ppo ratio
                ratios = torch.exp(logp_eval - logp_seq[self._params_rl.N_BURNIN:])

                # surrogate loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self._params_rl.EPS_CLIP, 1 + self._params_rl.EPS_CLIP) * advantages
                loss = -torch.min(surr1, surr2) + 0.5 * ((v_eval - td_target.detach()) ** 2)


                # gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()


def main():
    # set up training
    env = gym.make('glider3D-v0', agent='decision_maker')

    # instantiate vertex tracker and updraft exploiter
    waypoint_controller = Controller_Wrapper(env)
    updraft_exploiter = model_updraft_exploiter.ActorCritic().to(device)
    updraft_exploiter.load_state_dict(torch.load("updraft_exploiter_actor_critic_final_16-September-2020_23-51.pt"))

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

    # initialize logging variables
    returns = []
    scores = []
    average_returns = []
    average_scores = []
    policy_iterations = 0
    n_interactions = 0

    # training loop
    for n_epi in range(int(_params_rl.N_EPISODES + 1)):
        env.reset()
        lstm_in = ppo.policy.reset_lstm()
        done = False
        ret = 0
        obs = env.get_observation()

        while not done:
            with torch.no_grad():

                # run policy
                action_env, action_agent, action_agent_logprob, lstm_out = \
                    ppo.select_action(torch.FloatTensor(obs), lstm_in)
                next_obs, reward, done, _ = env.step(action_env)

                # store data in ppo.buffer
                ppo.buffer.store(obs=torch.FloatTensor(obs).to(device), act=action_agent.flatten(),
                                 rew=torch.FloatTensor([reward]), next_obs=torch.FloatTensor(next_obs).to(device),
                                 logp=action_agent_logprob.flatten(), lstm_h_in=lstm_in[0].flatten(),
                                 lstm_c_in=lstm_in[1].flatten(), done_mask=torch.FloatTensor([~done]))

                # update variables each interaction
                obs = next_obs
                lstm_in = lstm_out
                ret += reward
                n_interactions += 1

            # update policy every BATCHSIZE tuples stored in buffer
            if n_interactions != 0 and n_interactions % _params_rl.BATCHSIZE == 0:
                ppo.policy.train()
                ppo.update()
                ppo.policy.eval()
                n_interactions = 0
                policy_iterations += 1

            # stop rollout if episode is completed
            if done:
                returns.append(ret)
                scores.append(env.lap_counter * 200)
                break

        n_mean = _params_logging.PRINT_INTERVAL if len(returns) >= _params_logging.PRINT_INTERVAL else len(returns)
        average_returns.append(np.convolve(returns[-n_mean:], np.ones((n_mean,)) / n_mean, mode='valid')[0])
        average_scores.append(np.convolve(scores[-n_mean:], np.ones((n_mean,)) / n_mean, mode='valid')[0])

        if n_epi % _params_logging.PRINT_INTERVAL == 0:
            print("# episode: {}/{} \t\t avg. ret. (last {} eps): {:06.1f} \t\t avg. score (last {} eps): {:06.1f}"
                  .format(n_epi, int(_params_rl.N_EPISODES), _params_logging.PRINT_INTERVAL, average_returns[-1],
                          _params_logging.PRINT_INTERVAL, average_scores[-1]))
            with open("returnFile_running.dat", "a+") as returnFile:
                returnFile.write(format(n_epi) + "," + format(policy_iterations) + "," +
                                 '{:.1f}'.format(average_returns[-1]) + "," + '{:.1f}'.format(
                    average_scores[-1]) + "\n")

        if n_epi % _params_logging.SAVE_INTERVAL == 0:
            torch.save(ppo.policy.state_dict(), "decision_maker_actor_critic_episode_{}".format(n_epi) + ".pt")
            evaluate_decision_maker.main(env, ppo, n_epi, _params_agent, validation_mask=True)

    # display results
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

    env.close()


if __name__ == '__main__':
    main()
