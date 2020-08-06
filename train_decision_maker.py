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
import utils.core as core


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()

        # instantiate parameters
        self._params_model = params_decision_maker.params_model()
        self._params_rl = params_decision_maker.params_rl()

        # set model elements: lstm net
        self.lstm           = nn.LSTM(input_size=self._params_model.DIM_IN, hidden_size=self._params_model.DIM_HIDDEN,
                                      batch_first=True)

        # set model elements: ffwd layers
        self.out_actor      = nn.Linear(self._params_model.DIM_HIDDEN, self._params_model.DIM_OUT)
        self.out_critic     = nn.Linear(self._params_model.DIM_HIDDEN, 1)

    def actor(self, observation, lstm_hidden):

        # evaluate lstm
        x = observation.reshape(-1, 1, self._params_model.DIM_IN).to(device)  # batch_size x seq_len x  input_size
        x, lstm_hidden = self.lstm(x, lstm_hidden)

        # forward lstm hidden state for t = seq_len, i.e., "x"
        x = x.reshape(-1, self._params_model.DIM_HIDDEN)  # batch_size x hidden_size
        x = torch.tanh(x)

        # evaluate feedforward layer
        z = self.out_actor(x)

        # map output to [0, 1]
        p_exploit = 0.5*(torch.tanh(z) + 1)

        return p_exploit, lstm_hidden

    def critic(self, observation, lstm_hidden):

        # evaluate lstm
        x = observation.reshape(-1, 1, self._params_model.DIM_IN).to(device)  # batch_size x seq_len x  input_size
        x, lstm_hidden = self.lstm(x, lstm_hidden)

        # forward lstm hidden state for t = seq_len, i.e., "x"
        x = x.reshape(-1, self._params_model.DIM_HIDDEN)  # batch_size x hidden_size
        x = torch.tanh(x)

        # evaluate feedforward layer
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
        # dist_entropy = dist.entropy()

        # evaluate critic for sampled states
        state_values = self.critic(sampled_state, sampled_lstm_hidden)

        return action_logprobs.flatten(), state_values.flatten()

    def reset_lstm(self):
        return torch.zeros(1, 1, self._params_model.DIM_HIDDEN, device=device), \
               torch.zeros(1, 1, self._params_model.DIM_HIDDEN, device=device)

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, batch_size, hidden_size, gamma=0.99, lam=0.95):
        self.obs_buf = torch.zeros(batch_size, obs_dim, dtype=torch.float32, device=device)
        self.act_buf = torch.zeros(batch_size, act_dim, dtype=torch.float32, device=device)
        self.adv_buf = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.ret_buf = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.val_buf = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.logp_buf = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.lstm_h_in_buf = torch.zeros(batch_size, hidden_size, dtype=torch.float32, device=device)
        self.lstm_c_in_buf = torch.zeros(batch_size, hidden_size, dtype=torch.float32, device=device)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, batch_size

    def store(self, obs, act, rew, val, logp, lstm_h_in, lstm_c_in):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.lstm_h_in_buf[self.ptr] = lstm_h_in
        self.lstm_c_in_buf[self.ptr] = lstm_c_in
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
        path_slice = slice(0, self.ptr)
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

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std() + 1e-5)

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, lstm_h_in=self.lstm_h_in_buf,
                    lstm_c_in=self.lstm_c_in_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in data.items()}

class PPO:
    def __init__(self, vertex_tracker, updraft_exploiter, environment):
        # instantiate parameters
        self._params_rl = params_decision_maker.params_rl()
        self._params_model = params_decision_maker.params_model()

        # instantiate buffer for rollout
        self.buffer = PPOBuffer(self._params_model.DIM_IN, self._params_model.DIM_OUT, self._params_rl.BATCHSIZE,
                                self._params_model.DIM_HIDDEN, self._params_rl.GAMMA, self._params_rl.LAMBDA)

        # instantiate actor-critic model
        self.policy = ActorCritic().to(device)
        # self.policy.load_state_dict(torch.load("decision_maker_actor_critic_final_15-April-2020_09-12.pt"))

        # setup optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._params_rl.LEARNING_RATE, betas=(0.9, 0.999))
        self.MseLoss = nn.MSELoss()

        # instantiate vertex tracker and updraft exploiter
        self.vertex_tracker = vertex_tracker
        self.updraft_exploiter = updraft_exploiter
        self.env = environment
    
    def select_action(self, state, lstm_hidden, validation_mask=False):
        # evaluate decision maker
        action_agent, action_agent_logprob, lstm_hidden = self.policy.act(state, lstm_hidden, validation_mask)
        p_exploit = np.clip(action_agent.item(), 0, 1)

        # evaluate critic
        state_value = self.policy.critic(state, lstm_hidden)

        # evaluate vertex tracker
        action_vertex_tracker = self.vertex_tracker.select_action(self.env.get_full_observation)

        # evaluate updraft exploiter
        observation = torch.FloatTensor(self.env.get_rel_updraft_positions()).view(1, -1, 2).to(device)
        action_updraft_exploiter = self.updraft_exploiter.act(observation, None, True).cpu().data.numpy().flatten()

        # mix the actions
        action_env = p_exploit*action_updraft_exploiter + (1-p_exploit)*action_vertex_tracker

        return action_env, action_agent, state_value, action_agent_logprob, lstm_hidden

    # def put_data(self, transition):
    #     self.memory.append(transition)
    #
    # def make_batch(self):
    #     s_lst, a_lst, r_lst, s_prime_lst, logprob_lst, done_lst,\
    #     lstm_h_in_lst, lstm_c_in_lst, lstm_h_out_lst, lstm_c_out_lst = [], [], [], [], [], [], [], [], [], []
    #
    #     for transition in self.memory:
    #         s, a, r, s_prime, logprob, done, (lstm_h_in, lstm_c_in), (lstm_h_out, lstm_c_out) = transition
    #
    #         s_lst.append(s)
    #         a_lst.append([a])
    #         r_lst.append([r])
    #         s_prime_lst.append(s_prime)
    #         logprob_lst.append(logprob)
    #         done_mask = 0 if done else 1
    #         done_lst.append([done_mask])
    #
    #         lstm_h_in_lst.append(lstm_h_in)
    #         lstm_c_in_lst.append(lstm_c_in)
    #         lstm_h_out_lst.append(lstm_h_out)
    #         lstm_c_out_lst.append(lstm_c_out)
    #
    #     s, a, r, s_prime, logprob, done_mask\
    #         = torch.FloatTensor(s_lst).to(device),\
    #           torch.FloatTensor(a_lst).to(device),\
    #           torch.FloatTensor(r_lst).to(device),\
    #           torch.FloatTensor(s_prime_lst).to(device),\
    #           torch.FloatTensor(logprob_lst).to(device),\
    #           torch.FloatTensor(done_lst).to(device)
    #
    #     lstm_h_in, lstm_c_in = torch.squeeze(torch.stack(lstm_h_in_lst), 1),\
    #                            torch.squeeze(torch.stack(lstm_c_in_lst), 1)
    #
    #     lstm_h_out, lstm_c_out = torch.squeeze(torch.stack(lstm_h_out_lst), 1),\
    #                              torch.squeeze(torch.stack(lstm_c_out_lst), 1)
    #
    #     self.memory = []
    #     return s, a, r, s_prime, logprob, done_mask, lstm_h_in, lstm_c_in, lstm_h_out, lstm_c_out
    
    def update(self):

        # get sampled data
        data = self.buffer.get()
        actions, observations, logprobs, lstm_hiddens, lstm_states, advantages, returns_to_go\
            = data['act'], data['obs'], data['logp'], data['lstm_h_in'], data['lstm_c_in'], data['adv'], data['ret']


        # put batch to data_loader
        train_loader = DataLoader(dataset=TensorDataset(actions, observations, logprobs, lstm_hiddens, lstm_states,
                                                        advantages, returns_to_go),
                                  batch_size=self._params_rl.MINIBATCHSIZE, shuffle=True)

        # Optimize policy for K epochs:
        for _ in range(self._params_rl.K_EPOCH):
            for mini_batch in train_loader:
                # get sampled data in mini-batch and send them to device
                act, obs, logp, lstm_h_in, lstm_c_in, adv, ret = mini_batch

                # evaluate policy for sampled states and actions
                logp_eval, val_eval = self.policy.evaluate(obs, act,
                                                           (lstm_h_in.reshape(1, self._params_rl.MINIBATCHSIZE, -1),
                                                            lstm_c_in.reshape(1, self._params_rl.MINIBATCHSIZE, -1)))

                # ppo ratio
                ratios = torch.exp(logp_eval - logp)

                # surrogate loss
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1 - self._params_rl.EPS_CLIP, 1 + self._params_rl.EPS_CLIP) * adv
                loss = -torch.min(surr1, surr2).mean() + 0.5 * self.MseLoss(val_eval, ret)

                # gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()


def main():
    # set up training
    tstart = time.time()
    env = gym.make('glider3D-v0', agent='decision_maker')

    # instantiate vertex tracker and updraft exploiter
    waypoint_controller = Controller_Wrapper(env)
    updraft_exploiter = model_updraft_exploiter.ActorCritic().to(device)
    updraft_exploiter.load_state_dict(torch.load("updraft_exploiter_actor_critic_final_08-April-2020_20-46.pt"))

    # instantiate agent
    ppo = PPO(waypoint_controller, updraft_exploiter, env)

    # load parameters
    _params_rl = params_decision_maker.params_rl()
    _params_agent = params_decision_maker.params_agent()
    _params_logging = params_decision_maker.params_logging()
    
    # create folder to store data for the experiment running
    experimentID = 1
    dirName = "{}_decision_maker_experiment_running".format(experimentID)
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
        lstm_in = ppo.policy.reset_lstm()
        done = False
        ret = 0
        observation = env.get_observation()

        while not done:
            with torch.no_grad():

                # run policy
                action_env, action_agent, state_value, action_agent_logprob, lstm_out =\
                    ppo.select_action(torch.FloatTensor(observation.reshape(1, 1, -1)), lstm_in)
                next_obs, reward, done, _ = env.step(action_env)
                ppo.buffer.store(torch.FloatTensor(observation).to(device), action_agent.flatten(),
                                 torch.FloatTensor([reward]),
                                 state_value.flatten(), action_agent_logprob.flatten(),
                                 lstm_in[0].flatten(), lstm_in[1].flatten())
                # ppo.put_data((state, action_agent, reward,
                #               next_state, action_agent_logprob.cpu().data.numpy().flatten(),
                #               done, lstm_in, lstm_out))

                # update variables
                observation = next_obs
                lstm_in = lstm_out
                ret += reward
                timestep_counter += 1

            # update policy every BATCHSIZE timesteps
            if timestep_counter % _params_rl.BATCHSIZE == 0:
                ppo.policy.train()
                ppo.update()
                ppo.policy.eval()
                timestep_counter = 0
                policy_iterations += 1

            # stop rollout if episode is completed
            if done:
                returns.append(ret)
                ppo.buffer.finish_path(torch.FloatTensor([0]).to(device))
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