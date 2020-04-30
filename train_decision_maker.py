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
        x, lstm_hidden = self.lstm(observation.to(device), lstm_hidden)

        # forward lstm hidden state for t = seq_len, i.e., "x"
        x = torch.tanh(x)

        # evaluate feedforward layer
        z = self.out_actor(x)

        # map output to [0, 1]
        p_exploit = 0.5*(torch.tanh(z) + 1)

        return p_exploit, lstm_hidden

    def critic(self, observation,  lstm_hidden):

        # evaluate lstm
        x, lstm_hidden = self.lstm(observation, lstm_hidden)

        # forward lstm hidden state for t = seq_len, i.e., "x"
        x = torch.tanh(x)

        # evaluate feedforward layer
        z = self.out_actor(x)

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

        return action_logprobs, torch.flatten(state_values, 1)

    def reset_lstm(self):
        return torch.zeros(1, 1, self._params_model.DIM_HIDDEN, device=device), \
               torch.zeros(1, 1, self._params_model.DIM_HIDDEN, device=device)

class PPO:
    def __init__(self, vertex_tracker, updraft_exploiter, environment):
        # instantiate parameters
        self._params_rl = params_decision_maker.params_rl()

        # instantiate memory to store data from rollout
        self.memory = []

        # instantiate actor-critic model
        self.policy = ActorCritic().to(device)
        # self.policy.load_state_dict(torch.load("decision_maker_actor_critic_final_15-April-2020_09-12.pt"))
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
    
    def select_action(self, state, lstm_hidden, validation_mask=False):
        # evaluate decision maker
        action_agent, action_agent_logprob, lstm_hidden = self.policy_old.act(state, lstm_hidden, validation_mask)
        p_exploit = np.clip(action_agent.item(), 0, 1)

        # evaluate vertex tracker
        action_vertex_tracker = self.vertex_tracker.select_action(self.env.get_full_observation)

        # evaluate updraft exploiter
        observation = torch.FloatTensor(self.env.get_rel_updraft_positions()).view(1, -1, 2).to(device)
        action_updraft_exploiter = self.updraft_exploiter.act(observation, None, True).cpu().data.numpy().flatten()

        # mix the actions
        action_env = p_exploit*action_updraft_exploiter + (1-p_exploit)*action_vertex_tracker

        return action_env, action_agent, action_agent_logprob, lstm_hidden

    def put_data(self, transition):
        self.memory.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, logprob_lst, done_lst,\
        lstm_h_in_lst, lstm_c_in_lst, lstm_h_out_lst, lstm_c_out_lst = [], [], [], [], [], [], [], [], [], []

        for transition in self.memory:
            s, a, r, s_prime, logprob, done, (lstm_h_in, lstm_c_in), (lstm_h_out, lstm_c_out) = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            logprob_lst.append(logprob)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

            lstm_h_in_lst.append(lstm_h_in)
            lstm_c_in_lst.append(lstm_c_in)
            lstm_h_out_lst.append(lstm_h_out)
            lstm_c_out_lst.append(lstm_c_out)

        s, a, r, s_prime, logprob, done_mask\
            = torch.FloatTensor(s_lst).reshape(self._params_rl.BATCHSIZE, 1, -1).to(device),\
              torch.FloatTensor(a_lst).to(device),\
              torch.FloatTensor(r_lst).to(device),\
              torch.FloatTensor(s_prime_lst).reshape(self._params_rl.BATCHSIZE, 1, -1).to(device),\
              torch.FloatTensor(logprob_lst).to(device),\
              torch.FloatTensor(done_lst).to(device)

        lstm_h_in, lstm_c_in = torch.squeeze(torch.stack(lstm_h_in_lst), 1),\
                               torch.squeeze(torch.stack(lstm_c_in_lst), 1)

        lstm_h_out, lstm_c_out = torch.squeeze(torch.stack(lstm_h_out_lst), 1),\
                                 torch.squeeze(torch.stack(lstm_c_out_lst), 1)

        self.memory = []
        return s, a, r, s_prime, logprob, done_mask, lstm_h_in, lstm_c_in, lstm_h_out, lstm_c_out
    
    def update(self):
        # convert list to tensor
        states, actions, rewards, next_states, logprobs, done_masks,\
        lstm_hiddens, lstm_states, next_lstm_hiddens, next_lstm_states = self.make_batch()

        # Generalized-Advantage-Estimation (GAE)
        next_values = self.policy_old.critic(next_states,
                                             (next_lstm_hiddens.permute(1, 0, 2), next_lstm_states.permute(1, 0, 2)))
        td_targets = rewards + self._params_rl.GAMMA * torch.squeeze(next_values, 1) * done_masks
        values = self.policy_old.critic(states, (lstm_hiddens.permute(1, 0, 2), lstm_states.permute(1, 0, 2)))
        deltas = td_targets - torch.squeeze(values, 1)
        deltas = deltas.detach().cpu().numpy()

        advantage_lst = []
        advantage = 0.0
        for item in deltas[::-1]:
            advantage = self._params_rl.GAMMA * self._params_rl.LAMBDA * advantage + item[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantages = torch.tensor(advantage_lst, dtype=torch.float).to(device)

        # put batch to data_loader
        train_loader = DataLoader(dataset=TensorDataset(actions, states, logprobs, lstm_hiddens, lstm_states,
                                                        advantages, td_targets),
                                  batch_size=self._params_rl.MINIBATCHSIZE, shuffle=True)

        # Optimize policy for K epochs:
        for _ in range(self._params_rl.K_EPOCH):
            for mini_batch in train_loader:
                # get sampled data in mini-batch and send them to device
                actions, states, logprobs, lstm_h_in, lstm_c_in, advantages, td_targets = mini_batch

                # evaluate policy for sampled states and actions
                logprobs_eval, state_values_eval = self.policy.evaluate(states, actions,
                                                                        (lstm_h_in.permute(1, 0, 2),
                                                                         lstm_c_in.permute(1, 0, 2)))

                # ppo ratio
                ratios = torch.exp(logprobs_eval - logprobs)

                # surrogate loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self._params_rl.EPS_CLIP, 1 + self._params_rl.EPS_CLIP) * advantages
                loss = -torch.min(surr1, surr2).mean() + 0.5*self.MseLoss(state_values_eval, td_targets.detach())

                # gradient step
                self.optimizer.zero_grad()
                loss.mean().backward(retain_graph=True)  # TODO: why doe I need retain_graph again?
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
    # evaluate_glider.main(env, ppo, 0, _params_agent, validation_mask=False)
    
    # training loop
    for n_epi in range(1, int(_params_rl.N_EPISODES + 1)):
        env.reset()
        lstm_in = ppo.policy.reset_lstm()
        done = False
        ret = 0
        state = env.get_observation()

        # with torch.no_grad():
        while not done:
            timestep_counter += 1

            # format state tensor
            # state = torch.FloatTensor(state.reshape(1, 1, -1))  # batch=1 x sequence=1 x observations

            # store: state, lstm_hidden_in
            # memory.states.append(state)
            # memory.lstm_hidden_in.append((lstm_hidden[0].cpu(), lstm_hidden[1].cpu()))

            # run policy
            action_env, action_agent, action_agent_logprob, lstm_out =\
                ppo.select_action(torch.FloatTensor(state.reshape(1, 1, -1)), lstm_in)
            next_state, reward, done, _ = env.step(action_env)
            ppo.put_data((state, action_agent, reward,
                          next_state, action_agent_logprob.cpu().data.numpy().flatten(),
                          done, lstm_in, lstm_out))

            # update variables
            state = next_state
            lstm_in = lstm_out
            ret += reward

            # # store: action, logprob, s_prime, reward, done mask, lstm_hidden_out
            # memory.actions.append(torch.FloatTensor([action]))
            # memory.logprobs.append(action_logprob.cpu())
            # memory.next_states.append(torch.FloatTensor(observation.reshape(1, 1, -1)))
            # memory.rewards.append(torch.FloatTensor([reward]))
            # memory.done_masks.append(torch.FloatTensor([not done]))
            # memory.lstm_hidden_out.append((lstm_hidden[0].cpu(), lstm_hidden[1].cpu()))

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