import datetime
import time
import os
import shutil

import gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import evaluateGlider
from params import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()

        self._params_model = params_model()
        self._params_rl = params_rl()

        self.lstm = nn.LSTM(input_size=3, hidden_size=(self._params_model.DIM_HIDDEN - self._params_model.DIM_IN),
                            batch_first=True)
        self.h_old = torch.zeros(1, 1, (self._params_model.DIM_HIDDEN - self._params_model.DIM_IN))
        self.c_old = torch.zeros(1, 1, (self._params_model.DIM_HIDDEN - self._params_model.DIM_IN))

        self.hiddenLayer_internal = nn.Linear(self._params_model.DIM_HIDDEN, self._params_model.DIM_HIDDEN)
        self.outputLayer = nn.Linear(self._params_model.DIM_HIDDEN,
                                     self._params_model.DIM_OUT * (self._params_rl.AUTO_EXPLORATION + 1))

    def forward(self, observation, updraft_obs):
        # evaluate lstm
        _, (h_n, c_n) = self.lstm(updraft_obs, (self.h_old.to(device), self.c_old.to(device)))

        # reset initial parameters (necessary for evaluation, only)
        self.h_old = h_n
        self.c_old = c_n

        # forward lstm hidden state for t = seq_len
        x = torch.tanh(h_n[-1])

        # concatenate std. observation & lstm output
        z = torch.cat((observation, x), dim=-1)

        # evaluate feedforward net
        for ii in range(self._params_model.NUM_HIDDEN - 1):
            z = torch.tanh(self.hiddenLayer_internal(z))
        output = self.outputLayer(z)

        return output

    def evaluate(self, observation, updraft_obs, validation=False):
        rawout = self.forward(observation.to(device), updraft_obs.to(device))

        pi = torch.empty(rawout.shape)
        pi[:, 0:self._params_model.DIM_OUT] = torch.tanh(rawout[:, 0:self._params_model.DIM_OUT])
        if self._params_rl.AUTO_EXPLORATION:
            pi[:, -self._params_model.DIM_OUT:] = rawout[:, -self._params_model.DIM_OUT:]

        action_mean = pi[:, 0:self._params_model.DIM_OUT]

        if self._params_rl.AUTO_EXPLORATION and (not validation):
            action_logstd = pi[:, -self._params_model.DIM_OUT:]
            distribution = Normal(action_mean.to(device), torch.exp(action_logstd).to(device))
        elif (not self._params_rl.AUTO_EXPLORATION) and (not validation):
            distribution = Normal(action_mean.to(device), self._params_rl.SIGMA)
        else:
            distribution = Normal(action_mean, 1e-6)

        return distribution

class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()

        self._params_model = params_model()

        self.lstm = nn.LSTM(input_size=3, hidden_size=(self._params_model.DIM_HIDDEN - self._params_model.DIM_IN),
                            batch_first=True)
        self.h_old = torch.zeros(1, 1, (self._params_model.DIM_HIDDEN - self._params_model.DIM_IN))
        self.c_old = torch.zeros(1, 1, (self._params_model.DIM_HIDDEN - self._params_model.DIM_IN))

        self.hiddenLayer_internal = nn.Linear(self._params_model.DIM_HIDDEN, self._params_model.DIM_HIDDEN)
        self.outputLayer = nn.Linear(self._params_model.DIM_HIDDEN, 1)

    def forward(self, observation, updraft_obs):
        # evaluate lstm
        _, (h_n, c_n) = self.lstm(updraft_obs, (self.h_old.to(device), self.c_old.to(device)))

        # reset initial parameters (necessary for evaluation, only)
        self.h_old = h_n
        self.c_old = c_n

        # forward lstm hidden state for t = seq_len
        x = torch.tanh(h_n[-1])

        # concatenate std. observation & lstm output
        z = torch.cat((observation, x), dim=-1)

        # evaluate feedforward net
        for ii in range(self._params_model.NUM_HIDDEN - 1):
            z = torch.tanh(self.hiddenLayer_internal(z))
        output = self.outputLayer(z)

        return output

class Model:
    def __init__(self, isActor):
        self._isActor = isActor
        self._params_rl = params_rl()

        # Init Net
        if self._isActor:
            self.Net = ActorNet().to(device)
            self.optimizer = torch.optim.Adam(self.Net.parameters(), lr=self._params_rl.LEARNING_RATE)
        else:
            self.Net = CriticNet().to(device)
            self.optimizer = torch.optim.Adam(self.Net.parameters(), lr=self._params_rl.LEARNING_RATE)

    def fit(self, loss):
        self.Net.train
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)  # TODO: "retain_graph" should be omitted, somehow
        self.optimizer.step()

    def resetLSTM(self, batchsize=1):
        self.Net.h_old = torch.zeros(1, batchsize, (self.Net._params_model.DIM_HIDDEN - self.Net._params_model.DIM_IN))
        self.Net.c_old = torch.zeros(1, batchsize, (self.Net._params_model.DIM_HIDDEN - self.Net._params_model.DIM_IN))


class PPO:
    def __init__(self):
        self._params_model = params_model()
        self._params_rl = params_rl()
        self.memory = []

        self.actor = Model(isActor=True)
        self.critic = Model(isActor=False)
        # self.actor.Net.load_state_dict(torch.load("actor_network_episode_2800.pt"))
        # self.critic.Net.load_state_dict(torch.load("critic_network_episode_2800.pt"))

    def put_data(self, transition):
        self.memory.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, logprob_lst, done_lst = [], [], [], [], [], []

        up_obs_input = torch.zeros(len(self.memory), len(self.memory), 3).to(device)
        up_obs_seq = []
        k = 0

        for transition in self.memory:
            s, a, r, s_prime, logprob, done, updraft_obs = transition

            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            logprob_lst.append(logprob)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

            up_obs_seq.append(updraft_obs)
            up_obs_input[k, 0:(k+1), :] = torch.FloatTensor(up_obs_seq)
            k += 1

        s, a, r, s_prime, done_mask, logprob = torch.tensor(s_lst, dtype=torch.float).to(device).detach(), \
                                               torch.tensor(a_lst, dtype=torch.float).to(device).detach(), \
                                               torch.tensor(r_lst, dtype=torch.float).to(device).detach(), \
                                               torch.tensor(s_prime_lst, dtype=torch.float).to(device).detach(), \
                                               torch.tensor(done_lst, dtype=torch.float).to(device).detach(), \
                                               torch.tensor(logprob_lst, dtype=torch.float).to(device).detach()

        # pack padded variable length sequence as lstm input
        up_obs_input_packed = pack_padded_sequence(up_obs_input.to(device).detach(),
                                                   torch.arange(1, len(self.memory)+1), batch_first=True,
                                                   enforce_sorted=False)

        self.memory = []
        return s, a, r, s_prime, done_mask, logprob, up_obs_input_packed

    def act(self, observation, updraft_obs, validation=False):
        dist = self.actor.Net.evaluate(observation, updraft_obs, validation)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob.detach()

    def evaluate(self, observation, updraft_obs, action):
        dist = self.actor.Net.evaluate(observation, updraft_obs)
        logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return logprob, dist_entropy

    def update(self):
        s, a, r, s_prime, done_mask, logprob, updraft_obs = self.make_batch()
        self.actor.resetLSTM(len(s))
        self.critic.resetLSTM(len(s))

        returnToGo = []
        discounted_reward = 0
        for reward, mask in zip(reversed(r.tolist()), reversed(done_mask.tolist())):
            discounted_reward = reward[0] + (self._params_rl.GAMMA*discounted_reward*mask[0])
            returnToGo.insert(0, discounted_reward)

        # normalize
        returnToGo = torch.tensor(returnToGo).to(device)
        returnToGo = (returnToGo - returnToGo.mean()) / (returnToGo.std() + 1e-5)
        returnToGo = returnToGo.unsqueeze(1)

        for _ in range(self._params_rl.N_UPDATE):
            # TODO: GAE - advantage estimation
            # td_target = r + self._params_rl.GAMMA*self.policy.critic(s_prime)*done_mask
            # delta = td_target - self.policy.critic(s)
            # advantage = delta
            # delta = delta.detach().numpy()
            # advantage_lst = []
            # advantage = 0.0
            # for index in reversed(range(len(delta))):
            #     advantage = self._params_rl.GAMMA*self._params_rl.LAMBDA*advantage*done_mask[index].item()\
            #                 + delta[index].item()
            #     advantage_lst.append([advantage])
            # advantage_lst.reverse()
            # advantage = torch.tensor(advantage_lst, dtype=torch.float)

            # ppo ratio
            logprob_new, dist_entropy = self.evaluate(s, updraft_obs, a)
            ratio = torch.exp(logprob_new - logprob)

            # surrogate loss
            state_value = self.critic.Net.forward(s, updraft_obs)
            advantage = returnToGo - state_value.detach()
            surr1 = ratio*advantage
            surr2 = torch.clamp(ratio, 1 - self._params_rl.EPS_CLIP, 1 + self._params_rl.EPS_CLIP)*advantage
            loss_actor = -torch.min(surr1, surr2).mean()
            loss_critic = F.mse_loss(state_value, returnToGo).sum()

            # gradient step
            self.actor.fit(loss_actor)
            self.critic.fit(loss_critic)


def main():

    # set up training
    tstart = time.time()
    env = gym.make('glider3D-v0')
    ppo = PPO()

    # load parameters
    _params_rl      = params_rl()
    _params_model   = params_model()
    _params_task    = params_task()
    _params_sim     = params_sim()
    _params_glider  = params_glider()
    _params_physics = params_physics()
    _params_logging = params_logging()
    _params_wind    = params_wind()

    # create folder to store data for the experiment running
    experimentID = 1
    dirName = "experiment_running_{}".format(experimentID)
    while os.path.exists(dirName):
        experimentID += 1
        dirName = dirName.rsplit('_', 1)[0] + "_{}".format(experimentID)
    os.mkdir(dirName)
    shutil.copytree(os.getcwd(), os.path.join(dirName, "Sources_unzipped"),
                    ignore=shutil.ignore_patterns('experiment*', 'archive', 'tests', '.git*',
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

    # initialize further variables
    returns = []
    average_returns = []
    policy_iterations = 0

    # showcase behavior before training takes place
    evaluateGlider.main(env, ppo, 0)

    # actual training
    for n_epi in range(1, int(_params_rl.N_EPOCH + 1)):
        env.reset()
        ppo.actor.resetLSTM()
        ppo.critic.resetLSTM()
        s = env.state2observation()
        updraft_obs = env.get_updraft_obs()
        done = False
        ret = 0

        while not done:
            action, logprob = ppo.act(torch.FloatTensor(s).view(1, -1),               # timesteps x observations
                                      torch.FloatTensor(updraft_obs).view(1, -1, 3))  # timesteps x seq_len x features
            s_prime, r, done, info = env.step(action.cpu().data.numpy().flatten())
            updraft_obs = env.get_updraft_obs()
            ppo.put_data((s, action.cpu().data.numpy().flatten(), r, s_prime, logprob.cpu().data.numpy().flatten(),
                          done, updraft_obs))
            s = s_prime
            ret += r

            if done:
                returns.append(ret)
                break

        if n_epi % _params_rl.N_EPPERITER == 0:
            ppo.update()
            policy_iterations += 1

        n_mean = _params_logging.PRINT_INTERVAL if len(returns) >= _params_logging.PRINT_INTERVAL else len(returns)
        average_returns.append(np.convolve(returns[-n_mean:], np.ones((n_mean,)) / n_mean, mode='valid')[0])

        if n_epi % _params_logging.PRINT_INTERVAL == 0:
            print("# episode: {}, vertices hit: {}, avg return over last {} episodes: {:.1f}"
                  .format(n_epi, (env.lapCounter*3 + env.vertexCounter),
                          _params_logging.PRINT_INTERVAL, average_returns[-1]))

            with open("returnFile_running.dat", "a+") as returnFile:
                returnFile.write(format(n_epi) + "," + format(policy_iterations) + ","
                                 + '{:.1f}'.format(average_returns[-1]) + "\n")

        if n_epi % _params_logging.SAVE_INTERVAL == 0:
            torch.save(ppo.actor.Net.state_dict(), "actor_network_episode_{}".format(n_epi) + ".pt")
            torch.save(ppo.critic.Net.state_dict(), "critic_network_episode_{}".format(n_epi) + ".pt")
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
    torch.save(ppo.actor.Net.state_dict(), "actor_network_final_" + now.strftime("%d-%B-%Y_%H-%M") + ".pt")
    torch.save(ppo.critic.Net.state_dict(), "critic_network_final_" + now.strftime("%d-%B-%Y_%H-%M") + ".pt")

    # rename parameter file consistently
    os.rename(parameterFile.name, "parameters_" + now.strftime("%d-%B-%Y_%H-%M") + ".txt")

    # rename return file consistently
    returnFile.close()
    os.rename(returnFile.name, "average_returns_" + now.strftime("%d-%B-%Y_%H-%M") + ".dat")

    env.close()

if __name__ == '__main__':
    main()