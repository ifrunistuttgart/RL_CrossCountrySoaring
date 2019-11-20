import gym
import glider
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal
import numpy as np
import datetime
import matplotlib.pyplot as plt

from params import *
import evaluateGlider

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()

        self._params_model = params_model()
        self._params_rl = params_rl()

        self.hiddenLayer_in = nn.Linear(self._params_model.DIM_IN, self._params_model.DIM_HIDDEN)
        self.hiddenLayer_internal = nn.Linear(self._params_model.DIM_HIDDEN, self._params_model.DIM_HIDDEN)
        self.outputLayer_actor = nn.Linear(self._params_model.DIM_HIDDEN,
                                           self._params_model.DIM_OUT * (self._params_rl.AUTO_EXPLORATION + 1))
        self.outputLayer_critic = nn.Linear(self._params_model.DIM_HIDDEN, 1)

    def actor(self, observation):
        x = torch.tanh(self.hiddenLayer_in(observation))
        for ii in range(self._params_model.NUM_HIDDEN - 1):
            x = torch.tanh(self.hiddenLayer_internal(x))
        x = self.outputLayer_actor(x)

        batchsize = 1 if observation.ndimension() == 1 else observation.shape[0]
        output = x.reshape(batchsize, self._params_model.DIM_OUT * (self._params_rl.AUTO_EXPLORATION + 1))

        pi = torch.empty(batchsize, self._params_model.DIM_OUT * (self._params_rl.AUTO_EXPLORATION + 1))
        pi[0:batchsize, 0:self._params_model.DIM_OUT] = torch.tanh(output[0:batchsize, 0:self._params_model.DIM_OUT])
        if self._params_rl.AUTO_EXPLORATION:
            pi[0:batchsize, -self._params_model.DIM_OUT:] = output[0:batchsize, -self._params_model.DIM_OUT:]
        return pi

    def critic(self, observation):
        x = torch.tanh(self.hiddenLayer_in(observation))
        for ii in range(self._params_model.NUM_HIDDEN - 1):
            x = torch.tanh(self.hiddenLayer_internal(x))
        v = self.outputLayer_critic(x)
        return v

    def act(self, observation, validation=False):
        pi = self.actor(observation)
        action_mean = pi[:, 0:self._params_model.DIM_OUT]

        if self._params_rl.AUTO_EXPLORATION and (not validation):
            action_logstd = pi[:, -self._params_model.DIM_OUT:]
            dist = Normal(action_mean, torch.exp(action_logstd))
        elif (not self._params_rl.AUTO_EXPLORATION) and (not validation):
            dist = Normal(action_mean, self._params_rl.SIGMA)
        else:
            dist = Normal(action_mean, 1e-6)

        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob

class PPO:
    def __init__(self):
        self._params_model = params_model()
        self._params_rl = params_rl()
        self.memory = []
        self.policy = ActorCritic()
        # self.policy.load_state_dict(torch.load("actor_critic_network_final_20-November-2019_10-11.pt"))

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._params_rl.LEARNING_RATE)
        self.MseLoss = nn.MSELoss()

    def put_data(self, transition):
        self.memory.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, logprob_lst, done_lst = [], [], [], [], [], []

        for transition in self.memory:
            s, a, r, s_prime, logprob, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            logprob_lst.append([logprob])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, logprob = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                            torch.tensor(done_lst, dtype=torch.float), torch.tensor(logprob_lst)
        self.memory = []
        return s, a, r, s_prime, done_mask, logprob

    def evaluate(self, observation, action):
        pi = self.policy.actor(observation)
        action_mean = pi[:, 0:self._params_model.DIM_OUT]

        if self._params_rl.AUTO_EXPLORATION:
            action_logstd = pi[:, -self._params_model.DIM_OUT:]
            dist = Normal(action_mean, torch.exp(action_logstd))
        else:
            dist = Normal(action_mean, self._params_rl.SIGMA)

        logprob = dist.log_prob(action)
        return logprob

    def update(self):
        s, a, r, s_prime, done_mask, logprob = self.make_batch()

        for _ in range(self._params_rl.N_UPDATE):
            # GAE - advantage estimation
            td_target = r + self._params_rl.GAMMA * self.policy.critic(s_prime) * done_mask
            delta = td_target - self.policy.critic(s)
            delta = delta.detach().numpy()
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:  # TODO: is this valid for N_EPPERITER > 1?
                advantage = self._params_rl.GAMMA * self._params_rl.LAMBDA * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            # ppo ratio
            logprob_new = self.evaluate(s, a)
            ratio = torch.exp(logprob_new - logprob)

            # surrogate loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self._params_rl.EPS_CLIP, 1 + self._params_rl.EPS_CLIP) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.policy.critic(s), td_target.detach())

            # gradient step
            self.policy.train()
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

def main():
    env = gym.make('glider2D-v1')

    ppo = PPO()

    _params_rl      = params_rl()
    _params_model   = params_model()
    _params_task    = params_task()
    _params_sim     = params_sim()
    _params_glider  = params_glider()
    _params_physics = params_physics()
    _params_logging = params_logging()

    returns = []
    average_returns = []
    evaluateGlider.main(env, ppo.policy, 0)

    for n_epi in range(int(_params_rl.N_EPOCH)):
        s = env.reset()
        s = env.standardize_observations(s)
        done = False
        ret = 0

        while not done:
            action, logprob = ppo.policy.act(torch.from_numpy(s).float())
            s_prime, r, done, info = env.step(action[0].numpy())
            ppo.put_data((s, action, r, s_prime, logprob, done))
            s = s_prime
            ret += r

            if done:
                returns.append(ret)
                break

        if n_epi % _params_rl.N_EPPERITER == 0 and n_epi != 0:
            ppo.update()

        n_mean = _params_logging.PRINT_INTERVAL if len(returns) >= _params_logging.PRINT_INTERVAL else len(returns)
        average_returns.append(np.convolve(returns[-n_mean:], np.ones((n_mean,)) / n_mean, mode='valid')[0])

        if n_epi % _params_logging.PRINT_INTERVAL == 0 and n_epi != 0:
            print("# episode: {}, avg return over last {} episodes: {:.1f}".format(n_epi,
                                                                                   _params_logging.PRINT_INTERVAL,
                                                                                   average_returns[-1]))
        if n_epi % _params_logging.SAVE_INTERVAL == 0 and n_epi != 0:
            torch.save(ppo.policy.state_dict(), "actor_critic_network_episode_{}".format(n_epi) + ".pt")
            evaluateGlider.main(env, ppo.policy, n_epi)

    now = datetime.datetime.now()
    torch.save(ppo.policy.state_dict(), "actor_critic_network_final_" + now.strftime("%d-%B-%Y_%H-%M") + ".pt")
    evaluateGlider.main(env, ppo.policy, "final", True)
    f = open("parameters_" + now.strftime("%d-%B-%Y_%H-%M") + ".txt", "a+")
    f.write(format(vars(_params_rl)) + "\n" + format(vars(_params_model)) + "\n" + format(vars(_params_task)) + "\n" +
            format(vars(_params_sim)) + "\n" + format(vars(_params_glider)) + "\n" +
            format(vars(_params_physics)) + "\n" + format(vars(_params_logging)))
    f.close()

    plt.figure("average returns")
    plt.plot(average_returns)
    plt.ylabel("average returns (-)")
    plt.xlabel("episodes (-)")
    plt.grid(True)
    plt.savefig("average_returns_" + now.strftime("%d-%B-%Y_%H-%M") + ".png")
    plt.show()

    env.close()

if __name__ == '__main__':
    main()