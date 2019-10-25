import gym
import glider
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import normal
import numpy as np
import datetime
import matplotlib.pyplot as plt

from params import *
import evaluateGlider

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self._params_rl = params_rl()
        self._params_model = params_model()
        
        self.hiddenLayer_in         = nn.Linear(self._params_model.DIM_IN, self._params_model.DIM_HIDDEN)
        self.hiddenLayer_internal   = nn.Linear(self._params_model.DIM_HIDDEN, self._params_model.DIM_HIDDEN)
        self.outputLayer_actor      = nn.Linear(self._params_model.DIM_HIDDEN, self._params_model.DIM_OUT)
        self.outputLayer_critic     = nn.Linear(self._params_model.DIM_HIDDEN, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self._params_rl.LEARNING_RATE)

    def actor(self, x):
        x = torch.tanh(self.hiddenLayer_in(x))
        for ii in range(self._params_model.NUM_HIDDEN - 1):
            x = torch.tanh(self.hiddenLayer_internal(x))
        pi = torch.tanh(self.outputLayer_actor(x))
        return pi
    
    def critic(self, x):
        x = torch.tanh(self.hiddenLayer_in(x))
        for ii in range(self._params_model.NUM_HIDDEN - 1):
            x = torch.tanh(self.hiddenLayer_internal(x))
        v = self.outputLayer_critic(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, a_logprob_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, a_logprob, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            a_logprob_lst.append([a_logprob])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, a_logprob = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                                 torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                                 torch.tensor(done_lst, dtype=torch.float), torch.tensor(a_logprob_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, a_logprob
        
    def train_net(self):
        s, a, r, s_prime, done_mask, a_logprob_old = self.make_batch()

        for i in range(self._params_rl.K_EPOCH):
            td_target = r + self._params_rl.GAMMA * self.critic(s_prime) * done_mask
            delta = td_target - self.critic(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self._params_rl.GAMMA * self._params_rl.LMBDA * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.actor(s)
            pdf = normal.Normal(pi[0], self._params_rl.SIGMA)
            a_new = pdf.sample()
            a_logprob_new = pdf.log_prob(a_new)

            ratio = torch.exp(a_logprob_new - a_logprob_old)  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self._params_rl.EPS_CLIP, 1 + self._params_rl.EPS_CLIP) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.critic(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
def main():
    env = gym.make('glider-v0')
    model = PPO()

    _params_rl = params_rl()
    _params_logging = params_logging()

    returns = []
    average_returns = []

    for n_epi in range(int(_params_rl.N_EPOCH)):
        s = env.reset()
        s = env.standardize_observations(s)
        done = False
        ret = 0

        while not done:
            pi = model.actor(torch.from_numpy(s).float())
            pdf = normal.Normal(pi[0], _params_rl.SIGMA)
            a = pdf.sample()
            s_prime, r, done, info = env.step(a.numpy())

            model.put_data((s, a, r, s_prime, pdf.log_prob(a), done))
            s = s_prime
            ret += r

            if done:
                returns.append(ret)
                break

        model.train_net()

        n_mean = _params_logging.PRINT_INTERVAL if len(returns) >= _params_logging.PRINT_INTERVAL else len(returns)
        average_returns.append(np.convolve(returns[-n_mean:], np.ones((n_mean,)) / n_mean, mode='valid')[0])

        if n_epi % _params_logging.PRINT_INTERVAL == 0 and n_epi != 0:
            print("# episode: {}, avg return over last {} episodes: {:.1f}".format(n_epi,
                                                                                   _params_logging.PRINT_INTERVAL,
                                                                                   average_returns[-1]))

        if n_epi % _params_logging.SAVE_INTERVAL == 0 and n_epi != 0:
            torch.save(model, "actor_critic_network_episode_{}".format(n_epi) + ".pt")

    now = datetime.datetime.now()
    torch.save(model, "actor_critic_network_final_" + now.strftime("%d-%B-%Y_%H-%M") + ".pt")

    plt.figure("average returns")
    plt.plot(average_returns)
    plt.ylabel("average returns (-)")
    plt.xlabel("episodes (-)")
    plt.grid(True)
    plt.savefig("average_returns" + now.strftime("%d-%B-%Y_%H-%M") + ".png")

    evaluateGlider.main(model)
    env.close()

if __name__ == '__main__':
    main()