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
# import evaluateGlider


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
        s_lst, a_lst, r_lst, s_prime_lst, pi_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, pi_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            pi_a_lst.append([pi_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s, a, r, s_prime, done_mask, pi_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(pi_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, pi_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, pi_a = self.make_batch()

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
            mu = pi[0]
            # var = torch.pow(torch.exp(pi[1]), 2)
            # var_a = torch.pow(torch.exp(pi_a[1]), 2)
            var = torch.pow(torch.tensor(self._params_rl.SIGMA), 2).double()
            # prob_new = (1/torch.exp(pi[1]))*torch.exp(-torch.pow(a - mu, 2)/(2*var))  # TODO: express this more compact
            # prob_old = (1/torch.exp(pi_a[1]))*torch.exp(-torch.pow(a - pi_a[0], 2)/(2*var_a))

            prob_new = torch.exp(-torch.pow(a - mu, 2)/(2*var))
            prob_old = torch.exp(-torch.pow(a - pi_a, 2)/(2*var))

            ratio = torch.exp(torch.log(prob_new) - torch.log(prob_old))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self._params_rl.EPS_CLIP, 1 + self._params_rl.EPS_CLIP) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.critic(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
def main():
    env = gym.make('glider-v0')
    model = PPO()
    model.eval()

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
            # pdf = normal.Normal(pi[0], torch.exp(pi[1]))
            a = pdf.sample()
            s_prime, r, done, info = env.step(a.numpy())

            model.put_data((s, a, r, s_prime, pi, done))
            s = s_prime
            ret += r

            if done:
                returns.append(ret)
                break

        model.train_net()

        n_mean = _params_logging.PRINT_INTERVAL if len(returns) >= _params_logging.PRINT_INTERVAL else len(returns)
        average_returns.append(np.convolve(returns[-n_mean:], np.ones((n_mean,)) / n_mean, mode='valid')[0])

        if n_epi % _params_logging.PRINT_INTERVAL == 0 and n_epi != 0:
            print("# of episode: {}, avg return: {:.1f}".format(n_epi, average_returns[-1]))

        if n_epi % _params_logging.SAVE_INTERVAL == 0 and n_epi != 0:
            torch.save(model, "actor_critic_network_episode_{}".format(n_epi) + ".pt")

    now = datetime.datetime.now()
    torch.save(model, "actor_critic_network_final_" + now.strftime("%d-%B-%Y_%H-%M") + ".pt")

    plt.figure("average returns")
    plt.plot(average_returns)
    plt.ylabel("average returns (-)")
    plt.xlabel("episodes (-)")
    plt.grid(True)

    evaluateGlider.main(model)
    env.close()

if __name__ == '__main__':
    main()