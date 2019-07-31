import gym
import glider
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.distributions import Categorical
from torch.distributions import normal
import numpy as np
import datetime
import matplotlib.pyplot as plt


# Hyperparameters
learning_rate = 1e-5
gamma         = 0.99
lmbda         = 1.0
eps_clip      = 0.2
K_epoch       = 5

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        self.inputLayer = nn.Linear(4, 64)
        self.hiddenLayer = nn.Linear(64, 64)
        self.outputLayer_pi = nn.Linear(64, 1)
        self.outputLayer_v = nn.Linear(64, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x):
        x = torch.tanh(self.inputLayer(x))
        x = torch.tanh(self.hiddenLayer(x))
        prob = torch.tanh(self.outputLayer_pi(x))
        return prob
    
    def v(self, x):
        x = torch.tanh(self.inputLayer(x))
        x = torch.tanh(self.hiddenLayer(x))
        v = self.outputLayer_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, mu_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, mu_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            mu_a_lst.append([mu_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s, a, r, s_prime, done_mask, mu_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(mu_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, mu_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, mu_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            mu = self.pi(s)
            var = torch.pow(torch.tensor(np.pi/180), 2).double()  # constantly one deg std-dev for exploration
            pi_a = torch.exp(-torch.pow(a - mu, 2)/(2*var))
            pi_old = torch.exp(-torch.pow(a - mu_a, 2)/(2*var))
            ratio = torch.exp(torch.log(pi_a) - torch.log(pi_old))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
def main():
    env = gym.make('glider-v0')
    model = PPO()

    print_interval = 20
    returns = []
    average_returns = [0]

    for n_epi in range(100000):
        s = env.reset()
        s = env.standardize_observations(s)
        done = False
        ret = 0

        while not done:
            mu_a = model.pi(torch.from_numpy(s).float())
            pdf = normal.Normal(mu_a, (np.pi/180))
            a = pdf.sample()
            s_prime, r, done, info = env.step(a.numpy())

            model.put_data((s, a, r, s_prime, mu_a, done))
            s = s_prime
            ret += r

            if done:
                returns.append(ret)
                break

        model.train_net()

        n_mean = print_interval if len(returns) >= print_interval else len(returns)
        average_returns.append(np.convolve(returns[-n_mean:], np.ones((n_mean,)) / n_mean, mode='valid')[0])

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode: {}, avg return: {:.1f}".format(n_epi, average_returns[-1]))

    now = datetime.datetime.now()
    torch.save(model, "actor_critic_network_" + now.strftime("%d-%B-%Y_%H-%M") + ".pt")

    plt.plot(average_returns)
    plt.show()

    env.close()

if __name__ == '__main__':
    main()