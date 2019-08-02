import gym
import glider
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import normal
import matplotlib.pyplot as plt

# TODO: via class definition import -> https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules
class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.inputLayer = nn.Linear(4, 64)
        self.hiddenLayer = nn.Linear(64, 64)
        self.outputLayer_mu = nn.Linear(64, 1)
        self.outputLayer_sig = nn.Linear(64, 1)
        self.outputLayer_v = nn.Linear(64, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x):
        x = torch.tanh(self.inputLayer(x))
        x = torch.tanh(self.hiddenLayer(x))
        mu_a = torch.tanh(self.outputLayer_mu(x))
        sig_a = self.outputLayer_sig(x)
        return mu_a, sig_a

    def v(self, x):
        x = torch.tanh(self.inputLayer(x))
        x = torch.tanh(self.hiddenLayer(x))
        v = self.outputLayer_v(x)
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

            pi = self.pi(s)
            mu = pi[0]
            var = torch.pow(torch.exp(pi[1]), 2)
            var_a = torch.pow(torch.exp(pi_a[1]), 2)
            # var = torch.pow(torch.tensor(np.pi/180), 2).double()  # constantly one deg std-dev for exploration
            prob_new = (1/torch.exp(pi[1]))*torch.exp(-torch.pow(a - mu, 2)/(2*var))  # TODO: express this more compact
            prob_old = (1/torch.exp(pi_a[1]))*torch.exp(-torch.pow(a - pi_a[0], 2)/(2*var_a))
            ratio = torch.exp(torch.log(prob_new) - torch.log(prob_old))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

env = gym.make('glider-v0')
model = torch.load("actor_critic_network_01-August-2019_18-33.pt")
model.eval()


state = env.reset()
observation = env.standardize_observations(state)
done = False
time = 0
pos_list = [[state[0], -state[1]]]
act_list = []

while not done:

    # print(state)

    # evaluate and apply policy
    pi = model.pi(torch.from_numpy(state).float())
    action = pi[0]
    observation, reward, done, info = env.step(action.data[0].numpy())

    # scale to actual state
    state = observation * env._obs_std + env._obs_mean
    pos_list.append([state[0], -state[1]])

    # scale to actual AoA
    action = env.scale_action(action.detach().numpy())
    act_list.append([action])

    time += env.action_dt

print("Episode finished after {} seconds".format(time))

# plot position and control trajectory
plt.figure()
plt.subplot(211)
plt.plot(np.array(pos_list)[:, 0], np.array(pos_list)[:, 1])
plt.subplot(212)
plt.plot(np.array(pos_list)[0:-1, 0], (180/np.pi)*np.array(act_list).reshape(len(act_list), 1))
plt.show()

env.close()