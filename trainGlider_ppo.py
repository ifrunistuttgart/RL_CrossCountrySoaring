import gym
import glider
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import normal, MultivariateNormal
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

        self.action_var = torch.full((self._params_model.DIM_OUT,),
                                     self._params_rl.SIGMA * self._params_rl.SIGMA).to(device)

    def actor(self, observation):
        x = torch.tanh(self.hiddenLayer_in(observation))
        for ii in range(self._params_model.NUM_HIDDEN - 1):
            x = torch.tanh(self.hiddenLayer_internal(x))
        x = self.outputLayer_actor(x)

        batchsize = x.shape[0]
        pi = torch.Tensor(self._params_model.DIM_OUT * batchsize * (self._params_rl.AUTO_EXPLORATION + 1))
        pi[0:self._params_model.DIM_OUT * batchsize] = \
            torch.tanh(x[0:(self._params_model.DIM_OUT * batchsize)]).reshape(-1)
        if self._params_rl.AUTO_EXPLORATION:
            pi[(self._params_model.DIM_OUT * batchsize)] = x[-(self._params_model.DIM_OUT * batchsize):]
        return pi

    def critic(self, observation):
        x = torch.tanh(self.hiddenLayer_in(observation))
        for ii in range(self._params_model.NUM_HIDDEN - 1):
            x = torch.tanh(self.hiddenLayer_internal(x))
        v = self.outputLayer_critic(x)
        return v

    def act(self, observation, validation=False):
        pi = self.actor(observation)
        action_mean = pi[0:(self._params_model.DIM_OUT * observation.shape[0])]

        if self._params_rl.AUTO_EXPLORATION and (not validation):
            action_logStd = pi[-(self._params_model.DIM_OUT * observation.shape[0]):]
            action_var = torch.exp(action_logStd)**2
            action_var = action_var.reshape(-1) #TODO assure that this works for DIM_OUT > 1
        elif (not self._params_rl.AUTO_EXPLORATION) and (not validation):
            action_var = self.action_var
        else:
            action_var = torch.full((self._params_model.DIM_OUT,), 1e-6)

        dist = MultivariateNormal(action_mean.unsqueeze(-1), torch.diag(action_var).to(device))
        action = dist.sample()
        logprob = dist.log_prob(action)

        return action, logprob

class PPO:
    def __init__(self):
        self._params_model = params_model()
        self._params_rl = params_rl()
        self.memory = []

        self.policy = ActorCritic()
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=self._params_rl.LEARNING_RATE, betas=(0.9, 0.999))
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

    def update(self):

        s, a, r, s_prime, done_mask, logprob = self.make_batch()

        ##### K_EPOCH times policy optimization ######
        for _ in range(self._params_rl.K_EPOCH):

            # advantage estimation
            td_target = r + self._params_rl.GAMMA * self.policy.critic(s_prime) * done_mask
            delta = td_target - self.policy.critic(s)
            delta = delta.detach().numpy()
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self._params_rl.GAMMA * self._params_rl.LMBDA * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            # ppo ratio
            _, logprob_new = self.policy.act(s)
            ratio = torch.exp(logprob_new.unsqueeze(-1) - logprob)

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

    env.close()

if __name__ == '__main__':
    main()