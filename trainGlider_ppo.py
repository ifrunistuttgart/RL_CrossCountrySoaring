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

class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()

        self._params_model = params_model()
        self._params_rl = params_rl()

        self.hiddenLayer_in = nn.Linear(self._params_model.DIM_IN, self._params_model.DIM_HIDDEN)
        self.hiddenLayer_internal = nn.Linear(self._params_model.DIM_HIDDEN, self._params_model.DIM_HIDDEN)
        self.outputLayer = nn.Linear(self._params_model.DIM_HIDDEN,
                                     self._params_model.DIM_OUT * (self._params_rl.AUTO_EXPLORATION + 1))

    def forward(self, observation):
        x = torch.tanh(self.hiddenLayer_in(observation))
        for ii in range(self._params_model.NUM_HIDDEN - 1):
            x = torch.tanh(self.hiddenLayer_internal(x))
        output = self.outputLayer(x)
        return output

    def evaluate(self, observation, validation=False):
        rawout = self.forward(observation)

        batchsize = 1 if observation.ndimension() == 1 else observation.shape[0]
        output = rawout.reshape(batchsize, self._params_model.DIM_OUT * (self._params_rl.AUTO_EXPLORATION + 1))
        pi = torch.empty(batchsize, self._params_model.DIM_OUT * (self._params_rl.AUTO_EXPLORATION + 1))
        pi[0:batchsize, 0:self._params_model.DIM_OUT] = torch.tanh(output[0:batchsize, 0:self._params_model.DIM_OUT])
        if self._params_rl.AUTO_EXPLORATION:
            pi[0:batchsize, -self._params_model.DIM_OUT:] = output[0:batchsize, -self._params_model.DIM_OUT:]

        action_mean = pi[:, 0:self._params_model.DIM_OUT]

        if self._params_rl.AUTO_EXPLORATION and (not validation):
            action_logstd = pi[:, -self._params_model.DIM_OUT:]
            distribution = Normal(action_mean, torch.exp(action_logstd))
        elif (not self._params_rl.AUTO_EXPLORATION) and (not validation):
            distribution = Normal(action_mean, self._params_rl.SIGMA)
        else:
            distribution = Normal(action_mean, 1e-6)

        return distribution

class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()

        self._params_model = params_model()

        self.hiddenLayer_in = nn.Linear(self._params_model.DIM_IN, self._params_model.DIM_HIDDEN)
        self.hiddenLayer_internal = nn.Linear(self._params_model.DIM_HIDDEN, self._params_model.DIM_HIDDEN)
        self.outputLayer = nn.Linear(self._params_model.DIM_HIDDEN, 1)

    def forward(self, observation):
        x = torch.tanh(self.hiddenLayer_in(observation))
        for ii in range(self._params_model.NUM_HIDDEN - 1):
            x = torch.tanh(self.hiddenLayer_internal(x))
        output = self.outputLayer(x)
        return output

class Model:
    def __init__(self, isActor):
        self._isActor = isActor
        self._params_rl = params_rl()

        # Init Net
        if self._isActor:
            self.Net = ActorNet()
            self.optimizer = torch.optim.Adam(self.Net.parameters(), lr=self._params_rl.LEARNING_RATE)
        else:
            self.Net = CriticNet()
            self.optimizer = torch.optim.Adam(self.Net.parameters(), lr=self._params_rl.LEARNING_RATE)

        # self.Net = self.Net.double()

    def fit(self, loss):
        self.Net.train
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()


class PPO:
    def __init__(self):
        self._params_model = params_model()
        self._params_rl = params_rl()
        self.memory = []
        self.actor = Model(isActor=True)
        self.critic = Model(isActor=False)

        # self.MseLoss = nn.MSELoss()

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

    def act(self, observation, validation=False):
        dist = self.actor.Net.evaluate(observation, validation)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob

    def evaluate(self, observation, action):
        dist = self.actor.Net.evaluate(observation)
        logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return logprob, dist_entropy

    def update(self):
        s, a, r, s_prime, done_mask, logprob = self.make_batch()

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
            # GAE - advantage estimation
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
            logprob_new, dist_entropy = self.evaluate(s, a)
            ratio = torch.exp(logprob_new - logprob)

            # surrogate loss
            state_value = self.critic.Net.forward(s)
            advantage = returnToGo - state_value
            surr1 = ratio*advantage
            surr2 = torch.clamp(ratio, 1 - self._params_rl.EPS_CLIP, 1 + self._params_rl.EPS_CLIP)*advantage
            loss_actor = -torch.min(surr1, surr2).mean()
            loss_critic = F.mse_loss(state_value, returnToGo).sum()

            # gradient step
            self.actor.fit(loss_actor)
            self.critic.fit(loss_critic)

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
    evaluateGlider.main(env, ppo, 0)

    for n_epi in range(1, int(_params_rl.N_EPOCH + 1)):
        s = env.reset()
        s = env.standardize_observations(s)
        done = False
        ret = 0

        while not done:
            action, logprob = ppo.act(torch.from_numpy(s).float())
            s_prime, r, done, info = env.step(action[0].numpy())
            ppo.put_data((s, action, r, s_prime, logprob, done))
            s = s_prime
            ret += r

            if done:
                returns.append(ret)
                break

        if n_epi % _params_rl.N_EPPERITER == 0:
            ppo.update()

        n_mean = _params_logging.PRINT_INTERVAL if len(returns) >= _params_logging.PRINT_INTERVAL else len(returns)
        average_returns.append(np.convolve(returns[-n_mean:], np.ones((n_mean,)) / n_mean, mode='valid')[0])

        if n_epi % _params_logging.PRINT_INTERVAL == 0:
            print("# episode: {}, avg return over last {} episodes: {:.1f}".format(n_epi,
                                                                                   _params_logging.PRINT_INTERVAL,
                                                                                   average_returns[-1]))
        if n_epi % _params_logging.SAVE_INTERVAL == 0:
            torch.save(ppo.actor.Net.state_dict(), "actor_network_episode_{}".format(n_epi) + ".pt")
            torch.save(ppo.critic.Net.state_dict(), "critic_network_episode_{}".format(n_epi) + ".pt")
            evaluateGlider.main(env, ppo, n_epi)

    now = datetime.datetime.now()
    torch.save(ppo.actor.Net.state_dict(), "actor_network_final_" + now.strftime("%d-%B-%Y_%H-%M") + ".pt")
    torch.save(ppo.critic.Net.state_dict(), "critic_network_final_" + now.strftime("%d-%B-%Y_%H-%M") + ".pt")
    evaluateGlider.main(env, ppo, "final", True)
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