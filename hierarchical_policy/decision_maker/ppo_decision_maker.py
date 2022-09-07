import torch
import torch.nn as nn
from scipy.signal import lfilter

import numpy as np
from torch.utils.data import Dataset, TensorDataset
from hierarchical_policy.decision_maker import params_decision_maker

from hierarchical_policy.decision_maker.model_decision_maker import DecisionMakerActorCritic

# Choose device here
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class PPOBuffer:
    """ Partly adopted from 'Spinning Up in Deep Reinforcement Learning' (2018) by Joshua Achiam, OpenAI
        https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py

        A buffer for storing trajectories experienced by a PPO agent interacting
        with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
        for calculating the advantages of state-action pairs.

        Attributes
        ----------

        obs_buf : Tensor
            Observation buffer

        act_buf : Tensor
            Action buffer

        adv_buf : Tensor
            Advantage estimation buffer

        rew_buf : Tensor
            Reward buffer

        ret_buf : Tensor
            Return buffer

        val_buf : Tensor
            State value buffer

        logp_buf : Tensor
            Log probability buffer

        lstm_h_in_buf : Tensor
            LSTM hidden state buffer

        lstm_c_in_buf : Tensor
            LSTM cell state buffer

        done_buf : Tensor
            Done flag buffer

        gamma : float
            Discount factor

        lam : float
            Lambda for Generalized Advantage Estimation (GAE)

        ptr: int
            Pointer to indicate storage position in buffer

        path_start_idx : int
            Start index of current trajectory

        max_size : int
            Maximum buffe size

    """

    def __init__(self, obs_dim, act_dim, batch_size, lstm_hidden_size, gamma=0.99, lam=0.95):
        self.obs_buf = torch.zeros(batch_size, obs_dim, dtype=torch.float32, device=device)
        self.act_buf = torch.zeros(batch_size, act_dim, dtype=torch.float32, device=device)
        self.adv_buf = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.ret_buf = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.val_buf = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.logp_buf = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.lstm_h_in_buf = torch.zeros(batch_size, lstm_hidden_size, dtype=torch.float32, device=device)
        self.lstm_c_in_buf = torch.zeros(batch_size, lstm_hidden_size, dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, batch_size

    def store(self, obs, act, rew, val, logp, lstm_h_in, lstm_c_in, done):
        """ Append one timestep of agent-environment interaction to the buffer at pointer position

            Parameters
            ----------
            obs : Tensor
                Observation at timestep

            act : Tensor
                Action at timestep

            rew : float
                Reward at timestep

            val : float
                State value at timestep

            logp : Tensor
                Log probability at timestep

            lstm_h_in :
                LSTM hidden state at timestep

            lstm_c_in :
                LSTM cell state at timestep

            done : bool
                Done flag at timestep
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.lstm_h_in_buf[self.ptr] = lstm_h_in
        self.lstm_c_in_buf[self.ptr] = lstm_c_in
        self.done_buf[self.ptr] = done
        self.ptr += 1

    def finish_path(self, last_val=0):
        """ Call this at the end of a trajectory, or when one gets cut off by an epoch ending. This looks back in the
            buffer to where the trajectory started, and uses rewards and value estimates from the whole trajectory to
            compute advantage estimates with GAE-Lambda, as well as compute the rewards-to-go for each state, to use as
            the targets for the value function.
            The "last_val" argument should be 0 if the trajectory ended because the agent reached a terminal
            state (died), and otherwise should be V(s_T), the value function estimated for the last state.
            This allows us to bootstrap the reward-to-go calculation to account for timesteps beyond the arbitrary
            episode horizon (or epoch cutoff).

            Parameters
            ----------
            last_val : float
                Reward and state value for terminal state

        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = torch.cat((self.rew_buf[path_slice], last_val))
        vals = torch.cat((self.val_buf[path_slice], last_val))

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = torch.FloatTensor(
            (discount_cumsum(deltas.cpu().numpy(), self.gamma * self.lam)).copy()).to(device)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = torch.FloatTensor(
            (discount_cumsum(rews.cpu().numpy(), self.gamma)[:-1]).copy()).to(device)

        self.path_start_idx = self.ptr

    def get(self):
        """ Gets all of the data from the buffer and resets pointer.

            Returns
            -------
            return_data : dictionary
                Dictionary with content of PPO buffer. Keys are: obs, act, ret, adv, logp, lstm_h_in, lstm_c_in, done with
                the corresponding values from buffer
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        # the next line implements the advantage normalization trick
        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std() + 1e-5)

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf,
                    lstm_h_in=self.lstm_h_in_buf, lstm_c_in=self.lstm_c_in_buf, done=self.done_buf)
        return_data = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in data.items()}

        return return_data


class MyDataset(Dataset):
    """ Extends pytorch's Dataset class to create dataset objects from the PPO buffer data

        Attributes
        ----------
        data: TensorDataset
            Data from PPO buffer

        window: int
            Length of sequence, which is used for stochastic gradient descent (SGD)
    """

    def __init__(self, data, window):
        self.data = data
        self.window = window

    def __getitem__(self, index):
        x = self.data[index:index + self.window]
        return x

    def __len__(self):
        return len(self.data) - self.window


class PPO:
    """ Implements Proximal Policy Optimization (PPO) for training of the decision maker

        Attributes
        ----------
        _params_rl: LearningParameters
            Hyperparameters for training the actor-critic model

        _params_model: ModelParameters
            Hyperparameters which describe the ANN-architecture of the decision maker

        buffer: PPOBuffer
            Buffer object to store the trajectories of the glider during interaction with the environment

        model: DecisionMakerActorCritic
            Actor-Critic model of the decision maker

        pi_optimizer: Adam
            Policy optimizer

        vf_optimizer: Adam
            Value function optimizer

        vertex_tracker: object
            Control algorithm for vertex tracking

        updraft_exploiter: UpdraftExploiterActorCritic
            Updraft exploiter model

        env: GliderEnv3D
            Simulation of glider environment
        """

    def __init__(self, vertex_tracker, updraft_exploiter, environment):
        # instantiate parameters
        self._params_rl = params_decision_maker.LearningParameters()
        self._params_model = params_decision_maker.ModelParameters()

        # instantiate buffer for rollout
        self.buffer = PPOBuffer(self._params_model.DIM_IN, self._params_model.DIM_OUT, self._params_rl.BATCHSIZE,
                                self._params_model.DIM_LSTM, gamma=self._params_rl.GAMMA, lam=self._params_rl.LAMBDA)

        # instantiate actor-critic model
        self.model = DecisionMakerActorCritic().to(device)

        """ Put in filepath to a valid decision maker .pt-file and uncomment to use a already trained model"""
        # self.model.actor.load_state_dict(torch.load("decision_maker_actor_final_03-November-2020_15-17.pt"))
        # self.model.critic.load_state_dict(torch.load("decision_maker_critic_final_03-November-2020_15-17.pt"))

        # setup optimizers
        self.pi_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=self._params_rl.LEARNING_RATE_PI)
        self.vf_optimizer = torch.optim.Adam(self.model.critic.parameters(), lr=self._params_rl.LEARNING_RATE_VF)

        # instantiate vertex tracker and updraft exploiter
        self.vertex_tracker = vertex_tracker
        self.updraft_exploiter = updraft_exploiter
        self.env = environment

    def select_action(self, state, lstm_hidden, validation_mask=False):
        """ Select action (control command) either from updraft exploiter or vertex tracker

        Parameters
        ----------
        state : Tensor
            Observation for decision maker (time, altitude and distance to finish)

        lstm_hidden : Tensor
             Previous state of hidden LSTM layer

        validation_mask : bool
            Deactivates random action sampling for validation

        Returns
        -------
        action_env : Tensor
            Action which is used for interaction with the environment

        action_agent : float
            Action of the decision maker

        action_agent_logprob :
            Logarithmic probability of decision maker action

        state_value :
            Computed state value from the critic network

        lstm_hidden :
            Hidden state of the decision maker LSTM
        """
        # evaluate decision maker
        action_agent, action_agent_logprob, lstm_hidden = self.model.act(state, lstm_hidden,
                                                                         validation_mask=validation_mask)

        # induce bias: height=0 leads to bias->1, initial_height (400 m) leads to bias->0
        inductive_bias = .5 * (1 - torch.tanh(state[1]))
        p_exploit = np.clip(action_agent.item() + inductive_bias.item(), 0, 1)

        # evaluate vertex tracker
        action_vertex_tracker = self.vertex_tracker.select_action()

        # evaluate updraft exploiter
        normalized_updraft_positions = self.env.get_updraft_positions()
        observation = torch.FloatTensor(normalized_updraft_positions).view(1, -1, 2).to(device)
        action_updraft_exploiter = self.updraft_exploiter.act(observation,
                                                              memory=None,
                                                              validation_mask=True).cpu().data.numpy().flatten()

        # switch between the sub-agents' actions
        action_env = action_updraft_exploiter if (p_exploit > .5) else action_vertex_tracker

        # evaluate critic
        state_value = self.model.critic(state)

        return action_env, action_agent, action_agent_logprob, state_value, lstm_hidden

    def update(self):
        """ Calculate policy update for actor and critic network

        """

        # get sampled data
        data = self.buffer.get()
        obs, act, ret, adv, logp, lstm_h_in, lstm_c_in, done \
            = data['obs'], data['act'], data['ret'], data['adv'], data['logp'], \
              data['lstm_h_in'], data['lstm_c_in'], data['done']

        # put batch to sliding-window data_loader
        data_set = MyDataset(TensorDataset(obs, act, ret, adv, logp, lstm_h_in, lstm_c_in, done),
                             self._params_rl.SEQ_LEN)

        # Optimize policy for K epochs:
        for _ in range(self._params_rl.K_EPOCH):
            for start_index in range(0, (data_set.__len__()) + 1,
                                     int(self._params_rl.SEQ_LEN - self._params_rl.OVERLAP)):
                # get sampled sequence/mini-batch
                obs_seq, act_seq, ret_seq, adv_seq, logp_seq, lstm_h_in_seq, lstm_c_in_seq, done_seq = \
                    data_set.__getitem__(start_index)

                if any(done_seq == 1):
                    #  never evaluate sequences that cross episode boundaries
                    done_index = done_seq.nonzero()[0].item()  # index of first done flag in sequence
                    if done_index > (self._params_rl.SEQ_LEN_MIN - 1):
                        obs_seq = obs_seq[slice(0, done_index + 1)]
                        act_seq = act_seq[slice(0, done_index + 1)]
                        ret_seq = ret_seq[slice(0, done_index + 1)]
                        adv_seq = adv_seq[slice(0, done_index + 1)]
                        logp_seq = logp_seq[slice(0, done_index + 1)]
                        lstm_h_in_seq = lstm_h_in_seq[slice(0, done_index + 1)]
                        lstm_c_in_seq = lstm_c_in_seq[slice(0, done_index + 1)]
                    else:
                        continue

                # "burn in" lstm hidden state (cf. "R2D2")
                with torch.no_grad():
                    _, (lstm_h_burned_in, lstm_c_burned_in) = \
                        self.model.actor(obs_seq[0:self._params_rl.N_BURN_IN, :], (lstm_h_in_seq[0], lstm_c_in_seq[0]))

                # evaluate policy for remainder sampled sequence of states and actions
                logp_eval = self.model.evaluate_actor(obs_seq[self._params_rl.N_BURN_IN:, :],
                                                      act_seq[self._params_rl.N_BURN_IN:],
                                                      (lstm_h_burned_in, lstm_c_burned_in))

                # ppo ratio
                ratios = torch.exp(logp_eval - logp_seq[self._params_rl.N_BURN_IN:])

                # surrogate loss (PPO)
                surr1 = ratios * adv_seq[self._params_rl.N_BURN_IN:]
                surr2 = torch.clamp(ratios, 1 - self._params_rl.EPS_CLIP, 1 + self._params_rl.EPS_CLIP) \
                        * adv_seq[self._params_rl.N_BURN_IN:]
                loss_pi = -torch.min(surr1, surr2).mean()

                # policy gradient step
                self.pi_optimizer.zero_grad()
                loss_pi.backward()
                nn.utils.clip_grad_norm_(self.model.actor.parameters(), 1.0)
                self.pi_optimizer.step()

                # value function gradient step
                loss_vf = ((self.model.critic(obs_seq) - ret_seq) ** 2).mean()
                self.vf_optimizer.zero_grad()
                loss_vf.backward()
                nn.utils.clip_grad_norm_(self.model.critic.parameters(), 1.0)
                self.vf_optimizer.step()


def discount_cumsum(x, discount):
    """ Function taken from rllab for computing discounted cumulative sums of vectors.
    https://github.com/rll/rllab

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
