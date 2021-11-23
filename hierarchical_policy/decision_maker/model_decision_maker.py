import torch
import torch.nn as nn

from hierarchical_policy.decision_maker import params_decision_maker
from torch.distributions import Normal

# Choose device here
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class DecisionMakerActorCritic(nn.Module):
    """ Actor-critic model for the decision maker, which decides between updraft exploiting and vertex tracking

        Attributes
        ----------
        _params_model: ModelParameters
            Shape of the network layers

        _params_rl: LearningParameters
            Hyperparameters for training

        actor: LSTMActor
            Actor network which has a linear input layer, a LSTM hidden layer and a linear output layer

        critic: Critic
            Critic network with linear input, hidden and output layer
    """

    def __init__(self):
        super().__init__()

        # instantiate parameters
        self._params_model = params_decision_maker.ModelParameters()
        self._params_rl = params_decision_maker.LearningParameters()

        # setup ANN
        self.actor = LSTMActor(obs_dim=self._params_model.DIM_IN, act_dim=self._params_model.DIM_OUT,
                               hidden_size=self._params_model.DIM_HIDDEN, lstm_size=self._params_model.DIM_LSTM)
        self.critic = Critic(obs_dim=self._params_model.DIM_IN, hidden_size=self._params_model.DIM_HIDDEN)

    def act(self, state, lstm_hidden, validation_mask=False):
        """ Evaluates current actor model for one single state (observation). The observation contains time, altitude
            and distance to finish

        Parameters
        ----------
        state : Tensor
            Single observation for decision maker (time, altitude and distance to finish)

        lstm_hidden : Tensor
            Previous hidden LSTM state

        validation_mask : bool
            Deactivates random action sampling

        Returns
        -------
        action : Tensor
            Sampled probability for updraft exploitation or vertex tracking

        action_logprob : Tensor
            Logarithmic probability of action

        lstm_hidden : Tensor
            New hidden LSTM state
        """
        # evaluate current actor to sample action for rollout
        action_mean, lstm_hidden = self.actor.forward(state, lstm_hidden)
        dist = Normal(action_mean, self._params_rl.SIGMA * (not validation_mask))
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action, action_logprob, lstm_hidden

    def evaluate_actor(self, sampled_state, sampled_action, sampled_lstm_hidden):
        """ Evaluates actor network for sampled inputs during PPO update step

        Parameters
        ----------
        sampled_state : Tensor
            Sample of states from PPO Buffer

        sampled_action : Tensor
            Sampled actions from PPO buffer

        sampled_lstm_hidden : Tensor
            Sampled lstm states from PPO buffer

        Returns
        -------
        flattened_action_logprobs : Tensor
            Logprobs for all sampled actions as a flattened 1D array

        """
        # evaluate actor for sampled states
        action_mean, _ = self.actor.forward(sampled_state, sampled_lstm_hidden)
        dist = Normal(torch.flatten(action_mean, 1), self._params_rl.SIGMA)

        # get logprobs for distribution subject to current actor, evaluated for sampled actions
        action_logprobs = dist.log_prob(sampled_action)
        flattened_action_logprobs = action_logprobs.flatten()

        return flattened_action_logprobs

    def reset_lstm(self):
        """ Resets LSTM inital and cell state

        Returns
        -------
        h_0 : Tensor
            Reset initial state of LSTM

        c_0 : Tensor
            Reset cell state of LSTM
        """

        h_0 = torch.zeros(1, 1, self._params_model.DIM_LSTM, device=device)
        c_0 = torch.zeros(1, 1, self._params_model.DIM_LSTM, device=device)

        return h_0, c_0

class LSTMActor(nn.Module):
    """ This class implements the actor model for the decision maker. It has three layers. Input and output layers are
        simple linear layers. The hidden layer is a Long Short-Term Memory (LSTM) layer

        Attributes
        ----------
        input_layer : torch.nn.Linear
            Linear input layer

        lstm : torch.nn.LSTM
            Hidden LSTM layer

        output_layer : torch.nn.Linear
            Linear output layer

        _obs_dim : object
            Dimension of observation vector

        _act_dim : object
            Dimension of action space

        _hidden_size : object
            Dimension of LSTM input size.

        _lstm_size : object
            Dimension of LSTM layer
    """
    def __init__(self, obs_dim, act_dim, hidden_size, lstm_size):
        super().__init__()

        self.input_layer = nn.Linear(obs_dim, hidden_size)
        self.lstm = nn.LSTM(hidden_size, lstm_size)
        self.output_layer = nn.Linear(lstm_size, act_dim)

        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._hidden_size = hidden_size
        self._lstm_size = lstm_size

    def forward(self, observation, lstm_hidden):
        """ Computes forward pass through actor network of decision maker. Output is a probability for using the
            updraft exploiter sub-policy

        Parameters
        ----------
        observation : Tensor
            Observation for decision maker (time, altitude and distance to finish). Can be a sequence of states
            with variable length

        lstm_hidden :
            Previous state of hidden LSTM layer

        Returns
        -------
        action : float
            Probability for updraft exploitation
        """
        # evaluate input
        x = observation.reshape(-1, self._obs_dim).to(device)  # seq_len x  input_size
        x = torch.tanh(self.input_layer(x))

        # evaluate lstm
        x = x.reshape(-1, 1, self._hidden_size)  # seq_len x batch_size x  lstm_in_size
        x, lstm_hidden = self.lstm(x, (lstm_hidden[0].reshape(1, 1, self._hidden_size),
                                       lstm_hidden[1].reshape(1, 1, self._hidden_size)))

        # evaluate actor output layer
        x = x.reshape(-1, self._lstm_size)  # seq_len x lstm_out_size
        action = self.output_layer(x)

        # # map output to [0, 1]
        # p_exploit = 0.5 * (torch.tanh(z) + 1)

        return action, lstm_hidden


class Critic(nn.Module):
    """ This class implements the actor-critic model for the decision maker, which decides between updraft exploiting
        and vertex tracking

        Attributes
        ----------
        input_layer : torch.nn.Linear
            Input layer of critic network

        hidden_layer : torch.nn.Linear
            Hidden layer of critic network

        output_layer : torch.nn.Linear
            Output layer of critic network

        _obs_dim : int
            Dimension of input to critic network
    """

    def __init__(self, obs_dim, hidden_size):
        super().__init__()

        self.input_layer = nn.Linear(obs_dim, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

        self._obs_dim = obs_dim

    def forward(self, observation):
        """ Computes forward pass trough critic network, which puts out the value of the current
            state (observation)

        Parameters
        ----------
        observation : Tensor
            Observation for decision maker (time, altitude and distance to finish)

        Returns
        -------
        value : float
            State value of observation
        """
        # evaluate input
        x = observation.reshape(-1, self._obs_dim).to(device)  # batch_size x  input_size
        x = torch.tanh(self.input_layer(x))

        # evaluate hidden layer
        x = torch.tanh(self.hidden_layer(x))

        # evaluate critic output layer
        value = self.output_layer(x)

        return value