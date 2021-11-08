" Script to export matplotlib plots from training to tikz"

import train_decision_maker
import numpy as np
import evaluate_decision_maker
from subtasks.updraft_exploiter import model_updraft_exploiter
from plotting import plot_decision_maker
from subtasks.vertex_tracker.waypoint_controller import ControllerWrapper
from parameters import params_decision_maker
import torch
import gym
import glider

device = torch.device('cpu')

env = gym.make('glider3D-v0', agent='decision_maker')
env.seed(42)
np.random.seed(42)
waypoint_controller = ControllerWrapper(env)
updraft_exploiter = model_updraft_exploiter.UpdraftExploiterActorCritic().to(device)
updraft_exploiter.load_state_dict(torch.load("updraft_exploiter_actor_critic_final_17-October-2021_20-21.pt", map_location=torch.device('cpu')))
ppo = train_decision_maker.PPO(waypoint_controller, updraft_exploiter, env)
_params_agent = params_decision_maker.AgentParameters()

# plot_decision_maker.main(env, ppo, 4000, _params_agent, validation_mask=True)

plot_decision_maker.main(env, ppo, 4000, _params_agent, validation_mask=True)