" Script to export matplotlib plots from training to tikz"

from decision_maker.ppo_decision_maker import PPO
from subtasks.updraft_exploiter import model_updraft_exploiter
from test_trained_policies import plot_decision_maker
from subtasks.vertex_tracker.waypoint_controller import ControllerWrapper
from decision_maker import params_decision_maker
import torch
import gym

device = torch.device('cpu')
env = gym.make('glider3D-v0', agent='decision_maker')

# set seed to fix updraft distribution and trajectory
#env.seed(42)
#np.random.seed(42)

waypoint_controller = ControllerWrapper(env)
updraft_exploiter = model_updraft_exploiter.UpdraftExploiterActorCritic().to(device)
updraft_exploiter.load_state_dict(torch.load("updraft_exploiter_actor_critic_final_17-October-2021_20-21.pt", map_location=torch.device('cpu')))
ppo = PPO(waypoint_controller, updraft_exploiter, env)
ppo.model.actor.load_state_dict(torch.load("decision_maker_actor_final_30-October-2021_11-02.pt", map_location=torch.device('cpu')))
_params_agent = params_decision_maker.AgentParameters()

for plot_number in range(0, 5):
    print("Running iteration number {}!".format(plot_number))
    plot_decision_maker.main(env, ppo, plot_number, _params_agent, validation_mask=True)