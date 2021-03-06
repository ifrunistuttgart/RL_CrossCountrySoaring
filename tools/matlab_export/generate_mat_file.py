"""
This scripts is used to export the weights and biases from the .pt-Files of updraft exploiter and decision maker
to .mat-Files
"""

from policy_training import train_decision_maker
from hierarchical_policy.updraft_exploiter import model_updraft_exploiter
from hierarchical_policy.vertex_tracker.waypoint_controller import ControllerWrapper
from mat_file_generator import MatFileExporter
from glider.envs.glider_env_3D import GliderEnv3D
import torch

device = torch.device('cpu')

# Enter filepath for updraft_exploiter and decision_maker
updraft_exploiter_file = "policies/updraft_exploiter_actor_critic_final_17-October-2021_20-21.pt"
decision_maker_file = "policies/decision_maker_actor_final_30-October-2021_11-02.pt"

# Choose export targets
export_decision_maker = True
export_updraft_exploiter = True

# Load updraft exploiter
updraft_exploiter = model_updraft_exploiter.UpdraftExploiterActorCritic().to(device)
updraft_exploiter.load_state_dict(torch.load(updraft_exploiter_file, map_location=torch.device('cpu')))

# Create ppo-object and load decision maker
# env = gym.make('glider3D-v0', agent='decision_maker')
env = GliderEnv3D(agent='decision_maker')

waypoint_controller = ControllerWrapper(env)
ppo = train_decision_maker.PPO(waypoint_controller, updraft_exploiter, env)
ppo.model.actor.to(device)
ppo.model.actor.load_state_dict(torch.load(decision_maker_file, map_location=torch.device('cpu')))

# create filespaths for .mat-files
path_decision_maker = "./mat_files/{}".format(decision_maker_file.replace('.pt', '.mat').replace('policies/', ''))
path_updraft_exploiter = "./mat_files/{}".format(updraft_exploiter_file.replace('.pt', '.mat').replace('policies/', ''))

# Export updraft exploiter
if export_updraft_exploiter:
    MatFileExporter.export_updraft_exploiter(ppo.updraft_exploiter, path_updraft_exploiter)
    print("Updraft exploiter successfully exported!")

# Export decision maker
if export_decision_maker:
    MatFileExporter.export_decision_maker(ppo.model.actor, path_decision_maker)
    print("Decision maker successfully exported!")

