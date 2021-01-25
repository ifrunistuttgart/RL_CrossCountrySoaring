import train_decision_maker
from subtasks.updraft_exploiter import model_updraft_exploiter
from subtasks.vertex_tracker.waypoint_controller import Controller_Wrapper
import mat_file_generator
import torch
import gym

device = torch.device('cpu')

# Enter filepaths of updraft_exploiter and decision_maker
updraft_exploiter_file = "updraft_exploiter_actor_critic_final_17-December-2020_11-06.pt"
decision_maker_file = "decision_maker_actor_final_18-January-2021_00-03.pt"

# Choose export targets
export_decision_maker = True
export_updraft_exploiter = True

# create glider env object
env = gym.make('glider3D-v0', agent='decision_maker')

# instantiate vertex tracker
waypoint_controller = Controller_Wrapper(env)

# Load updraft exploiter
updraft_exploiter = model_updraft_exploiter.ActorCritic().to(device)
updraft_exploiter.load_state_dict(torch.load(updraft_exploiter_file, map_location=torch.device('cpu')))

# Create ppo-object and load decision maker
ppo = train_decision_maker.PPO(waypoint_controller, updraft_exploiter, env)
ppo.model.actor.to(device)
ppo.model.actor.load_state_dict(torch.load(decision_maker_file, map_location=torch.device('cpu')))

# create filespaths for .mat-files
exportpath_decision_maker = "./mat_files/{}".format(decision_maker_file.replace('.pt', '.mat'))
exportpath_updraft_exploiter = "./mat_files/{}".format(updraft_exploiter_file.replace('.pt', '.mat'))

# Export updraft exploiter
if export_updraft_exploiter:
    updraft_generator = mat_file_generator.updraftGenerator()
    updraft_generator.generateFile(ppo.updraft_exploiter, exportpath_updraft_exploiter)
    print("Updraft exploiter successfully exported!")

# Export decision maker
if export_decision_maker:
    decision_generator = mat_file_generator.decisionGenerator()
    decision_generator.generateFile(ppo.model.actor, exportpath_decision_maker)
    print("Decision maker successfully exported!")

