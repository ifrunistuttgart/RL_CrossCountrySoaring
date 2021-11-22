import gym
import torch

from policy_training.decision_maker import evaluate_decision_maker

from policy_training.subtasks.vertex_tracker.waypoint_controller import ControllerWrapper
from policy_training.subtasks.vertex_tracker import params_vertex_tracker

device = torch.device("cuda:0")


def main():
    env = gym.make('glider3D-v0', agent='vertex_tracker')
    controller = ControllerWrapper(env)
    evaluate_decision_maker.main(env, controller, 0, params_vertex_tracker.AgentParameters(), validation_mask=False)
    env.close()


if __name__ == '__main__':
    main()