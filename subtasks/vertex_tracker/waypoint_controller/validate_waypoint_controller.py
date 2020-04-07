import gym
import torch
import numpy as np

import evaluate_glider

from subtasks.vertex_tracker.waypoint_controller.waypoint_controller import Waypoint_Controller, Controller_Wrapper
from subtasks.vertex_tracker import params_vertex_tracker

device = torch.device("cuda:0")


def main():
    env = gym.make('glider3D-v0', agent='vertex_tracker')
    controller = Controller_Wrapper(env)
    evaluate_glider.main(env, controller, 0, params_vertex_tracker.params_agent(), validation_mask=False)
    env.close()

if __name__ == '__main__':
    main()