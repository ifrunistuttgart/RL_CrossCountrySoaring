import datetime
import time
import os
import shutil
import gym
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset

import evaluate_glider
from parameters import params_environment, params_triangle_soaring
from subtasks.vertex_tracker.waypoint_controller import Waypoint_Controller
from subtasks.vertex_tracker import params_vertex_tracker

device = torch.device("cuda:0")

class Controller_Wrapper:
    def __init__(self, environment):
        self.waypoint_controller = Waypoint_Controller()
        self.env = environment

    def select_action(self, obserervation, memory=None, validation_mask=False):
        phi_cmd, alpha_cmd = self.waypoint_controller.get_control(self.env.state, self.env.activeVertex)
        action = np.array([phi_cmd, alpha_cmd])
        return action

def main():
    env = gym.make('glider3D-v0', agent='vertex_tracker')
    controller = Controller_Wrapper(env)
    evaluate_glider.main(env, controller, 0, params_vertex_tracker.params_agent(), validation_mask=False)
    env.close()

if __name__ == '__main__':
    main()