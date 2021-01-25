# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:00:28 2020

@author: Gregor

This takes a pytorch model to generate a .mat-File with all the tensors from the
state dictionary.
"""

# take state dict as input
# import torch
# import numpy as np

from scipy.io import savemat


class mFileGenerator():
    """
    Superclass for the MATLAB-File generator.
    """

    def __init__(self):
        pass

    def printStateDict(self, pytorch_model):

        for e in pytorch_model.state_dict():
            print(e)


class updraftGenerator(mFileGenerator):

    def __init__(self):
        super().__init__()

    def generateFile(self, updraft_exploiter, filepath):

        # extract weight vectors from model
        lstm_w_ih = updraft_exploiter.lstm.weight_ih_l0
        lstm_w_hh = updraft_exploiter.lstm.weight_hh_l0
        lstm_b_ih = updraft_exploiter.lstm.bias_ih_l0
        lstm_b_hh = updraft_exploiter.lstm.bias_hh_l0

        # list which stores weights and biases
        weight_list = []

        name_list = ["w_ii", "w_if", "w_ig", "w_io",
                     "w_hi", "w_hf", "w_hg", "w_ho",
                     "b_ii", "b_if", "b_ig", "b_io",
                     "b_hi", "b_hf", "b_hg", "b_ho"]

        """
        weight vectors are split the following way
        w_ii, w_if, w_ig, w_io = lstm_w_ih.chunk(4,0)
        w_hi, w_hf, w_hg, w_ho = lstm_w_hh.chunk(4,0)
        b_ii, b_if, b_ig, b_io = lstm_b_ih.chunk(4,0)
        b_hi, b_hf, b_hg, b_ho = lstm_b_hh.chunk(4,0)
        """
        # split weights and biases and copy them to a list
        weight_list.extend(list(lstm_w_ih.chunk(4, 0)))
        weight_list.extend(list(lstm_w_hh.chunk(4, 0)))
        weight_list.extend(list(lstm_b_ih.chunk(4, 0)))
        weight_list.extend(list(lstm_b_hh.chunk(4, 0)))

        # dictionary to save weights
        weight_dict = {}

        # convert tensors to numpy array and save weights to dictionary
        for name, weight in zip(name_list, weight_list):
            weight = weight.detach().numpy()
            weight_dict[name] = weight
            # add to dict

        # add hidden and actor/critic layer to dict
        _ = updraft_exploiter
        weight_dict['w_hidden'] = _.hidden_layer.weight.detach().numpy()
        weight_dict['b_hidden'] = _.hidden_layer.bias.detach().numpy()
        weight_dict['w_actor'] = _.out_actor.weight.detach().numpy()
        weight_dict['b_actor'] = _.out_actor.bias.detach().numpy()
        weight_dict['w_critic'] = _.out_critic.weight.detach().numpy()
        weight_dict['b_critic'] = _.out_critic.bias.detach().numpy()

        # save weights to matlab file
        savemat(filepath, weight_dict)


class decisionGenerator(mFileGenerator):
    """
    Generates .mat-File of the decision_maker
    """

    def __init__(self):
        super().__init__()

    def generateFile(self, decision_maker, filepath):

        # extract weight vectors from model
        lstm_w_ih = decision_maker.lstm.weight_ih_l0
        lstm_w_hh = decision_maker.lstm.weight_hh_l0
        lstm_b_ih = decision_maker.lstm.bias_ih_l0
        lstm_b_hh = decision_maker.lstm.bias_hh_l0

        # list which stores weights and biases
        weight_list = []

        name_list = ["w_ii", "w_if", "w_ig", "w_io",
                     "w_hi", "w_hf", "w_hg", "w_ho",
                     "b_ii", "b_if", "b_ig", "b_io",
                     "b_hi", "b_hf", "b_hg", "b_ho"]

        """
        weight vectors are split the following way
        w_ii, w_if, w_ig, w_io = lstm_w_ih.chunk(4,0)
        w_hi, w_hf, w_hg, w_ho = lstm_w_hh.chunk(4,0)
        b_ii, b_if, b_ig, b_io = lstm_b_ih.chunk(4,0)
        b_hi, b_hf, b_hg, b_ho = lstm_b_hh.chunk(4,0)
        """
        # split weights and biases and copy them to a list
        weight_list.extend(list(lstm_w_ih.chunk(4, 0)))
        weight_list.extend(list(lstm_w_hh.chunk(4, 0)))
        weight_list.extend(list(lstm_b_ih.chunk(4, 0)))
        weight_list.extend(list(lstm_b_hh.chunk(4, 0)))

        # dictionary to save weights
        weight_dict = {}

        # convert tensors to numpy array and save weights to dictionary
        for name, weight in zip(name_list, weight_list):
            weight = weight.detach().numpy()
            weight_dict[name] = weight
            # add to dict

        # add hidden and actor/critic layer to dict
        # add input and output layer to dict
        _ = decision_maker
        weight_dict['w_input'] = _.input_layer.weight.detach().numpy()
        weight_dict['b_input'] = _.input_layer.bias.detach().numpy()
        weight_dict['w_output'] = _.output_layer.weight.detach().numpy()
        weight_dict['b_output'] = _.output_layer.bias.detach().numpy()

        # save weights to matlab file
        savemat(filepath, weight_dict)
