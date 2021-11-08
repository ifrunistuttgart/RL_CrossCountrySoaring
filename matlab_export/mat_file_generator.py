"""
Exports trained pytorch model of updraft exploiter and decision maker to MATLAB. Generates a .mat-File with all
the tensors from the state dictionary.
"""

from scipy.io import savemat


class MatFileExporter:
    """ Class which contains functions for exporting model parameters of updraft exploiter and decision maker
        to .mat-File.
    """

    @staticmethod
    def get_lstm_parameters(model):
        """ Extracts weights and biases from LSTM ans stores them to a dictionary

        Parameters
        ----------
        model : decision maker or updraft exploiter model

        Returns
        -------
        weight_dict : dictionary
            dictionary, which contains weights and biases of LSTM. Keys are:
                    ["w_ii", "w_if", "w_ig", "w_io",
                     "w_hi", "w_hf", "w_hg", "w_ho",
                     "b_ii", "b_if", "b_ig", "b_io",
                     "b_hi", "b_hf", "b_hg", "b_ho"]
        """

        # extract LSTM weights and biases
        lstm_w_ih = model.lstm.weight_ih_l0
        lstm_w_hh = model.lstm.weight_hh_l0
        lstm_b_ih = model.lstm.bias_ih_l0
        lstm_b_hh = model.lstm.bias_hh_l0

        # declare list which stores weights and biases
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

        return weight_dict

    @staticmethod
    def export_updraft_exploiter(updraft_exploiter, filepath):
        """ This function takes the weights and biases of the updraft exploiter model and stores
            them into a .mat-File.

        Parameters
        ----------
        updraft_exploiter : model_updraft_exploiter.DecisionMakerActorCritic
            updraft_exploiter object which contains parameters of trained model

        filepath : str
            filepath for saving the generated .mat-File

        """

        # get LSTM parameters
        weight_dict = MatFileExporter.get_lstm_parameters(updraft_exploiter)

        # add hidden and actor/critic layer to dict
        _ = updraft_exploiter
        weight_dict['w_hidden'] = _.hidden_layer.weight.detach().numpy()
        weight_dict['b_hidden'] = _.hidden_layer.bias.detach().numpy()
        weight_dict['w_actor'] = _.out_actor.weight.detach().numpy()
        weight_dict['b_actor'] = _.out_actor.bias.detach().numpy()
        weight_dict['w_critic'] = _.out_critic.weight.detach().numpy()
        weight_dict['b_critic'] = _.out_critic.bias.detach().numpy()

        # export dictionary to matlab file
        savemat(filepath, weight_dict)

    @staticmethod
    def export_decision_maker(decision_maker, filepath):
        """ This function takes the weights and biases of the decision maker model and stores
            them into a .mat-File.

        Parameters
        ----------
        decision_maker : model_updraft_exploiter.DecisionMakerActorCritic
            updraft_exploiter object which contains parameters of trained model

        filepath : str
            filepath for saving the generated .mat-File
        """

        # get LSTM parameters
        weight_dict = MatFileExporter.get_lstm_parameters(decision_maker)

        # add input and output layer to dictionary
        _ = decision_maker
        weight_dict['w_input'] = _.input_layer.weight.detach().numpy()
        weight_dict['b_input'] = _.input_layer.bias.detach().numpy()
        weight_dict['w_output'] = _.output_layer.weight.detach().numpy()
        weight_dict['b_output'] = _.output_layer.bias.detach().numpy()

        # export dictionary to matlab file
        savemat(filepath, weight_dict)
