import torch
import torch.nn as nn
from src.data.create_input_output_additive import PEOPLE, INPUT_DIM
import numpy as np
"""
Activation function used:
gamma(x; g, theta, alpha) = g * log(1+exp(alpha*(x-theta)))/alpha
where g is the gain, theta is threshold, alpha is the softening parameter. Only the gain is learned by the LSTM. 
"""

# See: https://stackoverflow.com/questions/47952930/how-can-i-use-lstm-in-pytorch-for-classification


class NetworkAdditive(nn.Module):
    """
    Create a network where each neuron is has its inner chemistry represented by an LSTM.
    All neurons are the same, i.e, they have the same inner chemistry mechanisms, therefore the same LSTM.
    All neurons are independent from one another.
    Each neuron receives a 1d time series. Neurons are grouped such that each neuron can represent a person within a time slot.
    At some random point, we send the neurons correspondent to the queried time slot a cue signal, indicating we want that neuron
    to respond to whether there was an input or not, leading to a one hot encoded output.
    We expect the neuron to have an output in 1 hot encoding for the person that had the slot in the queried time.
    """

    def __init__(self, hidden_size_LSTM=64, input_lstm_dim = 1):
        super(NetworkAdditive, self).__init__()

        self.hidden_size_LSTM = hidden_size_LSTM

        # create the LSTM
        self.lstm = nn.LSTM(input_size=input_lstm_dim, hidden_size=self.hidden_size_LSTM, num_layers=1)

        # create LSTM output
        self.linearLSTM = nn.Linear(self.hidden_size_LSTM, 1)  # LSTM outputs 1 parameter for the AF

        # create linear combination of input
        self.linear = nn.Linear(INPUT_DIM, input_lstm_dim)

        # # create neurons
        # self.neurons_input2hidden = nn.ModuleDict()
        # self.neurons_hidden2out = nn.ModuleDict()

        # create dictionary to save hidden states and g for all neurons
        self.neuron_variables = {}

        for neuron_number in range(len(PEOPLE)):
            # self.neurons_input2hidden["neuron{0}".format(neuron_number)] = nn.Linear(1, 10)
            # self.neurons_hidden2out["neuron{0}".format(neuron_number)] = nn.Linear(10, 1)
            self.neuron_variables["neuron{0}".format(neuron_number)] = {}

    def forward(self, input):
        """
        :param input:
        :return:
        """
        n_timepoints, n_batches = input.size(0), input.size(1)
        output = torch.zeros((n_batches, len(PEOPLE)))

        # Initialize hidden states of the LSTM for each neuron
        for neuron_number in range(len(PEOPLE)):
            self.neuron_variables["neuron{0}".format(neuron_number)]["hiddenLSTM"] = \
                (torch.zeros(1, n_batches, self.hidden_size_LSTM, dtype=torch.float32),
                 torch.zeros(1, n_batches, self.hidden_size_LSTM, dtype=torch.float32))
            # Initialize g for all neurons, such that we can save it later
            self.neuron_variables["neuron{0}".format(neuron_number)]["g"] = [1]


        outLin = self.linear(input)

        for neuron_number in range(len(PEOPLE)):
            self.neuron_variables["neuron{0}".format(neuron_number)]["hiddenLSTM"] = self.lstm(outLin,
                                                self.neuron_variables["neuron{0}".format(neuron_number)]["hiddenLSTM"] )


            g_neuron = self.linearLSTM(self.neuron_variables["neuron{0}".format(neuron_number)]["hiddenLSTM"][0][-1])
            self.neuron_variables["neuron{0}".format(neuron_number)]["g"].append(g_neuron)
            out = torch.mul(g_neuron, torch.log(1 + torch.exp(torch.mul(1, outLin[-1,:,:] - 1))))

            output[:, neuron_number] = out.reshape(-1,)




        return output, self.neuron_variables