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

    def __init__(self, hidden_size_LSTM=64, input_lstm_dim = 1, number_neurons = 20):
        super(NetworkAdditive, self).__init__()

        self.hidden_size_LSTM = hidden_size_LSTM
        self.n_neurons = number_neurons

        # create linear layer that maps input (Npeople+Ntimesteps) to (n_neurons)
        self.linear2neurons = nn.Linear(INPUT_DIM, self.n_neurons)

        # create the LSTM
        self.lstm = nn.LSTMCell(input_size=input_lstm_dim, hidden_size=self.hidden_size_LSTM)

        # create LSTM output
        self.linearLSTM = nn.Linear(self.hidden_size_LSTM, 1)  # LSTM outputs 1 parameter for the AF

        # create linear layer that maps output of neurons to actual expected output
        self.neurons2output = nn.Linear(self.n_neurons, len(PEOPLE))


        # # create neurons
        # self.neurons_input2hidden = nn.ModuleDict()
        # self.neurons_hidden2out = nn.ModuleDict()

        # create dictionary to save hidden states and g for all neurons
        self.neuron_variables = {}

        for neuron_number in range(self.n_neurons):
            # self.neurons_input2hidden["neuron{0}".format(neuron_number)] = nn.Linear(INPUT_DIM, input_lstm_dim)
            # self.neurons_hidden2out["neuron{0}".format(neuron_number)] = nn.Linear(10, input_lstm_dim)
            self.neuron_variables["neuron{0}".format(neuron_number)] = {}

    def forward(self, input):
        """
        :param input:
        :return:
        """
        n_timepoints, n_batches = input.size(0), input.size(1)
        output_neurons = torch.zeros((n_timepoints, n_batches, self.n_neurons))
        output_final = torch.zeros((n_timepoints, n_batches, len(PEOPLE)))

        # Initialize hidden states of the LSTM for each neuron
        for neuron_number in range(self.n_neurons):
            self.neuron_variables["neuron{0}".format(neuron_number)]["hiddenLSTM"] = \
                (torch.zeros(n_batches, self.hidden_size_LSTM, dtype=torch.float32),
                 torch.zeros(n_batches, self.hidden_size_LSTM, dtype=torch.float32))
            # Initialize g for all neurons, such that we can save it later
            self.neuron_variables["neuron{0}".format(neuron_number)]["g"] = [1]

        for i in range(n_timepoints):
            input2neuron = self.linear2neurons(input[i,:,:])
            for neuron_number in range(self.n_neurons):
                # outLin = self.neurons_input2hidden["neuron{0}".format(neuron_number)](input[i,:,:])
                neuron_input = input2neuron[:,neuron_number].reshape(-1,1)
                self.neuron_variables["neuron{0}".format(neuron_number)]["hiddenLSTM"] = self.lstm(neuron_input,
                                                    self.neuron_variables["neuron{0}".format(neuron_number)]["hiddenLSTM"] )
                g_t = self.linearLSTM(self.neuron_variables["neuron{0}".format(neuron_number)]["hiddenLSTM"][0])
                self.neuron_variables["neuron{0}".format(neuron_number)]["g"].append(g_t)
                g_previous = self.neuron_variables["neuron{0}".format(neuron_number)]["g"][i-1]
                out = torch.mul(g_previous, torch.log(1 + torch.exp(torch.mul(1, neuron_input - 1))))

                output_neurons[i,:, neuron_number] = out.reshape(-1,)

            output_final = self.neurons2output(output_neurons)

        return output_final, self.neuron_variables