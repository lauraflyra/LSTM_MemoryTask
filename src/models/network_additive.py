import torch
import torch.nn as nn
from src.data.create_input_output_additive import PEOPLE, INPUT_DIM
import numpy as np
"""
Activation function used:
gamma(x; g, theta, alpha) = g * log(1+exp(alpha*(x-theta)))/alpha
where g is the gain, theta is threshold, alpha is the softening parameter. Only the gain is learned by the LSTM.
alpha and theta are fixed to 1 here.
"""



class NetworkAdditive(nn.Module):
    """
    Create a network where each neuron is has its inner chemistry represented by an LSTM.
    All neurons are the same, i.e, they have the same inner chemistry mechanisms, therefore the same LSTM.
    All neurons are independent from one another.
    Here we have number of neurons as a parameter.
    The idea is that the input is one hot encoding of people + one hot encoding of time. A linear layer takes this input
    and gives it to each neuron, inside each neuron, an LSTM computed the gain, and the output of the neuron is put
    through the AF that also takes the gain as parameter. A linear mapping from neurons to output is made in the last layer.
    Output is one hot encoding of people, given the input for that time step.

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


        # create dictionary to save hidden states and g for all neurons
        self.neuron_variables = {}

        for neuron_number in range(self.n_neurons):
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