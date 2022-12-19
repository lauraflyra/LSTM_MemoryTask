import torch
import torch.nn as nn
from src.data.create_input_output_factorial import PEOPLE, INPUT_SIZE
import numpy as np
"""
Activation function used:
gamma(x; g, theta, alpha) = g * log(1+exp(alpha*(x-theta)))/alpha
where g is the gain, theta is threshold, alpha is the softening parameter. Only the gain is learned by the LSTM. 
"""

# See: https://stackoverflow.com/questions/47952930/how-can-i-use-lstm-in-pytorch-for-classification
# TODO: question on output:
# QUESTION: So far we have time series output, should we change to only one output as the one-hot encoded person?
# Would that make sense regarding our expectations for the behavior of the AF?


class Network(nn.Module):
    """
    Create a network where each neuron is has its inner chemistry represented by an LSTM.
    All neurons are the same, i.e, they have the same inner chemistry mechanisms, therefore the same LSTM.
    All neurons are independent from one another.
    Each neuron receives a 1d time series. Neurons are grouped such that each neuron can represent a person within a time slot.
    At some random point, we send the neurons correspondent to the queried time slot a cue signal, indicating we want that neuron
    to respond to whether there was an input or not, leading to a one hot encoded output.
    We expect the neuron to have an output in 1 hot encoding for the person that had the slot in the queried time.
    """

    def __init__(self, hidden_size_LSTM=64):
        """
        """
        super(Network, self).__init__()

        self.hidden_size_LSTM = hidden_size_LSTM

        # create the LSTM
        self.lstm = nn.LSTMCell(input_size=2, hidden_size=self.hidden_size_LSTM)

        # create LSTM output
        self.linearLSTM = nn.Linear(self.hidden_size_LSTM, 3)       # LSTM outputs 1 parameter for the AF

        # create neurons
        self.neurons = {}
        for neuron_number in range(INPUT_SIZE):
            self.neurons["neuron{0}".format(neuron_number)] = {}
            self.neurons["neuron{0}".format(neuron_number)]["input2hidden"] = nn.Linear(1,10)
            self.neurons["neuron{0}".format(neuron_number)]["hidden2out"] = nn.Linear(10, 1)

    def forward(self, input):
        """
        :param input:
        :return:
        """
        n_timepoints, n_batches = input.size(0), input.size(1)
        output = torch.zeros((n_timepoints, n_batches, INPUT_SIZE))

        # Initialize hidden states of the LSTM for each neuron
        for neuron_number in range(INPUT_SIZE):
            self.neurons["neuron{0}".format(neuron_number)]["hiddenLSTM"] = \
                (torch.zeros(n_batches, self.hidden_size_LSTM, dtype=torch.float32),
                 torch.zeros(n_batches, self.hidden_size_LSTM, dtype=torch.float32))
            # Initialize g for all neurons, such that we can save it later
            self.neurons["neuron{0}".format(neuron_number)]["g"] = [1]
            self.neurons["neuron{0}".format(neuron_number)]["alpha"] = [1]
            self.neurons["neuron{0}".format(neuron_number)]["theta"] = [1]
            self.neurons["neuron{0}".format(neuron_number)]["output"] = []

        for i in range(n_timepoints):
            for neuron_number in range(INPUT_SIZE):
                outHid = self.neurons["neuron{0}".format(neuron_number)]["input2hidden"](torch.reshape(input[i,:,neuron_number],(-1,1)))
                outLin = torch.reshape(self.neurons["neuron{0}".format(neuron_number)]["hidden2out"](outHid),(-1,))

                # Try first to train with LSTM getting input combined with output
                if i > 0:
                    # o = torch.tensor(output, requires_grad=True)
                    combined = torch.cat((torch.reshape(input[i,:,neuron_number],(-1,1)),torch.reshape(output[i-1,:,neuron_number],(-1,1)).float()),1)

                    self.neurons["neuron{0}".format(neuron_number)]["hiddenLSTM"] = self.lstm(combined,
                                                        self.neurons["neuron{0}".format(neuron_number)]["hiddenLSTM"] )
                    g_t, alpha_t, theta_t = self.linearLSTM(self.neurons["neuron{0}".format(neuron_number)]["hiddenLSTM"][0]).T
                    self.neurons["neuron{0}".format(neuron_number)]["g"].append(g_t)
                    self.neurons["neuron{0}".format(neuron_number)]["alpha"].append(alpha_t)
                    self.neurons["neuron{0}".format(neuron_number)]["theta"].append(theta_t)
                    g_previous = self.neurons["neuron{0}".format(neuron_number)]["g"][i-1]
                    alpha_previous = self.neurons["neuron{0}".format(neuron_number)]["alpha"][i - 1]
                    theta_previous = self.neurons["neuron{0}".format(neuron_number)]["theta"][i - 1]
                else:
                    g_previous = 1
                    alpha_previous = 1
                    theta_previous = 1

                out = torch.div(torch.mul(g_previous, torch.log(1 + torch.exp(torch.mul(alpha_previous, outLin - theta_previous)))),
                                alpha_previous)
                self.neurons["neuron{0}".format(neuron_number)]["output"].append(out)

                output[i,:, neuron_number] = out.reshape(-1,)




        return output, self.neurons