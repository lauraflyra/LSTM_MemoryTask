import torch
import torch.nn as nn
from src.data.create_input import PEOPLE, INPUT_SIZE
"""
Activation function used:
gamma(x; g, theta, alpha) = g * log(1+exp(alpha*(x-theta)))/alpha

where g is the gain, theta is threshold, alpha is the softening parameter. All 3 are learned by the LSTM! 
"""


class Network(nn.Module):
    """
    Create a network where each neuron is has its inner chemistry represented by an LSTM.
    All neurons are the same, i.e, they have the same inner chemistry mechanisms, therefore the same LSTM.
    All neurons are independent from one another.
    Each neuron receives a 2 dimensional input. In the first input dimension we have the professors time slots
    in the second input dimension we have his schedule for the day.
    At some random point, we send the neuron the query for which lab member had a meeting at a given time.
    We expect the neuron to have an output in 1 hot encoding for the correspondent person.
    """

    def __init__(self, hidden_size_LSTM=64):
        """

        """
        super(Network, self).__init__()

        self.hidden_size_LSTM = hidden_size_LSTM

        # create the LSTM
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size_LSTM) # TODO: think about this input size
        # create LSTM output
        self.linearLSTM = nn.Linear(self.hidden_size_LSTM, 3)       # LSTM outputs 3 parameters for the AF

        # create one neuron
        self.linearNeuron = nn.Linear(INPUT_SIZE, 1)    # TODO: QUESTION: neuron gives it 1 dimensional input to LSTM?
        # For now there's no hidden layer for the neuron
        self.neuronOutput = nn.Linear(1, len(PEOPLE))   # so we can get the one-hot encoding, we'll pass this through a softmax

    def forward(self, input):
        """

        :param input:
        :return:
        """
        n_samples, g, theta, alpha, outputs = input.size(1), [], [], [], []
        # initialize the hidden states for the LSTM
        h_t = torch.zeros(n_samples, self.hidden_size, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.hidden_size, dtype=torch.float32)
        # initialize the parameters from the transfer function
        # g_0 = torch.ones(n_samples)
        # theta_0 = torch.ones(n_samples)
        # alpha_0 = torch.ones(n_samples)

        for i in range(input.size()[0]):  # Assuming x is (time_steps, batch, input_size)

            # First the input enters the neuron
            neuronCombined = self.linearNeuron(input[i])
            # then the LSTM receives the input combined by the neuron weights
            h_t, c_t = self.lstm(neuronCombined, (h_t,c_t))
            # then the LSTM hidden states are used to compute the parameters of the AF
            g_t, theta_t, alpha_t = self.linearLSTM(h_t).T
            # then neuronCombined goes through the activation function
            outTemp = torch.div(torch.mul(g_t, torch.log(1 + torch.exp(torch.mul(alpha_t,neuronCombined-theta_t)))),alpha_t)
            #
            output = self.neuronOutput(outTemp)
            probOut = nn.functional.softmax(output)  # TODO: need to insert softmax dimension

            output = torch.reshape(probOut, (-1,len(PEOPLE)))
            outputs.append(output)

            g.append(g_t)
            theta.append(theta_t)
            alpha.append(alpha_t)

        out = torch.stack(outputs, dim=0)
        g_out = torch.stack(g, dim = 0)
        theta_out = torch.stack(theta, dim=0)
        alpha_out = torch.stack(alpha, dim=0)
        return out, g_out, theta_out, alpha_out

