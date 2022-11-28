import torch
import torch.nn as nn
from src.data.create_input import PEOPLE, INPUT_SIZE
"""
Activation function used:
gamma(x; g, theta, alpha) = g * log(1+exp(alpha*(x-theta)))/alpha
where g is the gain, theta is threshold, alpha is the softening parameter. Only the gain is learned by the LSTM. 
"""

# See: https://stackoverflow.com/questions/47952930/how-can-i-use-lstm-in-pytorch-for-classification

class Network(nn.Module):
    """
    Create a network where each neuron is has its inner chemistry represented by an LSTM.
    All neurons are the same, i.e, they have the same inner chemistry mechanisms, therefore the same LSTM.
    All neurons are independent from one another.
    Each neuron receives a Number of People + 1 dimensional input. In the first Np dimensions we have the one hot
    encoding for people, in the last dimension we have the time slot for that person.
    At some random point, we send the neuron the query for which lab member had a meeting at a given time.
    The query comes as zeros in the first Np dimensions and the time slot.
    We expect the neuron to have an output in 1 hot encoding for the person that had the slot in the queried time.
    """

    def __init__(self, hidden_size_LSTM=64):
        """
        """
        super(Network, self).__init__()

        self.hidden_size_LSTM = hidden_size_LSTM

        # create the LSTM
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size_LSTM, batch_first=True) # TODO: think about this input size
        # create LSTM output
        self.linearLSTM = nn.Linear(self.hidden_size_LSTM, 1)       # LSTM outputs 1 parameter for the AF

        # create neurons
        self.neuronsInput = {}
        self.neuronsOutput = {}
        for neuron_number in range(len(PEOPLE)):
            self.neuronsInput["neuron{0}".format(neuron_number)] = nn.Linear(INPUT_SIZE, 1) # TODO: QUESTION: neuron gives it 1 dimensional input to LSTM?
            # Each neuron outputs something between 0 and 1, then overall I want all neurons output to be compared to the
            # one hot encoded output
            self.neuronsOutput["neuron{0}".format(neuron_number)] = nn.Linear(1, 1)


    def forward(self, input):
        """
        :param input:
        :return:
        """
        n_samples, g, theta, alpha, outputs = input.size(0), [], [], [], []
        # initialize the hidden states for the LSTM
        hidden_lstm = {}
        af_params = {}
        # TODO: maybe initialize hidden states in a separate function
        # for neuron_number in range(len(PEOPLE)):
        #     hidden_lstm["h_t{0}".format(neuron_number)] = torch.zeros(n_samples, self.hidden_size, dtype=torch.float32)
        #     hidden_lstm["c_t{0}".format(neuron_number)] = torch.zeros(n_samples, self.hidden_size, dtype=torch.float32)
            # af_params["g_t{0}".format(neuron_number)] = torch.ones(n_samples)
        # initialize the parameters from the transfer function
        # g_0 = torch.ones(n_samples)
        # theta_0 = torch.ones(n_samples)
        # alpha_0 = torch.ones(n_samples)

        # for i in range(input.size()[1]):  # Assuming x is (batch, time_steps, input_size) # TODO: probably I should not have this loop

        # First the input enters the neurons
        neurons_intermediate_out = {}
        prob_outputs = []
        for neuron_number in range(len(PEOPLE)):
            neurons_intermediate_out["neuron{0}".format(neuron_number)] = self.neuronsInput["neuron{0}".format(neuron_number)](input[i])
            # then the LSTM receives the input combined by the neuron weights
            hidden_lstm["h_t{0}".format(neuron_number)], hidden_lstm["c_t{0}".format(neuron_number)] = self.lstm(neurons_intermediate_out["neuron{0}".format(neuron_number)],
                                                                                                                 (hidden_lstm["h_t{0}".format(neuron_number)], hidden_lstm["c_t{0}".format(neuron_number)]))
            # then the LSTM hidden states are used to compute the parameters of the AF
            af_params["g_t{0}".format(neuron_number)] = self.linearLSTM(hidden_lstm["h_t{0}".format(neuron_number)]).T

            # then neuron intermediate outputs goes through the activation function
            outTemp = torch.div(torch.mul(af_params["g_t{0}".format(neuron_number)], torch.log(1 + torch.exp(torch.mul(1,neurons_intermediate_out["neuron{0}".format(neuron_number)]-1)))),1)
            #
            output = self.neuronsOutput["neuron{0}".format(neuron_number)](outTemp)
            prob_outputs.append(nn.functional.softmax(output))  # TODO: need to insert softmax dimension

            # TODO: THIS IMPLEMENTATION IS NOT COMPLETE, LIKE THIS I HAVE A TIME SERIES OUTPUT, BUT I WANT TO HAVE ONLY ONE TIME STEP OUTPUT AS A ONE HOT ENCODED VECTOR
            # As you feed your sentence in word-by-word (x_i-by-x_i+1), you get an output from each timestep. You want to interpret the entire sentence to classify it.
            # So you must wait until the LSTM has seen all the words. That is, you need to take h_t where t is the number of words in your sentence.


            # output should be one number, such that prob_outputs is a list of probabilities of being each person
            # output = prob_outputs
            # outputs.append(output)

            # g.append(g_t)
            # theta.append(theta_t)
            # alpha.append(alpha_t)

        out = torch.stack(outputs, dim=0)
        # g_out = torch.stack(g, dim = 0)
        # theta_out = torch.stack(theta, dim=0)
        # alpha_out = torch.stack(alpha, dim=0)
        # return out, g_out, theta_out, alpha_out