import torch
import torch.nn as nn



class NetworkOneNeuronGeadah(nn.Module):
    """
    Create a network where each neuron is an LSTM, and all neurons are the same,
    and independent from one another.
    For now I start with one neuron.
    Each neuron has a time series of 1 dimension as input and outputs 2 real numbers, s and n.
    """

    def __init__(self, input_dim=1, hidden_size=64):
        """

        :param input_dim: input and output dimension
        :param hidden_size: hidden size of the LSTM
        """
        super(NetworkOneNeuronGeadah, self).__init__()

        self.hidden_size = hidden_size
        # Create neuron number 1, linearLSTM is still part of the LSTM 'neuron internal dynamics'
        self.lstm1 = nn.LSTMCell(input_dim + input_dim,
                                 self.hidden_size)  # LSTM receives combined input and output, but input_dim = output_dim
        self.linearLSTM = nn.Linear(self.hidden_size, 2)

        self.linearGamma = nn.Linear(input_dim, input_dim)  # Input and output have the same dimension
        # self.linearHidden = nn.Linear(input_dim, 10)
        # self.hiddenGamma = nn.Linear(10, input_dim)

    def forward(self, input):

        # We need to initialize the output to feed into the LSTM, so we make it be equal to the first input
        # This would be desired, so it's all good ????
        outputs, n_samples, n, s = [], input.size(1), [], []
        h_t = torch.zeros(n_samples, self.hidden_size, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.hidden_size, dtype=torch.float32)
        n_0 = torch.ones(n_samples)
        s_0 = torch.ones(n_samples)
        for i in range(input.size()[0]):  # Assuming x is (time_steps, batch, input_size)
            if i > 0:
                combined = torch.cat((input[i], outputs[i - 1]), 1)
                h_t, c_t = self.lstm1(combined, (h_t, c_t))  # initial hidden and cell states
                n_t, s_t = self.linearLSTM(h_t).T
            else:
                n_t, s_t = n_0, s_0

            linOut = torch.reshape(self.linearGamma(input[i]), (-1,))
            gamma_one = torch.div(1, n_t) * torch.log(1 + torch.exp(n_t * linOut))
            gamma_two = torch.sigmoid(torch.mul(n_t, linOut))

            output = torch.reshape(torch.mul((1 - s_t), gamma_one) + torch.mul(s_t, gamma_two), (-1, 1))

            outputs.append(output)
            n.append(n_t)
            s.append(s_t)

        out = torch.stack(outputs, dim=0)
        n_out = torch.stack(n, dim = 0)
        s_out = torch.stack(s, dim=0)
        return out, (n_out, s_out)

