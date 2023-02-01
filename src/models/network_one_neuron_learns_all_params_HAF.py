import torch
import torch.nn as nn


"""
Activation Function:
gamma(x; g, theta, alpha) = g * log(1+exp(alpha*(x-theta)))/alpha
where g is the gain, theta is threshold, alpha is the softening parameter.  
The difference from this and network_one_neuron, is that alpha and theta are also learned by the Network.
HAF stands for Henning Activation Function, because it was proposed by him :)
"""


class NetworkOneNeuronLearnAll(nn.Module):
    """
    Create a network with one neuron, that has its inner chemistry represented by an LSTM.
    LSTM outputs gain parameter of the AF. Other two are set to 1.
    LSTM also receives recurrent feedback.
    """

    def __init__(self, input_dim=1, hidden_size=64):
        """

        :param input_dim: input and output dimension
        :param hidden_size: hidden size of the LSTM
        """
        super(NetworkOneNeuronLearnAll, self).__init__()

        self.hidden_size = hidden_size
        # Create neuron, linearLSTM is still part of the LSTM 'neuron internal dynamics'
        self.lstm = nn.LSTMCell(input_dim + input_dim,
                                 self.hidden_size)  # LSTM receives combined input and output, but input_dim = output_dim

        self.linearLSTM = nn.Linear(self.hidden_size, 3)

        # Input -> output
        self.linearHidden = nn.Linear(input_dim, 10)
        self.hiddenGamma = nn.Linear(10, input_dim)

    def forward(self, input):

        # We need to initialize the output to feed into the LSTM, so we make it be equal to the first input

        outputs, n_samples, g, theta, alpha = [], input.size(1), [], [], []
        h_t = torch.zeros(n_samples, self.hidden_size, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.hidden_size, dtype=torch.float32)
        g_0 = torch.ones(n_samples)
        theta_0 = torch.ones(n_samples)
        alpha_0 = torch.ones(n_samples)

        # We want the LSTM to learn the parameters for the next activation function, IN t+1 time!!

        for i in range(input.size()[0]):  # Assuming x is (time_steps, batch, input_size)

            outHid = self.linearHidden(input[i])
            linOut = torch.reshape(self.hiddenGamma(outHid), (-1,))

            if i > 0:
                combined = torch.cat((input[i], outputs[i - 1]), 1)
                h_t, c_t = self.lstm(combined, (h_t, c_t))  # initial hidden and cell states
                g_t, theta_t, alpha_t = self.linearLSTM(h_t).T
                out = torch.div(torch.mul(g[i-1], torch.log(1 + torch.exp(torch.mul(alpha[i-1], linOut - theta[i-1])))), alpha[i-1])
            else:
                g_t, theta_t, alpha_t = g_0, theta_0, alpha_0
                out = torch.div(torch.mul(g_t, torch.log(1 + torch.exp(torch.mul(alpha_t, linOut - theta_t)))),
                      alpha_t)


            output = torch.reshape(out, (-1,1))
            outputs.append(output)

            g.append(g_t)
            theta.append(theta_t)
            alpha.append(alpha_t)


        out = torch.stack(outputs, dim=0)
        g_out = torch.stack(g, dim = 0)
        theta_out = torch.stack(theta, dim=0)
        alpha_out = torch.stack(alpha, dim=0)

        return out, (g_out, theta_out, alpha_out)
