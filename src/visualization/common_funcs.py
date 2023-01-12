import numpy as np


def activation_Function(x, g, theta, alpha):
    """
    Activation Function used in the network models. It is implemented by hand in the network classes.
    Here we have it as a helper for the plotting AF functions
    :param x: function variable
    :param g: gain
    :param theta: threshold
    :param alpha: softening
    """
    return g * np.log(1 + np.exp(alpha * (x - theta))) / alpha
