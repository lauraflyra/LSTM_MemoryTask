import scipy.io as io
import os
import matplotlib.pyplot as plt
import numpy as np
from src.data.create_input import INPUT_SIZE

DATA_PATH = "/home/lauraflyra/Documents/SHK_SprekelerLab/LSTM_computations/LSTM_MemoryTask/src/data/"

def activation_Function(x, g, theta, alpha):
    return g * np.log(1 + np.exp(alpha * (x - theta))) / alpha


def plot_AF(params_file): #, go_signal_idx, go_signal_moments):

    data = io.loadmat(params_file, struct_as_record=False, squeeze_me = True)
    # data has keys like 'neuron0-g', 'neuron1-g'. We want to plot the non-linearities for each neurno
    n_t_samples, time_points, batch_size = data['neuron0-g'].shape

    which_from_batch = np.random.randint(batch_size, size=1)
    spacing = np.linspace(0, 20, INPUT_SIZE)

    x_range_af = np.linspace(-0.5,1.5, time_points)
    # Need to get function for specific time points

    colors = {
        'reds': plt.cm.Reds(np.linspace(0.1, 1, int(time_points))),
    }

    fig, ax = plt.subplots(nrows=INPUT_SIZE, sharex=True, figsize=(12, 20), gridspec_kw={'hspace': 0})
    fig.tight_layout()
    for i in range(INPUT_SIZE):
        neuron_g = data['neuron'+str(i)+'-g'][-1,:,which_from_batch]
        for t in range(time_points):
            ax[i].plot(x_range_af, activation_Function(x_range_af, neuron_g[0, t],1,1), color=colors['reds'][t])
            ax[i].text(1.2, 0.4, 'neuron {}'.format(i))
    plt.show()

    return


if __name__ == "__main__":
    plot_AF(params_file=os.path.join(DATA_PATH, "params.mat"))