import scipy.io as io
import os
import matplotlib.pyplot as plt
import numpy as np
from src.data.create_input_output_factorial import *
import matplotlib.patches as mpatches

DATA_PATH = "/home/lauraflyra/Documents/SHK_SprekelerLab/LSTM_computations/LSTM_MemoryTask/src/data/"

def activation_Function(x, g, theta, alpha):
    return g * np.log(1 + np.exp(alpha * (x - theta))) / alpha


def plot_AF(params_file, go_signal_idx, go_signal_moments, which_from_batch):

    data = io.loadmat(params_file, struct_as_record=False, squeeze_me = True)
    # data has keys like 'neuron0-g', 'neuron1-g'. We want to plot the non-linearities for each neurno
    n_t_samples, time_points, batch_size = data['neuron0-g'].shape
    which_time_slot = go_signal_idx[which_from_batch]
    when_go_signal = go_signal_moments[which_from_batch]
    spacing = np.linspace(0, 20, INPUT_SIZE)

    x_range_af = np.linspace(-0.5,1.5, time_points)
    # Need to get function for specific time points

    colors = {
        'greys': plt.cm.Greys(np.linspace(0.1, 1, int(time_points))),
        'reds': plt.cm.Reds(np.linspace(0.1, 1, int(time_points))),
        'greens': plt.cm.Greens(np.linspace(0.1, 1, int(time_points))),
    }

    fig, ax = plt.subplots(nrows=INPUT_SIZE, sharex=True, figsize=(12, 20), gridspec_kw={'hspace': 0})
    fig.tight_layout()
    for i in range(INPUT_SIZE):
        neuron_g = data['neuron'+str(i)+'-g'][-1,:,which_from_batch]
        ax[i].text(1.2, 0.4, 'neuron {}'.format(i))
        if 4*which_time_slot <= i < 4*which_time_slot+4:
            ax[i].text(1.0, 0.6, 'recovered times slot')
        for t in range(time_points):
            if t < CUE_TIME:
                ax[i].plot(x_range_af, activation_Function(x_range_af, neuron_g[0,t],1,1), color=colors['greys'][t])

            if CUE_TIME <= t <= when_go_signal:
                ax[i].plot(x_range_af, activation_Function(x_range_af, neuron_g[0,t], 1, 1), color=colors['reds'][t])

            if t > when_go_signal:
                ax[i].plot(x_range_af, activation_Function(x_range_af, neuron_g[0,t], 1, 1), color=colors['greens'][t])

    grey_patch = mpatches.Patch(color='grey', label='before cue')
    red_patch = mpatches.Patch(color='red', label='between cue and go')
    green_patch = mpatches.Patch(color='green', label='after go')

    plt.legend(handles=[grey_patch, red_patch, green_patch], bbox_to_anchor=[0.5, 1])

    plt.show()

    return

def plot_AF_separate_neurons(params_file, go_signal_idx, go_signal_moments, which_from_batch):

    data = io.loadmat(params_file, struct_as_record=False, squeeze_me = True)
    # data has keys like 'neuron0-g', 'neuron1-g'. We want to plot the non-linearities for each neurno
    n_t_samples, time_points, batch_size = data['neuron0-g'].shape
# TODO: better would be if which from batch could be argument, so we can compare this to result form plot result training.
#     which_from_batch = np.random.randint(batch_size, size=1)
    which_time_slot = go_signal_idx[which_from_batch]
    when_go_signal = go_signal_moments[which_from_batch]
    spacing = np.linspace(0, 20, INPUT_SIZE)

    x_range_af = np.linspace(-0.5,1.5, time_points)
    # Need to get function for specific time points

    colors = {
        'greys': plt.cm.Greys(np.linspace(0.1, 1, int(time_points))),
        'reds': plt.cm.Reds(np.linspace(0.1, 1, int(time_points))),
        'greens': plt.cm.Greens(np.linspace(0.1, 1, int(time_points))),
    }

    # fig, ax = plt.subplots(nrows=INPUT_SIZE, sharex=True, figsize=(12, 20), gridspec_kw={'hspace': 0})
    # fig.tight_layout()
    for i in range(INPUT_SIZE):
        neuron_g = data['neuron'+str(i)+'-g'][-1,:,which_from_batch]
        plt.title('neuron {}'.format(i))
        if 4*which_time_slot <= i < 4*which_time_slot+4:
            plt.title('neuron {}; recovered times slot'.format(i))
        for t in range(time_points):
            if t < CUE_TIME:
                plt.plot(x_range_af, activation_Function(x_range_af, neuron_g[0,t],1,1), color=colors['greys'][t])

            if CUE_TIME <= t < when_go_signal:
                plt.plot(x_range_af, activation_Function(x_range_af, neuron_g[0,t], 1, 1), color=colors['reds'][t])

            if t >= when_go_signal:
                plt.plot(x_range_af, activation_Function(x_range_af, neuron_g[0,t], 1, 1), color=colors['greens'][t])

        grey_patch = mpatches.Patch(color='grey', label='before cue')
        red_patch = mpatches.Patch(color='red', label='between cue and go')
        green_patch = mpatches.Patch(color='green', label='after go')

        plt.xlabel('z')
        plt.ylabel('gamma(z;g)')
        plt.legend(handles=[grey_patch, red_patch, green_patch], bbox_to_anchor=[0.5, 1])
        plt.show()




    return



if __name__ == "__main__":

    x = create_input()
    x, go_signal_idx, go_signal_moments = create_go_signal(x)
    output = create_output(x, go_signal_idx, go_signal_moments)
    plot_AF(params_file=os.path.join(DATA_PATH, "params.mat"), go_signal_idx = go_signal_idx,
            go_signal_moments = go_signal_moments, which_from_batch = 0)
