import scipy.io as io
import matplotlib.pyplot as plt
from src.visualization.common_funcs import activation_Function
from src.data.create_input_output_additive import *
import matplotlib.patches as mpatches



def plot_af_additive_neurons(params_file, which_from_batch=None):
    """
    :param params_file: file containing dictionary, created in the training process.
    It contains "neuronN-g" (corresponding to the g parameter for each neuron in the network)
     + "x", "y", "y_pred", "training_error", "go_signal_time_slots" and "go_signal_moments".
    :param which_from_batch: plot the activation functions corresponding to this sample from the batch.
    If None is provided, we choose a random sample from the batch.
    """
    data = io.loadmat(params_file, struct_as_record=False, squeeze_me=True)

    # data has keys like 'neuron0-g', 'neuron1-g'. We want to plot the non-linearities for each neuron

    n_t_samples, time_points, batch_size = data['neuron0-g'].shape
    go_signal_moments = data['go_signal_moments']

    if which_from_batch is None:
        which_from_batch = np.random.randint(batch_size)

    when_go_signal = go_signal_moments[which_from_batch]

    x_range_af = np.linspace(-0.5, 1.5, time_points)

    colors = {
        'greys': plt.cm.Greys(np.linspace(0.1, 1, int(time_points))),
        'reds': plt.cm.Reds(np.linspace(0.1, 1, int(time_points))),
        'greens': plt.cm.Greens(np.linspace(0.1, 1, int(time_points))),
    }

    for key in data.keys():
        if key.startswith('neuron'):
            neuron_g = data[key][-1, :, which_from_batch]
            plt.title(key)
            # if i == which_time_slot:
            #     plt.title('neuron {}; recovered times slot'.format(i))
            for t in range(time_points):
                if t < CUE_START_TIME:
                    plt.plot(x_range_af, activation_Function(x_range_af, neuron_g[t], 1, 1),
                             color=colors['greys'][t])

                if CUE_START_TIME <= t < when_go_signal:
                    plt.plot(x_range_af, activation_Function(x_range_af, neuron_g[t], 1, 1), color=colors['reds'][t])

                if t >= when_go_signal:
                    plt.plot(x_range_af, activation_Function(x_range_af, neuron_g[t], 1, 1),
                             color=colors['greens'][t])

            grey_patch = mpatches.Patch(color='grey', label='before cue')
            red_patch = mpatches.Patch(color='red', label='between cue and go')
            green_patch = mpatches.Patch(color='green', label='after go')

            plt.xlabel('z')
            plt.ylabel('gamma(z;g)')
            plt.legend(handles=[grey_patch, red_patch, green_patch], bbox_to_anchor=[0.5, 1])
            plt.show()

    return
