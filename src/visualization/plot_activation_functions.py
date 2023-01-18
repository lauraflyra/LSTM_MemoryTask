import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
from src.visualization.common_funcs import activation_Function
from src.data.create_input_output_additive import *
import matplotlib.patches as mpatches

def geadah_activation_function(x, n, s):
    first_factor =  (1 - s)*(np.log(1+np.exp(n*x)))/n
    second_factor = s*(np.exp(n*x)/(1+np.exp(n*x)))
    return first_factor*second_factor

def plot_params_af_in_time_one_neuron(params_file):
    data = io.loadmat(params_file, struct_as_record=False, squeeze_me=True)
    n_t_samples, time_points, batch_size = data['g'].shape

    cue_amps = data['cue_amp']

    which_plot_no_bump = np.random.choice(np.where(np.abs(cue_amps) < 0.5)[0])
    which_plot_wt_bump = np.random.choice(np.where(np.abs(cue_amps) > 0.5)[0])

    which_plots = [which_plot_no_bump, which_plot_wt_bump]

    from src.data.create_input_output_one_neuron import CUE_START_TIME, CUE_END_TIME, CUE_DURATION, GO_DURATION

    GO_START_TIME = data["go_start_time"]

    colors = {
        'greys': plt.cm.Greys(np.linspace(0.1, 1, int(time_points))),
        'reds': plt.cm.Reds(np.linspace(0.5, 1, GO_DURATION)),
        'greens': plt.cm.Greens(np.linspace(0.1, 1, CUE_DURATION)),
        'purples': plt.cm.Purples(np.linspace(0.1, 1, CUE_DURATION + 1)), }

    gs = data['g'][-1, :, which_plots]

    z = np.linspace(-0.5, 1.5)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    axs[0, 0].plot(data['x'][:, which_plots[0]], linewidth = 4, alpha = 0.5)
    axs[0, 0].set_xlabel('Time', fontsize = 15)
    axs[0, 0].set_ylabel('Input strength', fontsize=15)
    axs[0, 0].set_title('Input', fontsize = 15)
    axs[1, 0].plot(data['x'][:, which_plots[1]], linewidth = 4, alpha = 0.5)
    axs[1, 0].set_ylabel('Input strength', fontsize=15)
    axs[1, 0].set_xlabel('Time', fontsize = 15)

    GO_END_TIME = GO_START_TIME + GO_DURATION

    OUTPUT_RESPONSE_START = GO_START_TIME
    OUTPUT_RESPONSE_END = OUTPUT_RESPONSE_START + CUE_DURATION

    GO_START_1 = GO_START_TIME[which_plots[0]]
    GO_END_1 = GO_END_TIME[which_plots[0]]
    GO_START_2 = GO_START_TIME[which_plots[1]]
    GO_END_2 = GO_END_TIME[which_plots[1]]

    OUT_START_1 = OUTPUT_RESPONSE_START[which_plots[0]]
    OUT_END_1 = OUTPUT_RESPONSE_END[which_plots[0]]
    OUT_START_2 = OUTPUT_RESPONSE_START[which_plots[1]]
    OUT_END_2 = OUTPUT_RESPONSE_END[which_plots[1]]

    for t in range(time_points):
        if t > 0:
            axs[0, 1].plot(z, activation_Function(z, gs[0, t], 1,1), color=colors['greys'][t])
            axs[1, 1].plot(z, activation_Function(z, gs[1, t], 1,1), color=colors['greys'][t])

        if CUE_START_TIME < t < CUE_END_TIME:
            axs[0, 1].plot(z, activation_Function(z, gs[0, t], 1,1),
                           color=colors['purples'][t - CUE_START_TIME], linewidth=4)
            axs[1, 1].plot(z, activation_Function(z, gs[1, t], 1,1),
                           color=colors['purples'][t - CUE_START_TIME], linewidth=4)
            axs[0, 0].scatter(t, data['x'][t, which_plots[0]], color=colors['purples'][t - CUE_START_TIME], linewidth=4)
            axs[1, 0].scatter(t, data['x'][t, which_plots[1]], color=colors['purples'][t - CUE_START_TIME], linewidth=4)

        if OUT_START_1 <= t < OUT_END_1:
            axs[0, 1].plot(z, activation_Function(z, gs[0, t], 1,1),
                           color=colors['greens'][t - OUT_END_1], linewidth=3)
            axs[0, 0].scatter(t, data['x'][t, which_plots[0]], color=colors['greens'][t - OUT_END_1], linewidth=10)

        if OUT_START_2 <= t < OUT_END_2:
            axs[1, 1].plot(z, activation_Function(z, gs[1, t], 1,1),
                           color=colors['greens'][t - OUT_START_2], linewidth=3)
            axs[1, 0].scatter(t, data['x'][t, which_plots[1]], color=colors['greens'][t - OUT_START_2], linewidth=10)

        if GO_START_1 <= t < GO_END_1:
            axs[0, 1].plot(z, activation_Function(z, gs[0, t], 1,1),
                           color=colors['reds'][t - GO_START_1], linewidth=4)
            axs[0, 0].scatter(t, data['x'][t, which_plots[0]], color=colors['reds'][t - GO_START_1], linewidth=10)

        if GO_START_2 <= t < GO_END_2:
            axs[1, 1].plot(z, activation_Function(z, gs[1, t], 1,1),
                           color=colors['reds'][t - GO_START_2], linewidth=4)
            axs[1, 0].scatter(t, data['x'][t, which_plots[1]], color=colors['reds'][t - GO_START_2], linewidth=10)


    red_patch = mpatches.Patch(color=colors['reds'][-1], label='go signal times')
    purple_patch = mpatches.Patch(color=colors['purples'][-1], label='cue times')
    green_patch = mpatches.Patch(color=colors['greens'][-1], label='response times')

    plt.legend(handles=[purple_patch, red_patch, green_patch], bbox_to_anchor=[1, 1])

    axs[0, 1].set_title('Non-linearity', fontsize = 15)
    axs[1, 1].set_title('Colors from light to dark represent time points'.format(which_plots[1]), fontsize = 15)
    axs[0, 1].set_xlabel('z', fontsize = 15)
    axs[1, 1].set_xlabel('z', fontsize = 15)
    axs[0, 1].set_ylabel('gamma(z;n,s)', fontsize = 15)
    axs[1, 1].set_ylabel('gamma(z;n,s)', fontsize = 15)

    axs[0, 0].spines[['right', 'top']].set_visible(False)
    axs[0, 1].spines[['right', 'top']].set_visible(False)
    axs[1, 0].spines[['right', 'top']].set_visible(False)
    axs[1, 1].spines[['right', 'top']].set_visible(False)

    axs[0, 0].spines[['bottom', 'left']].set_linewidth(3)
    axs[0, 1].spines[['bottom', 'left']].set_linewidth(3)
    axs[1, 0].spines[['bottom', 'left']].set_linewidth(3)
    axs[1, 1].spines[['bottom', 'left']].set_linewidth(3)

    axs[0, 0].tick_params(width=3)
    axs[0, 1].tick_params(width=3)
    axs[1, 0].tick_params(width=3)
    axs[1, 1].tick_params(width=3)

    plt.show()
    return



def plot_params_af_in_time_one_neuron_learns_all_HAF(params_file):
    data = io.loadmat(params_file, struct_as_record=False, squeeze_me=True)
    n_t_samples, time_points, batch_size = data['g'].shape

    cue_amps = data['cue_amp']

    which_plot_no_bump = np.where(np.abs(cue_amps) < 0.5)[0][0]
    which_plot_wt_bump = np.where(np.abs(cue_amps) >= 0.5)[0][0]

    which_plots = [which_plot_no_bump, which_plot_wt_bump]

    from src.data.create_input_output_one_neuron import CUE_START_TIME, CUE_END_TIME, CUE_DURATION, GO_DURATION

    GO_START_TIME = data["go_start_time"]

    colors = {
        'greys': plt.cm.Greys(np.linspace(0.1, 1, int(time_points))),
        'reds': plt.cm.Reds(np.linspace(0.1, 1, GO_DURATION)),
        'greens': plt.cm.Greens(np.linspace(0.1, 1, CUE_DURATION)),
        'purples': plt.cm.Purples(np.linspace(0.1, 1, CUE_DURATION + 1)), }

    gs = data['g'][-1, :, which_plots]
    thetas = data['theta'][-1, :, which_plots]
    alphas = data['alpha'][-1, :, which_plots]

    z = np.linspace(-0.5, 1.5)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    axs[0, 0].plot(data['x'][:, which_plots[0]], linewidth = 4, alpha = 0.5)
    axs[0, 0].set_xlabel('Time', fontsize = 15)
    axs[0, 0].set_ylabel('Input strength', fontsize=15)
    axs[0, 0].set_title('Input', fontsize = 15)
    axs[1, 0].plot(data['x'][:, which_plots[1]], linewidth = 4, alpha = 0.5)
    axs[1, 0].set_ylabel('Input strength', fontsize=15)
    axs[1, 0].set_xlabel('Time', fontsize = 15)

    GO_END_TIME = GO_START_TIME + GO_DURATION

    OUTPUT_RESPONSE_START = GO_START_TIME
    OUTPUT_RESPONSE_END = OUTPUT_RESPONSE_START + CUE_DURATION

    GO_START_1 = GO_START_TIME[which_plots[0]]
    GO_END_1 = GO_END_TIME[which_plots[0]]
    GO_START_2 = GO_START_TIME[which_plots[1]]
    GO_END_2 = GO_END_TIME[which_plots[1]]

    OUT_START_1 = OUTPUT_RESPONSE_START[which_plots[0]]
    OUT_END_1 = OUTPUT_RESPONSE_END[which_plots[0]]
    OUT_START_2 = OUTPUT_RESPONSE_START[which_plots[1]]
    OUT_END_2 = OUTPUT_RESPONSE_END[which_plots[1]]

    for t in range(time_points):
        if t > 0:
            axs[0, 1].plot(z, activation_Function(z, gs[0, t], thetas[0, t], alphas[0, t]), color=colors['greys'][t])
            axs[1, 1].plot(z, activation_Function(z, gs[1, t], thetas[1, t], alphas[1, t]), color=colors['greys'][t])

        if CUE_START_TIME < t < CUE_END_TIME:
            axs[0, 1].plot(z, activation_Function(z, gs[0, t], thetas[0, t], alphas[0, t]),
                           color=colors['purples'][t - CUE_START_TIME], linewidth=4)
            axs[1, 1].plot(z, activation_Function(z, gs[1, t], thetas[1, t], alphas[1, t]),
                           color=colors['purples'][t - CUE_START_TIME], linewidth=4)
            axs[0, 0].scatter(t, data['x'][t, which_plots[0]], color=colors['purples'][t - CUE_START_TIME], linewidth=4)
            axs[1, 0].scatter(t, data['x'][t, which_plots[1]], color=colors['purples'][t - CUE_START_TIME], linewidth=4)

        if OUT_START_1 <= t < OUT_END_1:
            axs[0, 1].plot(z, activation_Function(z, gs[0, t], thetas[0, t], alphas[0, t]),
                           color=colors['greens'][t - OUT_END_1], linewidth=3)
            axs[0, 0].scatter(t, data['x'][t, which_plots[0]], color=colors['greens'][t - OUT_END_1], linewidth=10)

        if OUT_START_2 <= t < OUT_END_2:
            axs[1, 1].plot(z, activation_Function(z, gs[1, t], thetas[1, t], alphas[1, t]),
                           color=colors['greens'][t - OUT_START_2], linewidth=3)
            axs[1, 0].scatter(t, data['x'][t, which_plots[1]], color=colors['greens'][t - OUT_START_2], linewidth=10)

        if GO_START_1 <= t < GO_END_1:
            axs[0, 1].plot(z, activation_Function(z, gs[0, t], thetas[0, t], alphas[0, t]),
                           color=colors['reds'][t - GO_START_1], linewidth=4)
            axs[0, 0].scatter(t, data['x'][t, which_plots[0]], color=colors['reds'][t - GO_START_1], linewidth=10)

        if GO_START_2 <= t < GO_END_2:
            axs[1, 1].plot(z, activation_Function(z, gs[1, t], thetas[1, t], alphas[1, t]),
                           color=colors['reds'][t - GO_START_2], linewidth=4)
            axs[1, 0].scatter(t, data['x'][t, which_plots[1]], color=colors['reds'][t - GO_START_2], linewidth=10)


    red_patch = mpatches.Patch(color=colors['reds'][-2], label='go signal times')
    purple_patch = mpatches.Patch(color=colors['purples'][-2], label='cue times')
    green_patch = mpatches.Patch(color=colors['greens'][-2], label='response times')

    plt.legend(handles=[purple_patch, red_patch, green_patch], bbox_to_anchor=[1, 1])

    axs[0, 1].set_title('Non-linearity', fontsize = 15)
    axs[1, 1].set_title('Colors from light to dark represent time points'.format(which_plots[1]), fontsize = 15)
    axs[0, 1].set_xlabel('z', fontsize = 15)
    axs[1, 1].set_xlabel('z', fontsize = 15)
    axs[0, 1].set_ylabel('gamma(z;n,s)', fontsize = 15)
    axs[1, 1].set_ylabel('gamma(z;n,s)', fontsize = 15)

    axs[0,0].spines[['right', 'top']].set_visible(False)
    axs[0, 1].spines[['right', 'top']].set_visible(False)
    axs[1, 0].spines[['right', 'top']].set_visible(False)
    axs[1, 1].spines[['right', 'top']].set_visible(False)

    axs[0,0].spines[['bottom', 'left']].set_linewidth(3)
    axs[0, 1].spines[['bottom', 'left']].set_linewidth(3)
    axs[1, 0].spines[['bottom', 'left']].set_linewidth(3)
    axs[1, 1].spines[['bottom', 'left']].set_linewidth(3)

    axs[0,0].tick_params(width=3)
    axs[0, 1].tick_params(width=3)
    axs[1, 0].tick_params(width=3)
    axs[1, 1].tick_params(width=3)

    plt.show()
    return


def plot_params_af_in_time_one_neurons_Geadah(params_file):
    data = io.loadmat(params_file, struct_as_record=False, squeeze_me=True)
    n_t_samples, time_points, batch_size = data['n'].shape

    cue_amps = data['cue_amp']

    which_plot_no_bump = np.where(np.abs(cue_amps) < 0.5)[0][0]
    which_plot_wt_bump = np.where(np.abs(cue_amps) >= 0.5)[0][0]

    which_plots = [which_plot_no_bump, which_plot_wt_bump]

    from src.data.create_input_output_one_neuron import CUE_START_TIME, CUE_END_TIME, CUE_DURATION, GO_DURATION

    GO_START_TIME = data["go_start_time"]

    colors = {
        'greys': plt.cm.Greys(np.linspace(0.1, 1, int(time_points))),
        'reds': plt.cm.Reds(np.linspace(0.1, 1, GO_DURATION)),
        'greens': plt.cm.Greens(np.linspace(0.1, 1, CUE_DURATION)),
        'purples': plt.cm.Purples(np.linspace(0.1, 1, CUE_DURATION + 1)), }

    ns = data['n'][-1, :, which_plots]
    ss = data['s'][-1, :, which_plots]

    z = np.linspace(-0.5, 1.5)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    axs[0, 0].plot(data['x'][:, which_plots[0]], linewidth = 4, alpha = 0.5)
    axs[0, 0].set_xlabel('Time', fontsize = 15)
    axs[0, 0].set_ylabel('Input strength', fontsize=15)
    axs[0, 0].set_title('Input', fontsize = 15)
    axs[1, 0].plot(data['x'][:, which_plots[1]], linewidth = 4, alpha = 0.5)
    axs[1, 0].set_ylabel('Input strength', fontsize=15)
    axs[1, 0].set_xlabel('Time', fontsize = 15)

    GO_END_TIME = GO_START_TIME + GO_DURATION

    OUTPUT_RESPONSE_START = GO_END_TIME
    OUTPUT_RESPONSE_END = OUTPUT_RESPONSE_START + CUE_DURATION

    GO_START_1 = GO_START_TIME[which_plots[0]]
    GO_END_1 = GO_END_TIME[which_plots[0]]
    GO_START_2 = GO_START_TIME[which_plots[1]]
    GO_END_2 = GO_END_TIME[which_plots[1]]

    OUT_START_1 = OUTPUT_RESPONSE_START[which_plots[0]]
    OUT_END_1 = OUTPUT_RESPONSE_END[which_plots[0]]
    OUT_START_2 = OUTPUT_RESPONSE_START[which_plots[1]]
    OUT_END_2 = OUTPUT_RESPONSE_END[which_plots[1]]

    for t in range(time_points):
        if t > 0:
            axs[0, 1].plot(z, geadah_activation_function(z, ns[0, t], ss[0, t]), color=colors['greys'][t])
            axs[1, 1].plot(z, geadah_activation_function(z, ns[1, t], ss[1, t]), color=colors['greys'][t])

        if CUE_START_TIME < t < CUE_END_TIME:
            axs[0, 1].plot(z, geadah_activation_function(z, ns[0, t], ss[0, t]),
                           color=colors['purples'][t - CUE_START_TIME], linewidth=4)
            axs[1, 1].plot(z, geadah_activation_function(z, ns[1, t], ss[1, t]),
                           color=colors['purples'][t - CUE_START_TIME], linewidth=4)
            axs[0, 0].scatter(t, data['x'][t, which_plots[0]], color=colors['purples'][t - CUE_START_TIME], linewidth=4)
            axs[1, 0].scatter(t, data['x'][t, which_plots[1]], color=colors['purples'][t - CUE_START_TIME], linewidth=4)

        if OUT_START_1 <= t < OUT_END_1:
            axs[0, 1].plot(z, geadah_activation_function(z, ns[0, t], ss[0, t]),
                           color=colors['greens'][t - OUT_END_1], linewidth=3)
            axs[0, 0].scatter(t, data['x'][t, which_plots[0]], color=colors['greens'][t - OUT_END_1], linewidth=10)

        if OUT_START_2 <= t < OUT_END_2:
            axs[1, 1].plot(z, geadah_activation_function(z, ns[1, t], ss[1, t]),
                           color=colors['greens'][t - OUT_START_2], linewidth=3)
            axs[1, 0].scatter(t, data['x'][t, which_plots[1]], color=colors['greens'][t - OUT_START_2], linewidth=10)

        if GO_START_1 <= t < GO_END_1:
            axs[0, 1].plot(z, geadah_activation_function(z, ns[0, t], ss[0, t]),
                           color=colors['reds'][t - GO_START_1], linewidth=4)
            axs[0, 0].scatter(t, data['x'][t, which_plots[0]], color=colors['reds'][t - GO_START_1], linewidth=10)

        if GO_START_2 <= t < GO_END_2:
            axs[1, 1].plot(z, geadah_activation_function(z, ns[1, t], ss[1, t]),
                           color=colors['reds'][t - GO_START_2], linewidth=4)
            axs[1, 0].scatter(t, data['x'][t, which_plots[1]], color=colors['reds'][t - GO_START_2], linewidth=10)


    red_patch = mpatches.Patch(color=colors['reds'][-2], label='go signal times')
    purple_patch = mpatches.Patch(color=colors['purples'][-2], label='cue times')
    green_patch = mpatches.Patch(color=colors['greens'][-2], label='response times')

    plt.legend(handles=[purple_patch, red_patch, green_patch], bbox_to_anchor=[1, 1])

    axs[0, 1].set_title('Non-linearity', fontsize = 15)
    axs[1, 1].set_title('Colors from light to dark represent time points'.format(which_plots[1]), fontsize = 15)
    axs[0, 1].set_xlabel('z', fontsize = 15)
    axs[1, 1].set_xlabel('z', fontsize = 15)
    axs[0, 1].set_ylabel('gamma(z;n,s)', fontsize = 15)
    axs[1, 1].set_ylabel('gamma(z;n,s)', fontsize = 15)

    axs[0,0].spines[['right', 'top']].set_visible(False)
    axs[0, 1].spines[['right', 'top']].set_visible(False)
    axs[1, 0].spines[['right', 'top']].set_visible(False)
    axs[1, 1].spines[['right', 'top']].set_visible(False)

    axs[0,0].spines[['bottom', 'left']].set_linewidth(3)
    axs[0, 1].spines[['bottom', 'left']].set_linewidth(3)
    axs[1, 0].spines[['bottom', 'left']].set_linewidth(3)
    axs[1, 1].spines[['bottom', 'left']].set_linewidth(3)

    axs[0,0].tick_params(width=3)
    axs[0, 1].tick_params(width=3)
    axs[1, 0].tick_params(width=3)
    axs[1, 1].tick_params(width=3)

    plt.show()
    return

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
