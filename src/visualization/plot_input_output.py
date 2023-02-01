import matplotlib.pyplot as plt
import numpy as np
from src.data.common_vars import PEOPLE
import os
import scipy.io as io
import matplotlib.colors as mcolors

"""
Plotting functions for visualizing input, output and predicted output for the models. 
"""


def plot_result_training_one_neuron(params_file):

    data = io.loadmat(params_file, struct_as_record=False, squeeze_me=True)
    x = data['x']
    output = data['y']
    out_pred = data['y_pred']
    train_error = data['training_error']
    n_epochs = data['n_epochs']
    plot_every = data['save_loss_every']
    cue_amps = data['cue_amp']

    which_plot_no_bump = np.random.choice(np.where(np.abs(cue_amps) < 0.5)[0])
    which_plot_wt_bump = np.random.choice(np.where(np.abs(cue_amps) > 0.5)[0])

    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    axs[0].plot(x[:, which_plot_no_bump], label="input", linewidth = 5, alpha = 0.5)
    axs[0].set_title("No cue", fontsize = 17)
    axs[0].set_xlabel("Time", fontsize = 17)
    axs[0].set_ylabel("Neural response/ Input strength", fontsize=17)
    axs[0].legend()
    #
    axs[0].plot(output[:, which_plot_no_bump], label="target output", linewidth = 5, alpha = 0.7)
    #

    axs[0].plot(out_pred[-1, :, which_plot_no_bump], label="predicted output", linewidth = 5, alpha = 0.5)
    axs[0].legend()

    axs[1].plot(x[:, which_plot_wt_bump], label="input", linewidth = 5, alpha = 0.5)
    axs[1].set_title("Cue and go", fontsize = 17)
    axs[1].set_xlabel("Time", fontsize = 17)
    axs[1].set_ylabel("Neural response/ Input strength", fontsize=17)
    axs[1].legend()
    #
    axs[1].plot(output[:, which_plot_wt_bump], label="target output", linewidth = 5, alpha = 0.7)
    #

    axs[1].plot(out_pred[-1,:, which_plot_wt_bump], label="predicted output", linewidth = 5, alpha = 0.5)
    axs[1].legend()
    axs[0].set_ylim(axs[1].get_ylim())
    axs[0].spines[['right', 'top']].set_visible(False)
    axs[1].spines[['right', 'top']].set_visible(False)
    axs[0].spines[['right', 'top']].set_visible(False)
    axs[1].spines[['right', 'top']].set_visible(False)

    axs[0].spines[['bottom', 'left']].set_linewidth(4)
    axs[1].spines[['bottom', 'left']].set_linewidth(4)

    axs[0].tick_params(width=4)
    axs[1].tick_params(width=4)

    plt.show()

    fig, ax = plt.subplots(1,1)
    plt.plot(np.arange(plot_every, n_epochs, plot_every), train_error[1:], label="train error", linewidth = 4, alpha = 0.5)
    plt.xlabel("Epochs", fontsize = 15)
    plt.ylabel("MSE", fontsize = 15)
    plt.title("Mean Squared Error during training", fontsize = 15)
    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(4)
    # increase tick width
    ax.tick_params(width=4)
    plt.tight_layout()
    plt.show()

    return



def plot_input_factorial(x, y):
    tot_time_steps, batch_size, input_size = x.shape
    which_from_batch = np.random.randint(batch_size, size=1)
    spacing = np.linspace(0, 10, input_size)
    x_plot = x[:, which_from_batch, :].reshape(tot_time_steps, input_size) + spacing
    y_plot = y[:, which_from_batch, :].reshape(tot_time_steps, input_size) + spacing
    time_array = np.arange(0, tot_time_steps, 1)
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].plot(time_array, x_plot[:, 0:4], color='blue')
    axs[0].plot(time_array, x_plot[:, 4:8], color='red')
    axs[0].plot(time_array, x_plot[:, 8:], color='orange')
    axs[0].set_title("input")
    axs[1].plot(time_array, y_plot[:, 0:4], color='blue')
    axs[1].plot(time_array, y_plot[:, 4:8], color='red')
    axs[1].plot(time_array, y_plot[:, 8:], color='orange')
    axs[1].set_title("output")
    plt.show()


def plot_result_training_factorial(params_file):

    data = io.loadmat(params_file, struct_as_record=False, squeeze_me=True)
    x = data['x']
    output = data['y']
    out_pred = data['y_pred']
    train_error = data['training_error']
    n_epochs = data['n_epochs']
    plot_every = data['save_loss_every']


    tot_time_steps, batch_size, input_size = x.shape
    which_from_batch = np.random.randint(batch_size, size=1)

    spacing = np.arange(0, input_size, 1)

    x_plot = 0.8*x[:, which_from_batch, :].reshape(tot_time_steps, input_size) + spacing
    output_plot = 0.8*output[:, which_from_batch, :].reshape(tot_time_steps, input_size) + spacing
    out_pred_plot = 0.8*out_pred[-1,:, which_from_batch, :].reshape(tot_time_steps, input_size) + spacing

    time_array = np.arange(0, tot_time_steps, 1)

    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].plot(time_array, x_plot[:, 0:4], color='blue', linewidth = 4, alpha = 0.7)
    axs[0].plot(time_array, x_plot[:, 4:8], color='red', linewidth = 4, alpha = 0.7)
    axs[0].plot(time_array, x_plot[:, 8:], color='orange', linewidth = 4, alpha = 0.7)
    axs[0].set_title("input", fontsize = 15)

    axs[1].plot(time_array, output_plot[:, 0:4], color='blue', linewidth = 4, alpha = 0.7)
    axs[1].plot(time_array, output_plot[:, 4:8], color='red', linewidth = 4, alpha = 0.7)
    axs[1].plot(time_array, output_plot[:, 8:], color='orange', linewidth = 4, alpha = 0.7)
    axs[1].plot(time_array, out_pred_plot[:, 0:4], color='black', alpha=0.6, label='predicted', linewidth = 3)
    axs[1].plot(time_array, out_pred_plot[:, 4:8], color='black', alpha=0.6, linewidth = 3)
    axs[1].plot(time_array, out_pred_plot[:, 8:], color='black', alpha=0.6, linewidth = 3)
    axs[1].set_title("output", fontsize = 15)

    axs[0].set_xlabel('time points', fontsize = 15)
    axs[1].set_xlabel('time points', fontsize = 15)
    axs[0].set_ylabel('neuron number', fontsize = 15)

    # black_patch = mpatches.Patch(color='black', label='predicted')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower right')
    plt.show()

    fig, ax = plt.subplots(1,1)
    plt.plot(np.linspace(plot_every, n_epochs, len(train_error[1:])), train_error[1:], linewidth = 5, alpha = 0.7)
    plt.xlabel("Epochs", fontsize = 15)
    plt.ylabel("MSE", fontsize = 15)
    plt.title("Mean Squared Error during training", fontsize = 15)
    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(4)
    # increase tick width
    ax.tick_params(width=4)
    plt.tight_layout()
    plt.show()
    return which_from_batch


def plot_input_additive(x):
    tot_time_steps, batch_size, input_size = x.shape
    which_from_batch = np.random.randint(batch_size, size=1)
    spacing = np.arange(0, input_size, 1)
    x_plot = x[:, which_from_batch, :].reshape(tot_time_steps, input_size) - spacing
    time_array = np.arange(0, tot_time_steps, 1)
    plt.plot(time_array, x_plot[:, :len(PEOPLE)])
    plt.plot(time_array, x_plot[:, len(PEOPLE):], color='black', label="time slot")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title("input")
    plt.show()
    return


def plot_result_training_additive(params_file, which_from_batch=None):
    """
    :param params_file: file containing dictionary, created in the training process.
    It contains "neuronN-g","x", "y", "y_pred", "training_error", "go_signal_time_slots",
    "go_signal_moments", "n_epochs", "save_loss_every".
    :param which_from_batch: plot the input and output corresponding to this sample from the batch.
    If None is provided, we choose a random sample from the batch.
    :return: which_from_batch, in case we want to plot activation functions for the same sample
    """

    data = io.loadmat(params_file, struct_as_record=False, squeeze_me=True)

    x = data['x']
    output = data['y']
    out_pred = data['y_pred']
    train_error = data['training_error']
    n_epochs = data['n_epochs']
    plot_every = data['save_loss_every']

    tot_time_steps, batch_size, input_size = x.shape
    if which_from_batch is None:
        which_from_batch = np.random.randint(batch_size, size=1)

    # add some spacing so plotting looks nicer
    spacing_y = np.arange(0, len(PEOPLE), 1)
    spacing_x = np.arange(0, input_size, 1)

    x_plot = x[:, which_from_batch, :].reshape(tot_time_steps, input_size) + spacing_x
    output_plot = output[:, which_from_batch, :].reshape(tot_time_steps, len(PEOPLE)) + spacing_y

    # take the last from the predicted outputs, i.e, the one at the end of training
    out_pred_plot = out_pred[-1, :, which_from_batch, :].reshape(tot_time_steps, len(PEOPLE)) + spacing_y
    time_array = np.arange(0, tot_time_steps, 1)

    colors = list(mcolors.TABLEAU_COLORS.keys())

    # plot input
    for person_idx in range(len(PEOPLE)):
        plt.plot(time_array, x_plot[:, person_idx], color=colors[person_idx], linewidth = 4, alpha = 0.7)

    plt.plot(time_array, x_plot[:, len(PEOPLE):], color='black', label="time slot", linewidth = 4, alpha = 0.7)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title("input", fontsize = 15)
    plt.xlabel("time points", fontsize = 15)
    plt.ylabel("Neuron number", fontsize = 15)
    plt.show()

    # plot expected output and predicted output
    for person_idx in range(len(PEOPLE)):
        plt.plot(time_array, output_plot[:, person_idx], color=colors[person_idx], linewidth = 4, alpha = 0.7)

    plt.plot(time_array, out_pred_plot, color='black', alpha=0.5, label='predicted', linewidth = 2)
    plt.title("output", fontsize = 15)

    plt.xlabel('time points', fontsize = 15)
    plt.xlabel('time points', fontsize = 15)
    plt.ylabel('neuron number', fontsize = 15)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower right')
    plt.show()

    # plot the training error
    fig, ax = plt.subplots(1,1)
    plt.plot(np.linspace(plot_every, n_epochs, len(train_error[1:])), train_error[1:], linewidth = 4, alpha = 0.7)
    plt.xlabel("Epochs", fontsize = 15)
    plt.ylabel("MSE", fontsize = 15)
    plt.title("Mean Squared Error during training", fontsize = 15)
    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(4)
    # increase tick width
    ax.tick_params(width=4)
    plt.tight_layout()
    plt.show()
    return which_from_batch


def plot_result_multiple_training_additive(params_file,
                                           save_data_path,
                                           n_plots = 10,
                                           save_name_input = "additive_input_random_batch_",
                                           save_name_output = "additive_output_random_batch_"
                                           ):

    data = io.loadmat(params_file, struct_as_record=False, squeeze_me=True)
    # take random samples from batch
    time_steps, total_batch, input_dim = data["x"].shape
    random_batch = np.random.randint(0, total_batch, size=n_plots)
    x = data['x'][:, random_batch, :]
    y = data['y'][:, random_batch, :]
    out_pred = data['y_pred'][-1, :, random_batch, :].reshape(time_steps, n_plots, -1)

    spacing_y = np.arange(0, len(PEOPLE), 1)
    spacing_x = np.arange(0, input_dim, 1)

    time_array = np.arange(0, time_steps, 1)

    colors = list(mcolors.TABLEAU_COLORS.keys())

    for i in range(n_plots):
        x_plot = x[:, i, :] + spacing_x
        output_plot = y[:, i, :] + spacing_y
        out_pred_plot = out_pred[:, i, :] + spacing_y

        for person_idx in range(len(PEOPLE)):
            plt.plot(time_array, x_plot[:, person_idx], color=colors[person_idx])

        plt.plot(time_array, x_plot[:, len(PEOPLE):], color='black', label="time slot")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.title("input training random batch n {}".format(i), fontsize = 15)
        plt.xlabel("time points", fontsize = 15)
        plt.ylabel("Neuron number", fontsize = 15)
        name = save_name_input + str(i) + ".png"
        plt.savefig(os.path.join(save_data_path, name))
        plt.show()

        for person_idx in range(len(PEOPLE)):
            plt.plot(time_array, output_plot[:, person_idx], color=colors[person_idx], linewidth=3)

        plt.plot(time_array, out_pred_plot, color='black', alpha=0.6, label='predicted')
        plt.title("output training random batch n {}".format(i), fontsize = 15)

        plt.xlabel('time points', fontsize = 15)
        plt.ylabel('neuron number', fontsize = 15)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='lower right')
        name = save_name_output + str(i) + ".png"
        plt.savefig(os.path.join(save_data_path, name))
        plt.show()

    return



