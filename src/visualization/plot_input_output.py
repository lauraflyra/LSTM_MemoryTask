import matplotlib.pyplot as plt
import numpy as np
from src.data.common_vars import PEOPLE
import torch
import os
import scipy.io as io
import matplotlib.colors as mcolors


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


def plot_result_training_factorial(x, output, out_pred, train_error, n_epochs, plot_every, title="title"):
    tot_time_steps, batch_size, input_size = x.shape
    which_from_batch = np.random.randint(batch_size, size=1)
    spacing = np.arange(0, input_size, 1)
    x_plot = x[:, which_from_batch, :].reshape(tot_time_steps, input_size) + spacing
    output_plot = output[:, which_from_batch, :].reshape(tot_time_steps, input_size) + spacing
    out_pred_plot = out_pred[:, which_from_batch, :].reshape(tot_time_steps, input_size) + spacing
    time_array = np.arange(0, tot_time_steps, 1)
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].plot(time_array, x_plot[:, 0:4], color='blue')
    axs[0].plot(time_array, x_plot[:, 4:8], color='red')
    axs[0].plot(time_array, x_plot[:, 8:], color='orange')
    axs[0].set_title("input")

    axs[1].plot(time_array, output_plot[:, 0:4], color='blue', linewidth=3)
    axs[1].plot(time_array, output_plot[:, 4:8], color='red', linewidth=3)
    axs[1].plot(time_array, output_plot[:, 8:], color='orange', linewidth=3)
    axs[1].plot(time_array, out_pred_plot[:, 0:4], color='black', alpha=0.6, label='predicted')
    axs[1].plot(time_array, out_pred_plot[:, 4:8], color='black', alpha=0.6)
    axs[1].plot(time_array, out_pred_plot[:, 8:], color='black', alpha=0.6)
    axs[1].set_title("output")

    axs[0].set_xlabel('time points')
    axs[1].set_xlabel('time points')
    axs[0].set_ylabel('neuron number')

    # black_patch = mpatches.Patch(color='black', label='predicted')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower right')
    plt.suptitle(title)
    plt.show()

    plt.plot(np.linspace(plot_every, n_epochs, len(train_error[1:])), train_error[1:])
    plt.xlabel('epochs')
    plt.title("Training error")
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
        plt.plot(time_array, x_plot[:, person_idx], color=colors[person_idx])

    plt.plot(time_array, x_plot[:, len(PEOPLE):], color='black', label="time slot")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title("input")
    plt.xlabel("time points")
    plt.ylabel("Neuron number")
    plt.show()

    # plot expected output and predicted output
    for person_idx in range(len(PEOPLE)):
        plt.plot(time_array, output_plot[:, person_idx], color=colors[person_idx], linewidth=3)

    plt.plot(time_array, out_pred_plot, color='black', alpha=0.6, label='predicted')
    plt.title("output")

    plt.xlabel('time points')
    plt.xlabel('time points')
    plt.ylabel('neuron number')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower right')
    plt.show()

    # plot the training error
    plt.plot(np.linspace(plot_every, n_epochs, len(train_error[1:])), train_error[1:])
    plt.xlabel('epochs')
    plt.title("Training error")
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
        plt.title("input training random batch n {}".format(i))
        plt.xlabel("time points")
        plt.ylabel("Neuron number")
        name = save_name_input + str(i) + ".png"
        plt.savefig(os.path.join(save_data_path, name))
        plt.show()

        for person_idx in range(len(PEOPLE)):
            plt.plot(time_array, output_plot[:, person_idx], color=colors[person_idx], linewidth=3)

        plt.plot(time_array, out_pred_plot, color='black', alpha=0.6, label='predicted')
        plt.title("output training random batch n {}".format(i))

        plt.xlabel('time points')
        plt.xlabel('time points')
        plt.ylabel('neuron number')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='lower right')
        name = save_name_output + str(i) + ".png"
        plt.savefig(os.path.join(save_data_path, name))
        plt.show()

    return


if __name__ == "__main__":
    import os
    from src.visualization.plot_input_output import plot_result_training_additive

    DATA_PATH = "/home/lauraflyra/Documents/SHK_SprekelerLab/LSTM_computations/LSTM_MemoryTask/src/data/results"
    PATH_RESULTS_ADDITIVE = os.path.join(DATA_PATH, "additive")

    which_from_batch = plot_result_training_additive(params_file=os.path.join(PATH_RESULTS_ADDITIVE, "params_additive.mat"))

    plot_result_multiple_training_additive(os.path.join(PATH_RESULTS_ADDITIVE, "params_additive.mat"), save_data_path=PATH_RESULTS_ADDITIVE)

#
#     x = create_input()
#     x, go_signal_idx, go_signal_moments = create_go_signal(x)
#     output = create_output(x, go_signal_idx, go_signal_moments)
#     plot_input_factorial(x, output)
#     x, go_signal_time_slots, go_signal_moments = create_input_go_add()
#     output = create_output_add(x, go_signal_time_slots, go_signal_moments)
#     plot_input_additive(x)
