import matplotlib.pyplot as plt
import numpy as np
from src.data.common_vars import PEOPLE, TIME_SLOTS
from src.data.create_input_output_additive import create_input_go_add, create_output_add
from src.data.create_input_output_factorial import create_input, create_output, create_go_signal
import matplotlib.patches as mpatches

def plot_input_factorial(x, y):
    tot_time_steps, batch_size, input_size = x.shape
    which_from_batch = np.random.randint(batch_size, size = 1)
    spacing = np.linspace(0,10, input_size)
    x_plot = x[:,which_from_batch,:].reshape(tot_time_steps, input_size)+spacing
    y_plot = y[:, which_from_batch, :].reshape(tot_time_steps, input_size) + spacing
    time_array = np.arange(0, tot_time_steps, 1)
    fig, axs = plt.subplots(nrows=1,ncols=2)
    axs[0].plot(time_array, x_plot[:,0:4], color = 'blue')
    axs[0].plot(time_array, x_plot[:, 4:8], color='red')
    axs[0].plot(time_array, x_plot[:, 8:], color='orange')
    axs[0].set_title("input")
    axs[1].plot(time_array, y_plot[:,0:4], color = 'blue')
    axs[1].plot(time_array, y_plot[:, 4:8], color='red')
    axs[1].plot(time_array, y_plot[:, 8:], color='orange')
    axs[1].set_title("output")
    plt.show()


def plot_result_training_factorial(x, output, out_pred, train_error, n_epochs, plot_every, title = "title"):
    tot_time_steps, batch_size, input_size = x.shape
    which_from_batch = np.random.randint(batch_size, size=1)
    spacing = np.arange(0, input_size, 1)
    x_plot = x[:, which_from_batch, :].reshape(tot_time_steps, input_size) + spacing
    output_plot = output[:, which_from_batch, :].reshape(tot_time_steps, input_size) + spacing
    out_pred_plot = out_pred[:, which_from_batch, :].reshape(tot_time_steps, input_size) + spacing
    time_array = np.arange(0, tot_time_steps, 1)
    fig, axs = plt.subplots(nrows=1,ncols=2)
    axs[0].plot(time_array, x_plot[:,0:4], color = 'blue')
    axs[0].plot(time_array, x_plot[:, 4:8], color='red')
    axs[0].plot(time_array, x_plot[:, 8:], color='orange')
    axs[0].set_title("input")

    axs[1].plot(time_array, output_plot[:,0:4], color = 'blue', linewidth = 3)
    axs[1].plot(time_array, output_plot[:, 4:8], color='red', linewidth = 3)
    axs[1].plot(time_array, output_plot[:, 8:], color='orange', linewidth = 3)
    axs[1].plot(time_array, out_pred_plot[:, 0:4], color ='black', alpha=0.6, label = 'predicted')
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

    plt.plot(np.linspace(plot_every, n_epochs, len(train_error[1:])),train_error[1:])
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
    plt.plot(time_array, x_plot[:,:len(PEOPLE)])
    plt.plot(time_array, x_plot[:, len(PEOPLE):], color='black', label = "time slot")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title("input")
    plt.show()
    return

def plot_result_training_additive(x, output, out_pred, train_error, n_epochs, plot_every):
    tot_time_steps, batch_size, input_size = x.shape
    which_from_batch = np.random.randint(batch_size, size=1)
    spacing_y = np.arange(0, len(PEOPLE), 1)
    spacing_x = np.arange(0, input_size, 1)
    x_plot = x[:, which_from_batch, :].reshape(tot_time_steps, input_size) + spacing_x
    output_plot = output[:, which_from_batch, :].reshape(tot_time_steps, len(PEOPLE)) + spacing_y
    out_pred_plot = out_pred[:, which_from_batch, :].reshape(tot_time_steps, len(PEOPLE)) + spacing_y
    time_array = np.arange(0, tot_time_steps, 1)


    plt.plot(time_array, x_plot[:,:len(PEOPLE)])
    plt.plot(time_array, x_plot[:, len(PEOPLE):], color='black', label = "time slot")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title("input")
    plt.show()

    plt.plot(time_array, output_plot, color = 'blue', linewidth = 3)
    plt.plot(time_array, out_pred_plot, color ='black', alpha=0.6, label = 'predicted')
    plt.title("output")

    plt.xlabel('time points')
    plt.xlabel('time points')
    plt.ylabel('neuron number')

    # black_patch = mpatches.Patch(color='black', label='predicted')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower right')
    plt.show()

    plt.plot(np.linspace(plot_every, n_epochs, len(train_error[1:])),train_error[1:])
    plt.xlabel('epochs')
    plt.title("Training error")
    plt.show()
    return which_from_batch




if __name__ == "__main__":
    x = create_input()
    x, go_signal_idx, go_signal_moments = create_go_signal(x)
    output = create_output(x, go_signal_idx, go_signal_moments)
    plot_input_factorial(x, output)
    x, go_signal_time_slots, go_signal_moments = create_input_go_add()
    output = create_output_add(x, go_signal_time_slots, go_signal_moments)
    plot_input_additive(x)


