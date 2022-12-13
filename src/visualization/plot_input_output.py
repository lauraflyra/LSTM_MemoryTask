import matplotlib.pyplot as plt
import numpy as np
from src.data.create_input import create_input, create_output, create_go_signal

def plot_input(x, y):
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

def plot_result_training(x, output, out_pred, train_error, n_epochs, plot_every, title = "title"):
    tot_time_steps, batch_size, input_size = x.shape
    which_from_batch = np.random.randint(batch_size, size=1)
    spacing = np.linspace(0, 10, input_size)
    x_plot = x[:, which_from_batch, :].reshape(tot_time_steps, input_size) + spacing
    output_plot = output[:, which_from_batch, :].reshape(tot_time_steps, input_size) + spacing
    out_pred_plot = out_pred[:, which_from_batch, :].reshape(tot_time_steps, input_size) + spacing
    time_array = np.arange(0, tot_time_steps, 1)
    fig, axs = plt.subplots(nrows=1,ncols=2)
    axs[0].plot(time_array, x_plot[:,0:4], color = 'blue')
    axs[0].plot(time_array, x_plot[:, 4:8], color='red')
    axs[0].plot(time_array, x_plot[:, 8:], color='orange')
    axs[0].set_title("input")

    axs[1].plot(time_array, output_plot[:,0:4], color = 'blue')
    axs[1].plot(time_array, output_plot[:, 4:8], color='red')
    axs[1].plot(time_array, output_plot[:, 8:], color='orange')
    axs[1].plot(time_array, out_pred_plot[:, 0:4], color ='green', alpha=0.6)
    axs[1].plot(time_array, out_pred_plot[:, 4:8], color='green', alpha=0.6)
    axs[1].plot(time_array, out_pred_plot[:, 8:], color='green', alpha=0.6)
    axs[1].set_title("output")

    plt.suptitle(title)
    plt.show()

    plt.plot(np.linspace(plot_every, n_epochs, len(train_error[1:])),train_error[1:])
    plt.title("Training error")
    plt.show()
if __name__ == "__main__":
    x = create_input()
    x, go_signal_idx, go_signal_moments = create_go_signal(x)
    output = create_output(x, go_signal_idx, go_signal_moments)
    plot_input(x, output)


