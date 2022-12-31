import os.path
from src.data.create_test_data_factorial import *
import torch
import matplotlib.pyplot as plt
from src.visualization.plot_activation_func_factorial import activation_Function
import matplotlib.patches as mpatches


model = torch.load("/home/lauraflyra/Documents/SHK_SprekelerLab/LSTM_computations/LSTM_MemoryTask/src/data/model_factorial_all_neurons_same_linear.pt")
out_test, model_variables = model(torch.from_numpy(x_test).float())
SAVE_DATA_PATH = "/home/lauraflyra/Documents/SHK_SprekelerLab/LSTM_computations/LSTM_MemoryTask/src/data/data_all_neurons_same_linear_factorial_2"

def plot_result_test_factorial(x_test, y_test, out_test):
    tot_time_steps, batch_size, input_size = x_test.shape
    spacing = np.arange(0, input_size, 1)
    time_array = np.arange(0, tot_time_steps, 1)
    for i in range(batch_size):
        x_plot = x_test[:, i, :] + spacing
        output_plot = y_test[:, i, :] + spacing
        out_pred_plot = out_test[:, i, :] + spacing


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

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='lower right')
        plt.suptitle("Test batch n {}".format(i))
        name = "input_output_batch_"+str(i)+".png"
        plt.savefig(os.path.join(SAVE_DATA_PATH, name))
        plt.show()

    return

plot_result_test_factorial(x_test, y_test, out_test.detach().numpy())




def plot_AF_test_separate_neurons(data, go_signal_idx_test, go_signal_moments_test, batch_size):
    x_range_af = np.linspace(-0.5, 1.5, TOTAL_TIME_STEPS)
    colors = {
        'greys': plt.cm.Greys(np.linspace(0.1, 1, int(TOTAL_TIME_STEPS))),
        'reds': plt.cm.Reds(np.linspace(0.1, 1, int(TOTAL_TIME_STEPS))),
        'greens': plt.cm.Greens(np.linspace(0.1, 1, int(TOTAL_TIME_STEPS))),
    }
    for batch in range(batch_size):

        for i in range(INPUT_SIZE):
            neuron_g = torch.stack(data['neuron{}'.format(i)]['g'][1:]).reshape(TOTAL_TIME_STEPS, batch_size).detach().numpy()[:,batch]
            plt.title('batch {}, neuron {}'.format(batch, i))
            if 4*go_signal_idx_test[batch] <= i < 4*go_signal_idx_test[batch]+4:
                plt.title('batch {}, neuron {}; recovered times slot'.format(batch, i))
            for t in range(TOTAL_TIME_STEPS):
                if t < CUE_TIME:
                    plt.plot(x_range_af, activation_Function(x_range_af, neuron_g[t],1,1), color=colors['greys'][t])

                if CUE_TIME <= t < go_signal_moments_test[batch]:
                    plt.plot(x_range_af, activation_Function(x_range_af, neuron_g[t], 1, 1), color=colors['reds'][t])

                if t >= go_signal_moments_test[batch]:
                    plt.plot(x_range_af, activation_Function(x_range_af, neuron_g[t], 1, 1), color=colors['greens'][t])

            grey_patch = mpatches.Patch(color='grey', label='before cue')
            red_patch = mpatches.Patch(color='red', label='between cue and go')
            green_patch = mpatches.Patch(color='green', label='after go')

            plt.xlabel('z')
            plt.ylabel('gamma(z;g)')
            plt.legend(handles=[grey_patch, red_patch, green_patch], bbox_to_anchor=[0.5, 1])
            name = "activation_function_batch" + str(batch) +"neuron_"+str(i)+".png"
            plt.savefig(os.path.join(SAVE_DATA_PATH, name))
            plt.show()

    return

plot_AF_test_separate_neurons(model_variables, go_signal_idx_test, go_signal_moments_test, batch_size)