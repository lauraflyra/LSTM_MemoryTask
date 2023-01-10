import os.path
from src.data.create_test_data_additive import *
import torch
import matplotlib.pyplot as plt
from src.visualization.plot_activation_function_additive import activation_Function
import matplotlib.patches as mpatches

model = torch.load("/home/lauraflyra/Documents/SHK_SprekelerLab/LSTM_computations/LSTM_MemoryTask/src/data/model_additive_all_neurons_separate_linear.pt")
out_test, model_variables = model(torch.from_numpy(x_test).float())
SAVE_DATA_PATH = "/home/lauraflyra/Documents/SHK_SprekelerLab/LSTM_computations/LSTM_MemoryTask/src/data/data_all_neurons_separate_linear_additive"


def plot_result_test_additive(x_test, y_test, out_test):
    tot_time_steps, batch_size, input_size = x_test.shape
    spacing_y = np.arange(0, len(PEOPLE), 1)
    spacing_x = np.arange(0, input_size, 1)

    time_array = np.arange(0, tot_time_steps, 1)

    for i in range(batch_size):
        x_plot = x_test[:, i, :] + spacing_x
        output_plot = y_test[:, i, :] + spacing_y
        out_pred_plot = out_test[:, i, :] + spacing_y

        plt.plot(time_array, x_plot[:,:len(PEOPLE)])
        plt.plot(time_array, x_plot[:, len(PEOPLE):], color='black', label = "time slot")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.title("input test batch n {}".format(i))
        name = "additive_input_batch_"+str(i)+".png"
        plt.xlabel("time points")
        plt.ylabel("Neuron number")
        plt.savefig(os.path.join(SAVE_DATA_PATH, name))
        plt.show()

        plt.plot(time_array, output_plot, color = 'blue', linewidth = 3)
        plt.plot(time_array, out_pred_plot, color ='black', alpha=0.6, label = 'predicted')
        plt.title("output test batch n {}".format(i))

        plt.xlabel('time points')
        plt.xlabel('time points')
        plt.ylabel('neuron number')

        # black_patch = mpatches.Patch(color='black', label='predicted')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='lower right')
        name = "additive_output_batch_"+str(i)+".png"
        plt.savefig(os.path.join(SAVE_DATA_PATH, name))
        plt.show()

    return

plot_result_test_additive(x_test, y_test, out_test.detach().numpy())


def plot_AF_test_separate_neurons_add(data, go_signal_time_slots, go_signal_moments, batch_size):

    x_range_af = np.linspace(-0.5,1.5, TIME_STEPS)
    # Need to get function for specific time points

    colors = {
        'greys': plt.cm.Greys(np.linspace(0.1, 1, int(TIME_STEPS))),
        'reds': plt.cm.Reds(np.linspace(0.1, 1, int(TIME_STEPS))),
        'greens': plt.cm.Greens(np.linspace(0.1, 1, int(TIME_STEPS))),
    }
    for batch in range(batch_size):

        for i in range(len(PEOPLE)):

            neuron_g = torch.stack(data['neuron{}'.format(i)]['g'][1:]).reshape(TIME_STEPS, batch_size).detach().numpy()[:,batch]
            plt.title('batch {}, neuron {}'.format(batch,i))
            # if i == go_signal_time_slots[batch]:
            #     plt.title('neuron {}; recovered times slot'.format(i))
            for t in range(TIME_STEPS):
                if t < CUE_START_TIME:
                    plt.plot(x_range_af, activation_Function(x_range_af, neuron_g[t],1,1), color=colors['greys'][t])

                if CUE_START_TIME <= t < go_signal_moments[batch]:
                    plt.plot(x_range_af, activation_Function(x_range_af, neuron_g[t], 1, 1), color=colors['reds'][t])

                if t >= go_signal_moments[batch]:
                    plt.plot(x_range_af, activation_Function(x_range_af, neuron_g[t], 1, 1), color=colors['greens'][t])

            grey_patch = mpatches.Patch(color='grey', label='before cue')
            red_patch = mpatches.Patch(color='red', label='between cue and go')
            green_patch = mpatches.Patch(color='green', label='after go')

            plt.xlabel('z')
            plt.ylabel('gamma(z;g)')
            plt.legend(handles=[grey_patch, red_patch, green_patch], bbox_to_anchor=[0.5, 1])
            name = "additive_activation_function_batch" + str(batch) + "neuron_" + str(i) + ".png"
            plt.savefig(os.path.join(SAVE_DATA_PATH, name))
            plt.show()



    return

plot_AF_test_separate_neurons_add(model_variables, go_signal_time_slots, go_signal_moments, batch_size)