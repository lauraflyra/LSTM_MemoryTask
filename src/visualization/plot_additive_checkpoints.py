import torch
import matplotlib.pyplot as plt
from src.data.common_vars import PEOPLE, TIME_SLOTS
from src.models.network_additive import NetworkAdditive
from scipy import io
import os
import numpy as np

SAVE_DATA_PATH = "/home/lauraflyra/Documents/SHK_SprekelerLab/LSTM_computations/LSTM_MemoryTask/src/data/data_checkpoints_1kepochs_all_neurons_separate_linear_additive_random_batch"
DATA_PATH = "/home/lauraflyra/Documents/SHK_SprekelerLab/LSTM_computations/LSTM_MemoryTask/src/data/"
params_file = os.path.join(DATA_PATH, "params_additive.mat")

data = io.loadmat(params_file, struct_as_record=False, squeeze_me = True)

time_steps, total_batch, input_dim = data["x"].shape
random_batch = np.random.randint(0, total_batch, size=10)
x = data['x'][:, random_batch, :]

def plot_result_multiple_checkpoints_training_additive(data, model, epoch, x, random_batch):
    # take random samples from batch

    time_steps, total_batch, input_dim = data["x"].shape

    y = data['output'][:,random_batch,:]
    out_pred, model_variables = model(torch.from_numpy(x).float())

    spacing_y = np.arange(0, len(PEOPLE), 1)
    spacing_x = np.arange(0, input_dim, 1)

    time_array = np.arange(0, time_steps, 1)

    out_pred = out_pred.detach().numpy()
    for i in range(10):
        x_plot = x[:, i, :] + spacing_x
        output_plot = y[:, i, :] + spacing_y
        out_pred_plot = out_pred[:, i, :] + spacing_y

        plt.plot(time_array, x_plot[:,:len(PEOPLE)])
        plt.plot(time_array, x_plot[:, len(PEOPLE):], color='black', label = "time slot")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.title("input training random batch n {}".format(i))
        name = "additive_input_random_batch_"+str(i)+"_checkpoint_1kepochs_"+str(epoch)+".png"
        plt.savefig(os.path.join(SAVE_DATA_PATH, name))
        plt.show()

        plt.plot(time_array, output_plot, color = 'blue', linewidth = 3)
        plt.plot(time_array, out_pred_plot, color ='black', alpha=0.6, label = 'predicted')
        plt.title("output training random batch n {}".format(i))

        plt.xlabel('time points')
        plt.xlabel('time points')
        plt.ylabel('neuron number')

        # black_patch = mpatches.Patch(color='black', label='predicted')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='lower right')
        name = "additive_output_random_batch_"+str(i)+"_checkpoint_1kepochs_"+str(epoch)+".png"
        plt.savefig(os.path.join(SAVE_DATA_PATH, name))
        plt.show()

    return

for epoch in np.arange(0,1200,200):
    model = NetworkAdditive()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    checkpoint = torch.load("/home/lauraflyra/Documents/SHK_SprekelerLab/LSTM_computations/LSTM_MemoryTask/src/data/checkpoint_1kepochs_"+str(epoch)+".pt")

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']

    plot_result_multiple_checkpoints_training_additive(data, model, epoch, x, random_batch)