import torch
from src.models.train import train
from src.models.network_one_neuron import NetworkOneNeuron
from src.data.create_input_output_one_neuron import gen_input_output
from src.data.common_vars import DATA_PATH
import os

PATH_RESULTS_ONE_NEURON = os.path.join(DATA_PATH, "one_neuron")

x, y, CUE_AMP, GO_START_TIME = gen_input_output(batch_size=100)
dataset = (torch.from_numpy(x), torch.from_numpy(y), CUE_AMP, GO_START_TIME)
model = NetworkOneNeuron()
dict_results_one_neuron = train(dataset,
                                model,
                                n_epochs=110,
                                data_path=PATH_RESULTS_ONE_NEURON,
                                save_params_name=os.path.join(PATH_RESULTS_ONE_NEURON, "params_one_neuron.mat"),
                                save_model_name=os.path.join(PATH_RESULTS_ONE_NEURON, "model_one_neuron.pt"),
                                checkpoints_file_name="model_checkpoints_one_neuron")

#
# if __name__ == "__main__":
#     x, y, which_no_cue, which_wt_cue, CUE_AMP, GO_START_TIME = gen_input_out()
#     dataset = (torch.from_numpy(x), torch.from_numpy(y))
#     model = NetworkOneNeuron()
#     train(dataset, model, n_epochs=110, CUE_AMP=CUE_AMP, GO_START_TIME=GO_START_TIME,
#           saveNonLinearitiesName="save_non_linearities_one_neuron.mat",
#           saveModelName="model_one_neuron.pt", plot_every=100, log_every=100)
#
#     # y_pred = y_pred.detach().numpy()
#     # fig, axs = plt.subplots(2, 2, figsize=(15, 15))
#     # axs[0,0].plot(x[:, which_plot_no_bump, 0], label="x0")
#     # axs[0,0].set_title("No bump"+str(which_plot_no_bump))
#     # axs[0,0].set_xlabel("Time")
#     # axs[0,0].legend()
#     # #
#     # axs[0,0].plot(y[:, which_plot_no_bump, 0], label="target y0")
#     # #
#     #
#     # axs[0,0].plot(y_pred[:, which_plot_no_bump, 0], label="output y0")
#     # axs[0,0].legend()
#     #
#     # axs[1,0].plot(np.arange(0, n_epochs, plot_every), train_error, label="train error")
#     # axs[1, 0].set_xlabel("Epochs")
#     # axs[1, 0].set_ylabel("MSE")
#     # axs[1,0].legend()
#     #
#     # axs[0,1].plot(x[:, which_plot_wt_bump, 0], label="x0")
#     # axs[0, 1].set_title("With bump"+str(which_plot_wt_bump))
#     # axs[0, 1].set_xlabel("Time")
#     # axs[0,1].legend()
#     # #
#     # axs[0,1].plot(y[:, which_plot_wt_bump, 0], label="target y0")
#     # #
#     #
#     # axs[0,1].plot(y_pred[:, which_plot_wt_bump, 0], label="output y0")
#     # axs[0,1].legend()
#     #
#     # axs[1,1].plot(np.arange(0, n_epochs, plot_every), train_error, label="train error")
#     # axs[1, 1].set_xlabel("Epochs")
#     # axs[1, 1].set_ylabel("MSE")
#     # axs[1,1].legend()
#     # plt.suptitle("epochs: "+str(n_epochs))
#     # plt.show()
