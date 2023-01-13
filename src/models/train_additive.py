import torch
from src.models.train import train
from src.models.network_additive import NetworkAdditive
from src.data.create_input_output_additive import gen_input_output
from src.data.common_vars import DATA_PATH
import os

PATH_RESULTS_ADDITIVE = os.path.join(DATA_PATH, "additive")

x, y, go_signal_time_slots, go_signal_moments = gen_input_output(batch_size=100)
dataset = (torch.from_numpy(x).float(), torch.from_numpy(y).float(), go_signal_time_slots, go_signal_moments)
model = NetworkAdditive()
dict_results_additive = train(dataset,
                              model,
                              n_epochs=210,
                              data_path=PATH_RESULTS_ADDITIVE,
                              save_params_name=os.path.join(PATH_RESULTS_ADDITIVE, "params_additive.mat"),
                              save_model_name=os.path.join(PATH_RESULTS_ADDITIVE, "model_additive.pt"),
                              checkpoints_file_name="model_checkpoints_factorial")


from src.visualization.plot_input_output import plot_result_training_additive

which_from_batch = plot_result_training_additive(params_file=os.path.join(PATH_RESULTS_ADDITIVE, "params_additive.mat"))

from src.visualization.plot_activation_functions import plot_af_additive_neurons

plot_af_additive_neurons(params_file=os.path.join(PATH_RESULTS_ADDITIVE, "params_additive.mat"),
                         which_from_batch=which_from_batch)

# For plotting the results of the training for more than one sample from the batch,
# use function plot_result_multiple_training_additive!