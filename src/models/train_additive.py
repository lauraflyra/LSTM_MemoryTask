import torch
from src.models.train import train
from src.models.network_additive import NetworkAdditive
from src.data.create_input_output_additive import gen_input_output
from src.data.common_vars import DATA_PATH
import os

PATH_RESULTS_ADDITIVE = os.path.join(DATA_PATH, "additive")

x, y, go_signal_time_slots, go_signal_moments = gen_input_output(batch_size=200)
dataset = (torch.from_numpy(x).float(), torch.from_numpy(y).float(), go_signal_time_slots, go_signal_moments)
model = NetworkAdditive()
dict_results_additive = train(dataset,
                              model,
                              n_epochs=1510,
                              data_path=PATH_RESULTS_ADDITIVE,
                              save_params_name=os.path.join(PATH_RESULTS_ADDITIVE, "params_additive_all_neurons_separate_linear.mat"),
                              save_model_name=os.path.join(PATH_RESULTS_ADDITIVE, "model_additive_all_neurons_separate_linear.pt"),
                              checkpoints_file_name="model_checkpoints_additive_all_neurons_separate_linear")

