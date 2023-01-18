import torch
from src.models.train import train
from src.models.network_one_neuron import NetworkOneNeuron
from src.data.create_input_output_one_neuron import gen_input_output
from src.data.common_vars import DATA_PATH
import os

PATH_RESULTS_ONE_NEURON = os.path.join(DATA_PATH, "one_neuron")

x, y, CUE_AMP, GO_START_TIME = gen_input_output()
dataset = (torch.from_numpy(x), torch.from_numpy(y), CUE_AMP, GO_START_TIME)
model = NetworkOneNeuron()
dict_results_one_neuron = train(dataset,
                                model,
                                n_epochs=1000,
                                data_path=PATH_RESULTS_ONE_NEURON,
                                save_params_name=os.path.join(PATH_RESULTS_ONE_NEURON, "params_one_neuron.mat"),
                                save_model_name=os.path.join(PATH_RESULTS_ONE_NEURON, "model_one_neuron.pt"),
                                checkpoints_file_name="model_checkpoints_one_neuron")

