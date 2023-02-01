import torch
from src.models.train import train
from src.models.network_factorial import NetworkFactorial
from src.data.create_input_output_factorial import gen_input_output
from src.data.common_vars import DATA_PATH
import os

PATH_RESULTS_FACTORIAL = os.path.join(DATA_PATH, "factorial")

x, y, go_signal_time_slots, go_signal_moments = gen_input_output(batch_size=100)
dataset = (torch.from_numpy(x).float(), torch.from_numpy(y).float(), go_signal_time_slots, go_signal_moments)
model = NetworkFactorial()
dict_results_factorial = train(dataset,
                               model,
                               n_epochs=1510,
                               data_path=PATH_RESULTS_FACTORIAL,
                               save_params_name=os.path.join(PATH_RESULTS_FACTORIAL, "params_factorial.mat"),
                               save_model_name=os.path.join(PATH_RESULTS_FACTORIAL, "model_factorial.pt"),
                               checkpoints_file_name="model_checkpoints_factorial")

