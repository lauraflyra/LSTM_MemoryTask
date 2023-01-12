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
                               n_epochs=110,
                               data_path=PATH_RESULTS_FACTORIAL,
                               save_params_name=os.path.join(PATH_RESULTS_FACTORIAL, "params_factorial.mat"),
                               save_model_name=os.path.join(PATH_RESULTS_FACTORIAL, "model_factorial.pt"),
                               checkpoints_file_name="model_checkpoints_factorial")

#
# if __name__ == "__main__":
#     x = create_input(batch_size=100)
#     x, go_signal_idx, go_signal_moments = create_go_signal(x)
#     output = create_output(x, go_signal_idx, go_signal_moments)
#
#     dataset = (torch.from_numpy(x).float(), torch.from_numpy(output).float())
#     model = NetworkFactorial()
#     x, output, out_pred, train_error = train(dataset, model, n_epochs=610, saveParams=True,
#                 saveParamsName=os.path.join(DATA_PATH, "params_factorial_all_neurons_same_linear.mat"))
#
#     which_from_batch = plot_result_training_factorial(x, output, out_pred.detach().numpy(), train_error, n_epochs = 610, plot_every=100,
#                                                             title="Batch size = 100, n_epochs = 610")
#
#     plot_AF_separate_neurons(params_file=os.path.join(DATA_PATH, "params_factorial_all_neurons_same_linear.mat"), go_signal_idx=go_signal_idx,
#             go_signal_moments=go_signal_moments, which_from_batch=which_from_batch)
