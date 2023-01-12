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

#
# if __name__ == "__main__":
#     x, go_signal_time_slots, go_signal_moments = create_input_go_add(batch_size=500)
#     y = create_output_add(x, go_signal_time_slots, go_signal_moments)
#     model = NetworkAdditive()
#     dataset = (torch.from_numpy(x).float(), torch.from_numpy(y).float())
#     x, output, out_pred, train_error = train(dataset, model, n_epochs=1010,
#                         saveParamsName=os.path.join(DATA_PATH, "params_model_additive_more_dimensions_neurons.mat"),
#                         saveModelName=os.path.join(DATA_PATH, "model_additive_more_dimensions_neurons.pt"))
#     which_from_batch = plot_result_training_additive(x, output, out_pred.detach().numpy(), train_error, n_epochs=1010,
#                                                       plot_every=100)
#
#     plot_AF_separate_neurons_add(params_file=os.path.join(DATA_PATH, "params_model_additive_more_dimensions_neurons.mat"), go_signal_time_slots=go_signal_time_slots,
#             go_signal_moments=go_signal_moments, which_from_batch=which_from_batch)
#
#     # TODO: see if there are differences for longer, shorter cue times
#     # TODO: sample more from the results
#     # TODO: how to save model over the course of learning? How is learning?
