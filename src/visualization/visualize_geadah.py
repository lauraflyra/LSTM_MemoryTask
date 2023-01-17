from src.visualization.plot_input_output import plot_result_training_one_neuron
from src.visualization.plot_activation_functions import plot_params_af_in_time_one_neurons_Geadah
from src.data.common_vars import DATA_PATH
import os

PATH_RESULTS_ONE_NEURON = os.path.join(DATA_PATH, "one_neuron")
params_file = os.path.join(PATH_RESULTS_ONE_NEURON, "params_one_neuron_Geadah.mat")
plot_result_training_one_neuron(params_file)
plot_params_af_in_time_one_neurons_Geadah(params_file)