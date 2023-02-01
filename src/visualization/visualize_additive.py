from src.visualization.plot_input_output import plot_result_training_additive
from src.visualization.plot_activation_functions import plot_af_additive_neurons
from src.data.common_vars import DATA_PATH
import os

"""
Visualize results of a trained additive model.
"""

PATH_RESULTS_ADDITIVE = os.path.join(DATA_PATH, "additive")
params_file = os.path.join(PATH_RESULTS_ADDITIVE, "params_additive")
which_from_batch = plot_result_training_additive(params_file, which_from_batch=54)
plot_af_additive_neurons(params_file, which_from_batch=50)
