from src.visualization.plot_input_output import plot_result_training_factorial
from src.visualization.plot_activation_functions import plot_af_separate_neurons_factorial
from src.data.common_vars import DATA_PATH
import os

"""
Visualize results of a trained factorial model.
"""

PATH_RESULTS_FACTORIAL = os.path.join(DATA_PATH, "factorial")
params_file = os.path.join(PATH_RESULTS_FACTORIAL, "params_factorial.mat")
which_from_batch = plot_result_training_factorial(params_file)
plot_af_separate_neurons_factorial(params_file, which_from_batch)