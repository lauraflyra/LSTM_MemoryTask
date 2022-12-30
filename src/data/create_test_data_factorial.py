import numpy as np
from src.data.common_vars import PEOPLE, TIME_SLOTS
from src.data.create_input_output_factorial import TOTAL_TIME_STEPS, INPUT_SIZE, CUE_TIME, create_input, create_output

batch_size = 3
x_test = create_input(batch_size=batch_size)


def create_test_go_signal(x):
    batch_size = x.shape[1]
    # insert go signal
    # first we decide when the go signal is gonna happen
    go_signal_moments = [CUE_TIME + 2, TOTAL_TIME_STEPS - 2, int((CUE_TIME + TOTAL_TIME_STEPS) / 2)]
    # then we decide which time slot we want to recover
    go_signal_idx = [0, 1, 2]
    for i in range(batch_size):
        x[go_signal_moments[i], i, go_signal_idx[i] * 4:go_signal_idx[i] * 4 + 4] = 0.5
    for i in range(batch_size):
        assert np.sum(x[go_signal_moments[i], i, :]) == 4 * 0.5
    return x, go_signal_idx, go_signal_moments


x_test, go_signal_idx_test, go_signal_moments_test = create_test_go_signal(x_test)
y_test = create_output(x_test, go_signal_idx_test, go_signal_moments_test)
