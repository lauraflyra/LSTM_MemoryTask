import numpy as np
from src.data.common_vars import PEOPLE, TIME_SLOTS

TOTAL_TIME_STEPS = 20
INPUT_SIZE = len(PEOPLE) * len(TIME_SLOTS)
BATCH_SIZE = 700
CUE_TIME = 2


# Input has shape ((TOTAL_TIME_STEPS, BATCH_SIZE, INPUT_SIZE))
# input size = Number of People * Number of time slots -> one hot encoding of people for every time slot
# we also have number of neurons = Number of People * Number of time slots


def create_input(batch_size=BATCH_SIZE):
    # prepare input array
    x = np.zeros((TOTAL_TIME_STEPS, batch_size, INPUT_SIZE))
    one_hot_people = 0.8 * np.eye(len(PEOPLE))
    idx_sample_people = np.random.randint(len(PEOPLE), size=batch_size * len(TIME_SLOTS))
    one_hot_people = one_hot_people[idx_sample_people, :].reshape(batch_size, -1)
    x[CUE_TIME, :, :] = one_hot_people
    return x


def create_go_signal(x):
    batch_size = x.shape[1]
    # insert go signal
    # first we decide when the go signal is gonna happen
    possible_go_signal_moments = np.arange(CUE_TIME + 2, TOTAL_TIME_STEPS - 1, 1)
    go_signal_moments = np.random.choice(possible_go_signal_moments, size=batch_size, replace=True)
    # then we decide which time slot we want to recover
    go_signal_time_slots = np.random.choice(range(len(TIME_SLOTS)), size=batch_size, replace=True)
    for i in range(batch_size):
        x[go_signal_moments[i], i, go_signal_time_slots[i] * 4:go_signal_time_slots[i] * 4 + 4] = 0.5

    for i in range(batch_size):
        assert np.sum(x[go_signal_moments[i], i, :]) == 4 * 0.5
    return x, go_signal_time_slots, go_signal_moments


def create_output(x, go_signal_time_slots, go_signal_moments):
    # create output based on queries
    batch_size = x.shape[1]
    output = np.zeros((TOTAL_TIME_STEPS, batch_size, INPUT_SIZE))  # Output is gonna be in one-hot encoding of people
    # insert response to cue
    output[CUE_TIME, :, :] = x[CUE_TIME, :, :]
    # insert response to go signal
    for batch_item, i in enumerate(go_signal_time_slots):
        output[go_signal_moments[batch_item], batch_item, i * 4:i * 4 + 4] = x[CUE_TIME, batch_item, i * 4:i * 4 + 4]
    return output


def gen_input_output(batch_size):
    x = create_input(batch_size=batch_size)
    x, go_signal_time_slots, go_signal_moments = create_go_signal(x)
    output = create_output(x, go_signal_time_slots, go_signal_moments)
    return x, output, go_signal_time_slots, go_signal_moments



