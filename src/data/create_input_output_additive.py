import numpy as np
from src.data.common_vars import PEOPLE, TIME_SLOTS

INPUT_DIM = len(PEOPLE)+len(TIME_SLOTS) # We will have one hot encoding of people and one hot encoding of times

one_hot_people = np.eye(len(PEOPLE))
one_hot_times = np.eye(len(TIME_SLOTS))

# input time series can be such that this time the "cue_signal" actually lasts for 3 time steps, i.e,
# cue for 9 am, then cue for 10 am, than cue for 11 am.
# INPUT NEEDS TO BE IN SHAPE (time_steps, batch, input_size)

TIME_STEPS = 2 + len(TIME_SLOTS) + 15  # Starts with 2 time points zero, than send the cues, then let the go signal come whenever
BATCH_SIZE = 300
# generate cue
CUE_START_TIME = 2
CUE_END_TIME = CUE_START_TIME + len(TIME_SLOTS)

def create_input_go_add(batch_size=BATCH_SIZE):
    # prepare input array
    x = np.zeros((TIME_STEPS, batch_size, INPUT_DIM))
    for i in range(batch_size):
        time_slot_idx = np.arange(0,len(TIME_SLOTS),1)
        people_idx = np.random.choice(len(PEOPLE), len(TIME_SLOTS), replace=True)
        cue = np.hstack((one_hot_people[people_idx], one_hot_times[time_slot_idx]))
        x[CUE_START_TIME:CUE_END_TIME, i, :] = cue


    # generate go signal
    go_signal_moments = np.random.choice(np.arange(CUE_END_TIME+1, TIME_STEPS-1, 1), batch_size, replace=True)
    go_signal_time_slots = np.random.choice(len(TIME_SLOTS), batch_size, replace=True)

    for i in range(batch_size):
        x[go_signal_moments[i],i,-len(TIME_SLOTS):] = one_hot_times[go_signal_time_slots[i]]

    return x, go_signal_time_slots, go_signal_moments


def create_output_add(x, go_signal_time_slots, go_signal_moments):
    # generate output
    # response to the cue
    batch_size = x.shape[1]
    y = np.zeros((TIME_STEPS, batch_size, len(PEOPLE)))  # Output is one hot encoding of people
    y[CUE_START_TIME:CUE_END_TIME, :,:] = x[CUE_START_TIME:CUE_END_TIME, :, :len(PEOPLE)]
    for i in range(batch_size):
        selected_time_slot_one_hot  = one_hot_times[go_signal_time_slots[i]]
        x_time_columns = x[:,i,-len(TIME_SLOTS):]
        idx_row_x = np.where(np.sum(np.array([np.equal(row, selected_time_slot_one_hot) for row in x_time_columns]), axis=1) == len(TIME_SLOTS))[0][0]
        y[go_signal_moments[i], i, :] = x[idx_row_x, i, :len(PEOPLE)]

    return y


# if __name__ == "main":




