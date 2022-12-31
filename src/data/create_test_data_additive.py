import numpy as np
from src.data.common_vars import PEOPLE, TIME_SLOTS
from src.data.create_input_output_additive import TIME_STEPS, INPUT_DIM, CUE_START_TIME, CUE_END_TIME, one_hot_times, one_hot_people, create_output_add

batch_size = 3

def create_test_input_and_go_add(batch_size = batch_size):
    # prepare input array
    x = np.zeros((TIME_STEPS, batch_size, INPUT_DIM))
    for i in range(batch_size):
        time_slot_idx = np.random.choice(len(TIME_SLOTS), len(TIME_SLOTS), replace=False)
        people_idx = np.random.choice(len(PEOPLE), len(TIME_SLOTS), replace=True)
        cue = np.hstack((one_hot_people[people_idx], one_hot_times[time_slot_idx]))
        x[CUE_START_TIME:CUE_END_TIME, i, :] = cue


    # generate go signal
    go_signal_moments = [CUE_END_TIME+2, TIME_STEPS-2, int((CUE_END_TIME+TIME_STEPS)/2)]
    go_signal_time_slots = [0,1,2]

    for i in range(batch_size):
        x[go_signal_moments[i],i,-len(TIME_SLOTS):] = one_hot_times[go_signal_time_slots[i]]

    return x, go_signal_time_slots, go_signal_moments

x_test, go_signal_time_slots, go_signal_moments = create_test_input_and_go_add()
y_test = create_output_add(x_test, go_signal_time_slots, go_signal_moments)