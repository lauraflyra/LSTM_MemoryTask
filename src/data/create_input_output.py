import numpy as np

PEOPLE = ["Laura", "Joram", "Denis", "Felix"]
TIME_SLOTS = [9,10,11]

INPUT_DIM = len(PEOPLE)+len(TIME_SLOTS) # We will have one hot encoding of people and one hot encoding of times

one_hot_people = np.eye(len(PEOPLE))
one_hot_times = np.eye(len(TIME_SLOTS))

# input time series can be such that this time the "cue_signal" actually lasts for 3 time steps, i.e,
# cue for 9 am, then cue for 10 am, than cue for 11 am. TODO: question, should I always have them in order? Or can I present a "mixed schedule, eg 10,11,9?"
# INPUT NEEDS TO BE IN SHAPE (time_steps, batch, input_size)

TIME_STEPS = 2 + len(TIME_SLOTS) + 7  # Starts with 2 time points zero, than send the cues, then let the go signal come whenever
BATCH_SIZE = 3

x = np.zeros((TIME_STEPS, BATCH_SIZE, INPUT_DIM))
y = np.zeros_like(x)

# generate cue
CUE_START_TIME = 2
CUE_END_TIME = CUE_START_TIME + len(TIME_SLOTS)

# time_slot_idx = np.random.choice(len(TIME_SLOTS),len(TIME_SLOTS),replace=False)
# people_idx = np.random.choice(len(PEOPLE), len(TIME_SLOTS), replace=True)
#
# cue = np.hstack((one_hot_people[people_idx],one_hot_times[time_slot_idx]))  # TODO: make this for several batches!

for i in range(BATCH_SIZE):
    time_slot_idx = np.random.choice(len(TIME_SLOTS), len(TIME_SLOTS), replace=False)
    people_idx = np.random.choice(len(PEOPLE), len(TIME_SLOTS), replace=True)
    cue = np.hstack((one_hot_people[people_idx], one_hot_times[time_slot_idx]))
    x[CUE_START_TIME:CUE_END_TIME, i, :] = cue


# generate go signal
go_signal_moments = np.random.choice(np.arange(CUE_END_TIME+1, TIME_STEPS, 1), BATCH_SIZE, replace=True)
go_signal_time_slots = np.random.choice(len(TIME_SLOTS), BATCH_SIZE, replace=True)

for i in range(BATCH_SIZE):
    x[go_signal_moments[i],i,-len(TIME_SLOTS):] = one_hot_times[go_signal_time_slots[i]]

# TODO: create output. Time series???









