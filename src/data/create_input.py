import numpy as np

TIMES = np.array([9, 10, 11], dtype=int)  # daily time slots of the Prof.
# time_array = np.linspace(0.1, 0.9, len(TIMES))  # Dont want to have neither zero nor 1
PEOPLE = ["Laura", "Joram", "Dennis", "Felix"]

TOTAL_TIME_STEPS = 7
INPUT_SIZE = len(PEOPLE) * len(TIMES)
BATCH_SIZE = 500

CUE_TIME = 2

# Input has shape ((TOTAL_TIME_STEPS, BATCH_SIZE, INPUT_SIZE))
# input size = Number of People * Number of time slots -> one hot encoding of people for every time slot
# we also have number of neurons = Number of People * Number of time slots

# prepare input array

x = np.zeros((TOTAL_TIME_STEPS, BATCH_SIZE, INPUT_SIZE))
one_hot_people = np.eye(len(PEOPLE))
idx_sample_people = np.random.randint(len(PEOPLE), size=BATCH_SIZE * len(TIMES))
one_hot_people = one_hot_people[idx_sample_people, :].reshape(BATCH_SIZE, -1)
x[CUE_TIME, :, :] = one_hot_people

# insert go signal
# first we decide when the go signal is gonna happen
possible_go_signal_moments = np.arange(CUE_TIME + 1, TOTAL_TIME_STEPS - 1, 1)
go_signal_moments = np.random.choice(possible_go_signal_moments, size=BATCH_SIZE, replace=True)
# then we decide which time slot we want to recover
go_signal_idx = np.random.choice(range(len(TIMES)), size=BATCH_SIZE, replace=True)
for i in range(BATCH_SIZE):
    x[go_signal_moments[i], i , go_signal_idx[i] * 4:go_signal_idx[i] * 4 + 4] = 0.5

# create output based on queries
output = np.zeros((TOTAL_TIME_STEPS, BATCH_SIZE, INPUT_SIZE))  # Output is gonna be in one-hot encoding of people
for batch_item, i in enumerate(go_signal_idx):
    output[go_signal_moments[batch_item], batch_item, i * 4:i * 4 + 4] = x[CUE_TIME, batch_item, i * 4:i * 4 + 4]

