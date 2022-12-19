import numpy as np

TIMES = np.array([9, 10, 11], dtype=int)  # daily time slots of the Prof.
# time_array = np.linspace(0.1, 0.9, len(TIMES))  # Dont want to have neither zero nor 1
PEOPLE = ["Laura", "Joram", "Dennis", "Felix"]

TOTAL_TIME_STEPS = 9
INPUT_SIZE = len(PEOPLE) * len(TIMES)
BATCH_SIZE = 700
CUE_TIME = 2


# Input has shape ((TOTAL_TIME_STEPS, BATCH_SIZE, INPUT_SIZE))
# input size = Number of People * Number of time slots -> one hot encoding of people for every time slot
# we also have number of neurons = Number of People * Number of time slots


def create_input(batch_size=BATCH_SIZE):
    # prepare input array
    x = np.zeros((TOTAL_TIME_STEPS, batch_size, INPUT_SIZE))
    one_hot_people = 0.8 * np.eye(len(PEOPLE))
    idx_sample_people = np.random.randint(len(PEOPLE), size=batch_size * len(TIMES))
    one_hot_people = one_hot_people[idx_sample_people, :].reshape(batch_size, -1)
    x[CUE_TIME, :, :] = one_hot_people
    return x


def create_go_signal(x):
    batch_size = x.shape[1]
    # insert go signal
    # first we decide when the go signal is gonna happen
    possible_go_signal_moments = np.arange(CUE_TIME + 2, TOTAL_TIME_STEPS-1, 1)
    go_signal_moments = np.random.choice(possible_go_signal_moments, size=batch_size, replace=True)
    # then we decide which time slot we want to recover
    go_signal_idx = np.random.choice(range(len(TIMES)), size=batch_size, replace=True)
    for i in range(batch_size):
        x[go_signal_moments[i], i, go_signal_idx[i] * 4:go_signal_idx[i] * 4 + 4] = 0.5

    for i in range(batch_size):
        assert np.sum(x[go_signal_moments[i], i, :]) == 4 * 0.5
    return x, go_signal_idx, go_signal_moments


def create_output(x, go_signal_idx, go_signal_moments):
    # create output based on queries
    batch_size = x.shape[1]
    output = np.zeros((TOTAL_TIME_STEPS, batch_size, INPUT_SIZE))  # Output is gonna be in one-hot encoding of people
    # insert response to cue
    output[CUE_TIME, :, :] = x[CUE_TIME, :, :]
    # insert response to go signal
    for batch_item, i in enumerate(go_signal_idx):
        output[go_signal_moments[batch_item], batch_item, i * 4:i * 4 + 4] = x[CUE_TIME, batch_item, i * 4:i * 4 + 4]
    return output


if __name__ == "__main__":
    x = create_input()
    x, go_signal_idx, go_signal_moments = create_go_signal(x)
    output = create_output(x, go_signal_idx, go_signal_moments)


