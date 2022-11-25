import numpy as np
import torch

TIMES = np.array([9, 10, 11, 12, 13, 14, 15, 16], dtype=int)  # daily time slots of the Prof.
time_array = np.linspace(0.1,0.9,len(TIMES))    # Dont want to have neither zero nor 1
PEOPLE = ["Laura L.", "Joram", "Dennis",
          "Felix", "Rob", "Friedrich", "Loreen", "Mark", "Laura N."]

TOTAL_TIME_STEPS = 21
INPUT_SIZE = len(PEOPLE) + 1
BATCH_SIZE = 5


# Input has shape (batch size, Total time steps,  input size) #TODO: REMEMBER TO PUT BATCH FIRST
# input size = Number of People + 1 -> one hot encoding of people + time stamp of appointment

# prepare input array

input = np.zeros((BATCH_SIZE, TOTAL_TIME_STEPS,  INPUT_SIZE))
one_hot_people_time = np.zeros((len(PEOPLE),INPUT_SIZE))
one_hot_people_time[:,:-1] = np.eye(len(PEOPLE))
idx_sample_people = np.random.randint(len(PEOPLE), size = BATCH_SIZE*len(TIMES))
input[:,:len(TIMES),:] = one_hot_people_time[idx_sample_people,:].reshape(BATCH_SIZE, len(TIMES),-1)
input[:,:len(TIMES),-1] = time_array

# insert queries
# queries = np.random.choice(time_array, size=BATCH_SIZE, replace=True)
queries_idx = np.random.choice(range(len(TIMES)), size=BATCH_SIZE, replace=True)
queries = time_array[queries_idx]
possible_query_moments = np.arange(len(TIMES)+1, TOTAL_TIME_STEPS, 1)
queries_moments = np.random.choice(possible_query_moments, size=BATCH_SIZE, replace=True)
for i in range(BATCH_SIZE):
    input[i, queries_moments[i], -1] = queries[i]

# create output based on queries
output = np.zeros((BATCH_SIZE, len(PEOPLE)))  # Output is gonna be in one-hot enconding of people
