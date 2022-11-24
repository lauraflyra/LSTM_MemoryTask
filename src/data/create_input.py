import numpy as np

TIMES = np.array([9, 10, 11, 12, 13, 14, 15, 16], dtype=int)  # daily time slots of the Prof.
PEOPLE = ["Laura L.", "Joram", "Dennis",
          "Felix", "Rob", "Friedrich", "Loreen", "Mark", "Laura N."]

TOTAL_TIME_STEPS = 21
INPUT_SIZE = 2  # one neuron always receives an input with 2 dimensions: time slots and people

def lookup_table(people):
    """
    Creates a look up table, in which each different name is associated with
    a different integer
    :param people: list of all the people in the lab
    :return: dictionary of people an the associated integer
    """
    table = dict()
    for person in people:
        if person not in table:
            table[person] = len(table) + 1  # It doesn't make sense to have entry 0
    return table


def people2ix(schedule, lookup):
    """
    Encode the schedule using the lookup table
    :param schedule: list of peoples names, in the order correspondent to the slots in the profs. times
    :param lookup: dictionary, lookup table for people to integer
    :return: numpy array of integers, correspondent to the schedule
    """
    return np.array([lookup[person] for person in schedule], dtype=int)


class Input:
    def __init__(self, batch_size=200, people = PEOPLE):
        self.batch_size = batch_size
        self.input = np.zeros((TOTAL_TIME_STEPS, self.batch_size, INPUT_SIZE))
        self.people = people
        self.dict_people = lookup_table(self.people)
        self.schedules = []

    def init_input_times(self):
        self.input[0:len(TIMES), :, 0] = np.stack((TIMES,) * self.batch_size, axis=-1)

    def combine_people_and_times(self):
        p2ix = people2ix(self.people, self.dict_people)
        self.schedules = np.random.choice(p2ix, size=(len(TIMES), self.batch_size), replace=True)
        self.input[0:len(TIMES), :, 1] = self.schedules


    def select_query(self):
        queries = np.random.choice(TIMES, size=self.batch_size, replace=True)
        possible_query_times = np.arange(len(TIMES)+1, TOTAL_TIME_STEPS, 1)
        queries_times = np.random.choice(possible_query_times, size=self.batch_size, replace=True)
        for i in range(self.batch_size):
            self.input[queries_times[i], i, 0] = queries[i]

    def finalize_input(self):
        self.init_input_times()
        self.combine_people_and_times()
        self.select_query()


if __name__ == "__main__":
    xy_input = Input()
    xy_input.finalize_input()

