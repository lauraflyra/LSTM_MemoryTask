"""Pre-train tests

There's some tests that we can run without needing trained parameters. These tests include:

    check the shape of your model output and ensure it aligns with the labels in your dataset
    check the output ranges and ensure it aligns with our expectations (eg. the output of a classification model should be a distribution with class probabilities that sum to 1)
    make sure a single gradient step on a batch of data yields a decrease in your loss
    make assertions about your datasets
    check for label leakage between your training and validation datasets

The main goal here is to identify some errors early so we can avoid a wasted training job."""

import unittest
import numpy as np
from src.data.create_input import Input

def test_lookup_table():
    from src.data.create_input import lookup_table
    from test_helpers import mock_table, mock_people
    table = lookup_table(mock_people)
    assert mock_table == table


def test_people2ix():
    from src.data.create_input import people2ix
    from test_helpers import mock_schedule, mock_int_schedule, mock_table
    int_schedule = people2ix(mock_schedule, mock_table)
    assert (mock_int_schedule == int_schedule).all()


class TestInput(unittest.TestCase):

    def test_init_input_times(self):
        from src.data.create_input import TIMES, TOTAL_TIME_STEPS
        xy = Input(batch_size=1)
        xy.init_input_times()
        self.assertEqual(xy.input[:,0,0].tolist(),np.pad(TIMES, (0,TOTAL_TIME_STEPS-len(TIMES)), 'constant', constant_values=0).tolist())

    def test_combine_people_and_times(self):
        ...

    def test_select_query(self):
        ...


if __name__ == '__main__':
    unittest.main()