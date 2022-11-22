"""Pre-train tests

There's some tests that we can run without needing trained parameters. These tests include:

    check the shape of your model output and ensure it aligns with the labels in your dataset
    check the output ranges and ensure it aligns with our expectations (eg. the output of a classification model should be a distribution with class probabilities that sum to 1)
    make sure a single gradient step on a batch of data yields a decrease in your loss
    make assertions about your datasets
    check for label leakage between your training and validation datasets

The main goal here is to identify some errors early so we can avoid a wasted training job."""


def test_lookup_table():
    from src.data.create_input import lookup_table
    from test_helpers import mock_table, mock_people
    table = lookup_table(mock_people)
    assert mock_table == table


def test_people2ix():
    from src.data.create_input import people2ix
    from test_helpers import mock_schedule, mock_int_schedule, mock_table
    int_schedule = people2ix(mock_schedule, mock_table)
    assert mock_int_schedule == int_schedule



#class TestDataset(unittest.TestCase):
