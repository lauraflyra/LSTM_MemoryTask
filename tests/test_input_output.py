from src.data.create_input import TIMES, PEOPLE, TOTAL_TIME_STEPS, INPUT_SIZE, CUE_TIME

def test_create_input():
    from src.data.create_input import create_input
    test_input = create_input(batch_size=3)
    # x[CUE_TIME,:,:] -> check for the patterns of the one hot encoding
    ...

def test_create_go_signal():
    from src.data.create_input import create_go_signal
    ...









