import numpy as np
import matplotlib.pyplot as plt


L = 85  # length of each sample
DIM = 1  # input size - number of neurons
N = 300  # batch size

CUE_START_TIME = 5
CUE_DURATION = 1
CUE_END_TIME = CUE_START_TIME + CUE_DURATION


GO_DURATION = 1
GO_AMP = 0.6  # amplitude of the go signal


def generate_cue_amp(cue_height=0.8):
    cue_amp = np.random.choice([0, cue_height], size=(N))
    return cue_amp


def gen_input_out(CUE_AMP=generate_cue_amp()):

    GO_START_TIME = np.random.randint(CUE_END_TIME + 1, high=int(L - CUE_DURATION - GO_DURATION), size=N)
    GO_END_TIME = GO_START_TIME + GO_DURATION

    OUTPUT_RESPONSE_START = GO_START_TIME
    OUTPUT_RESPONSE_END = OUTPUT_RESPONSE_START + CUE_DURATION

    which_no_cue = np.where(np.abs(CUE_AMP) < 0.5)[0][0]  # which samples have no cue
    which_wt_cue = np.where(np.abs(CUE_AMP) >= 0.5)[0][0]  # which samples receive a cue

    # INPUT NEEDS TO BE IN SHAPE (time_steps, batch, input_size)

    x = np.zeros((L, N, DIM), dtype=np.float32)  # create the input
    y = np.zeros((L, N, DIM), dtype=np.float32)  # create the expected output
    noise = np.random.normal(0, 0.01, size=(L, N, DIM))

    # insert the cue for all inputs and outputs
    x[CUE_START_TIME:CUE_END_TIME, :, 0] = CUE_AMP
    y[CUE_START_TIME:CUE_END_TIME, :, 0] = CUE_AMP

    # insert go signal for all inputs
    for i in range(N):
        go_start = GO_START_TIME[i]
        go_end = GO_END_TIME[i]
        x[go_start:go_end, i, :] = GO_AMP

    x += noise

    # insert response to go signal for all outputs
    for i in range(N):
        o_start = OUTPUT_RESPONSE_START[i]
        o_end = OUTPUT_RESPONSE_END[i]
        y[o_start:o_end, i, :] = CUE_AMP[i]

    return x, y, which_no_cue, which_wt_cue, CUE_AMP, GO_START_TIME

if __name__ == "__main__":
    x, y, which_no_cue, which_wt_cue, CUE_AMP = gen_input_out()
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    which_plot = np.random.randint(0, N - 1)
    axs[0].plot(x[:, which_plot, 0], label="0", alpha=0.5)
    axs[1].plot(y[:, which_plot, 0], label="0", alpha=0.5)
    axs[0].set_title("Training sample: " + str(which_plot))
    axs[0].legend()
    axs[1].legend()
    axs[0].legend()
    axs[1].legend()
    axs[1].set_ylim(axs[0].get_ylim())

    plt.show()
