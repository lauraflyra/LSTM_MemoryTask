import numpy as np
import matplotlib.pyplot as plt


L = 85  # length of each sample
DIM = 1  # input size - number of neurons
BATCH_SIZE = 100

CUE_START_TIME = 5
CUE_DURATION = 1
CUE_END_TIME = CUE_START_TIME + CUE_DURATION
CUE_HEIGHT = 0.8

GO_DURATION = 1
GO_AMP = 0.6  # amplitude of the go signal


def generate_cue_amp(batch_size = BATCH_SIZE):
    cue_amp = np.random.choice([0, CUE_HEIGHT], size=(batch_size))
    return cue_amp


def gen_input_output(batch_size = BATCH_SIZE):
    CUE_AMP = generate_cue_amp(batch_size= batch_size)

    GO_START_TIME = np.random.randint(CUE_END_TIME + 1, high=int(L - CUE_DURATION - GO_DURATION), size=batch_size)
    GO_END_TIME = GO_START_TIME + GO_DURATION

    OUTPUT_RESPONSE_START = GO_START_TIME
    OUTPUT_RESPONSE_END = OUTPUT_RESPONSE_START + CUE_DURATION

    # INPUT NEEDS TO BE IN SHAPE (time_steps, batch, input_size)

    x = np.zeros((L, batch_size, DIM), dtype=np.float32)  # create the input
    y = np.zeros((L, batch_size, DIM), dtype=np.float32)  # create the expected output
    noise = np.random.normal(0, 0.01, size=(L, batch_size, DIM))

    # insert the cue for all inputs and outputs
    x[CUE_START_TIME:CUE_END_TIME, :, 0] = CUE_AMP
    y[CUE_START_TIME:CUE_END_TIME, :, 0] = CUE_AMP

    # insert go signal for all inputs
    for i in range(batch_size):
        go_start = GO_START_TIME[i]
        go_end = GO_END_TIME[i]
        x[go_start:go_end, i, :] = GO_AMP

    x += noise

    # insert response to go signal for all outputs
    for i in range(batch_size):
        o_start = OUTPUT_RESPONSE_START[i]
        o_end = OUTPUT_RESPONSE_END[i]
        y[o_start:o_end, i, :] = CUE_AMP[i]

    return x, y, CUE_AMP, GO_START_TIME

if __name__ == "__main__":
    x, y, CUE_AMP, GO_START_TIME = gen_input_output()
    fig, axs = plt.subplots(2, 2, figsize=(15, 5))
    which_plot = np.random.randint(0, BATCH_SIZE - 1)
    # which_val_plot = np.random.randint(0, VAL_BATCH_SIZE - 1)
    which_plot_no_bump = np.where(np.abs(CUE_AMP) < 0.5)[0][0]
    which_plot_wt_bump = np.where(np.abs(CUE_AMP) >= 0.5)[0][0]

    axs[0,0].plot(x[:, which_plot_wt_bump, 0], label="0", alpha=0.5, linewidth = 4)
    axs[1,0].plot(y[:, which_plot_wt_bump, 0], label="0", alpha=0.5, linewidth = 4)
    axs[0, 0].set_title("Input: cue + go signal", fontsize = 15)
    axs[1, 0].set_title("Desired output", fontsize = 15)
    axs[0,0].set_xlabel("Time steps", fontsize = 15)
    axs[0, 0].set_ylabel("Input strength", fontsize = 15)
    axs[1,0].set_xlabel("Time steps", fontsize = 15)
    axs[1, 0].set_ylabel("Neural response", fontsize = 15)
    # axs[0,1].legend()

    axs[0,1].plot(x[:, which_plot_no_bump, 0], label="0", alpha=0.5, linewidth = 4, color = 'black')
    axs[1,1].plot(y[:, which_plot_no_bump, 0], label="0", alpha=0.5, linewidth = 4, color = 'black')
    axs[0, 1].set_title("Input: only go", fontsize = 15)
    axs[1, 1].set_title("Desired output", fontsize = 15)
    axs[0,1].set_xlabel("Time steps", fontsize = 15)
    axs[0, 1].set_ylabel("Input strength", fontsize = 15)
    axs[1,1].set_xlabel("Time steps", fontsize = 15)
    axs[1, 1].set_ylabel("Neural response", fontsize = 15)

    axs[1,1].set_ylim(axs[0,0].get_ylim())
    axs[1,0].set_ylim(axs[0,0].get_ylim())
    axs[0,0].spines[['right', 'top']].set_visible(False)
    axs[0, 1].spines[['right', 'top']].set_visible(False)
    axs[1, 0].spines[['right', 'top']].set_visible(False)
    axs[1, 1].spines[['right', 'top']].set_visible(False)

    axs[0,0].spines[['bottom', 'left']].set_linewidth(3)
    axs[0, 1].spines[['bottom', 'left']].set_linewidth(3)
    axs[1, 0].spines[['bottom', 'left']].set_linewidth(3)
    axs[1, 1].spines[['bottom', 'left']].set_linewidth(3)

    axs[0,0].tick_params(width=3)
    axs[0, 1].tick_params(width=3)
    axs[1, 0].tick_params(width=3)
    axs[1, 1].tick_params(width=3)


    plt.tight_layout()
    plt.show()
