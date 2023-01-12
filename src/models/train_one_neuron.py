import torch
import torch.nn as nn
from src.models.network_one_neuron import NetworkOneNeuron
from src.data.create_input_output_one_neuron import gen_input_out
from scipy import io

DATA_PATH = "/home/lauraflyra/Documents/SHK_SprekelerLab/LSTM_computations/LSTM_MemoryTask/src/data/"


def train(dataset, model, n_epochs, CUE_AMP, GO_START_TIME,
          saveNonLinearitiesName="save_non_linearities.mat",
          saveModelName="model.pt", plot_every=100, log_every=100):  #
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x, y = dataset

    train_error = []
    store_non_linearity = {"g": [], "x": [], "y": [],
                           "CUE_START_TIME": [], "CUE_END_TIME": [],
                           "OUTPUT_RESPONSE_START": [], "OUTPUT_RESPONSE_END": [], "training_error": [],
                           "ZERO_AMP": []}
    store_non_linearity["x"].append(x.detach().numpy())
    store_non_linearity["y"].append(y.detach().numpy())
    store_non_linearity["GO_START_TIME"] = GO_START_TIME
    store_non_linearity["CUE_AMP"] = CUE_AMP

    for epoch in range(n_epochs):
        y_pred, g = model(x)

        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % plot_every == 0:
            print("training:", epoch, loss.item())
            train_error.append(loss.item())

        if epoch % log_every == 0:
            store_non_linearity["g"].append(g.detach().numpy())
            store_non_linearity["training_error"].append(loss.item())

    io.savemat(saveNonLinearitiesName, store_non_linearity)

    torch.save(model, saveModelName)

    return store_non_linearity


if __name__ == "__main__":
    x, y, which_no_cue, which_wt_cue, CUE_AMP, GO_START_TIME = gen_input_out()
    dataset = (torch.from_numpy(x), torch.from_numpy(y))
    model = NetworkOneNeuron()
    train(dataset, model, n_epochs=110, CUE_AMP=CUE_AMP, GO_START_TIME=GO_START_TIME,
          saveNonLinearitiesName="save_non_linearities_one_neuron.mat",
          saveModelName="model_one_neuron.pt", plot_every=100, log_every=100)

    # y_pred = y_pred.detach().numpy()
    # fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    # axs[0,0].plot(x[:, which_plot_no_bump, 0], label="x0")
    # axs[0,0].set_title("No bump"+str(which_plot_no_bump))
    # axs[0,0].set_xlabel("Time")
    # axs[0,0].legend()
    # #
    # axs[0,0].plot(y[:, which_plot_no_bump, 0], label="target y0")
    # #
    #
    # axs[0,0].plot(y_pred[:, which_plot_no_bump, 0], label="output y0")
    # axs[0,0].legend()
    #
    # axs[1,0].plot(np.arange(0, n_epochs, plot_every), train_error, label="train error")
    # axs[1, 0].set_xlabel("Epochs")
    # axs[1, 0].set_ylabel("MSE")
    # axs[1,0].legend()
    #
    # axs[0,1].plot(x[:, which_plot_wt_bump, 0], label="x0")
    # axs[0, 1].set_title("With bump"+str(which_plot_wt_bump))
    # axs[0, 1].set_xlabel("Time")
    # axs[0,1].legend()
    # #
    # axs[0,1].plot(y[:, which_plot_wt_bump, 0], label="target y0")
    # #
    #
    # axs[0,1].plot(y_pred[:, which_plot_wt_bump, 0], label="output y0")
    # axs[0,1].legend()
    #
    # axs[1,1].plot(np.arange(0, n_epochs, plot_every), train_error, label="train error")
    # axs[1, 1].set_xlabel("Epochs")
    # axs[1, 1].set_ylabel("MSE")
    # axs[1,1].legend()
    # plt.suptitle("epochs: "+str(n_epochs))
    # plt.show()
