from src.data.create_input import create_input, create_output, create_go_signal
from src.models.network import Network
import torch
import torch.nn as nn
import scipy.io
import os

DATA_PATH = "/home/lauraflyra/Documents/SHK_SprekelerLab/LSTM_computations/LSTM_MemoryTask/src/data/"


def train(dataset, model, n_epochs, saveParams=True, saveModel=True, outputTraining=True,
          saveParamsName=os.path.join(DATA_PATH, "params.mat"),
          saveModelName=os.path.join(DATA_PATH, "model.pt"), saveLossEvery=100, saveParamsEvery=100):
    """


    :param dataset:
    :param model:
    :param n_epochs:
    :param saveParams:
    :param saveModel:
    :param outputTraining:
    :param saveParamsName:
    :param saveModelName:
    :param saveLossEvery:
    :param saveParamsEvery:
    :return:
    """

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    x, output = dataset  # TODO: add more to the dataset if needed, like cue times or go signal info

    train_error = []

    store_non_linearity = {"x": [], "output": [], "training_error": []}
    if saveParams:
        store_non_linearity["x"].append(x.detach().numpy())
        store_non_linearity["output"].append(output.detach().numpy())

    for epoch in range(n_epochs):
        out_pred, model_neurons = model(x)

        loss = loss_fn(out_pred, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % saveLossEvery == 0 and outputTraining:
            print("training:", epoch, loss.item())
            store_non_linearity["training_error"].append(loss.item())
            train_error.append(loss.item())

        if epoch % saveParamsEvery == 0 and saveParams:
            if epoch == 0:
                for count, neuron in enumerate(model_neurons.keys()):
                    store_non_linearity[neuron + "-g"] = []
            for count, neuron in enumerate(model_neurons.keys()):
                store_non_linearity[neuron + "-g"].append(torch.stack(model_neurons[neuron]['g'][1:]).detach().numpy())

    if saveParams:
        scipy.io.savemat(saveParamsName, store_non_linearity)
    if saveModel:
        torch.save(model, saveModelName)

    return x, output, out_pred, train_error


if __name__ == "__main__":
    x = create_input(batch_size=2)
    x, go_signal_idx, go_signal_moments = create_go_signal(x)
    output = create_output(x, go_signal_idx, go_signal_moments)

    dataset = (torch.from_numpy(x).float(), torch.from_numpy(output).float())
    model = Network()
    x, output, out_pred, train_error = train(dataset, model, n_epochs=1000, saveParams=True)
    from src.visualization.plot_input_output import plot_result_training
    plot_result_training(x, output, out_pred.detach().numpy(), train_error, title="one LSTM per neuron")
    # x = torch.from_numpy(x)
    # model = Network()
    # model(x.float())

# TODO: plot predicted output and check training error!!!
