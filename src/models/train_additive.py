from src.data.create_input_output_additive import create_input_go_add, create_output_add
from src.models.network_additive import NetworkAdditive
from src.visualization.plot_input_output import plot_result_training_additive
from src.visualization.plot_activation_function_additive import plot_AF_separate_neurons_add
import torch
import torch.nn as nn
import scipy.io
import os


DATA_PATH = "/home/lauraflyra/Documents/SHK_SprekelerLab/LSTM_computations/LSTM_MemoryTask/src/data/"


def train(dataset, model, n_epochs, saveParams=True, saveModel=True, outputTraining=True,
          saveParamsName=os.path.join(DATA_PATH, "params_additive.mat"),
          saveModelName=os.path.join(DATA_PATH, "model.pt"), saveLossEvery=100, saveParamsEvery=100, saveModelEvery=200):
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

    x, output = dataset

    train_error = []

    store_non_linearity = {"x": [], "output": [], "training_error": []}
    if saveParams:
        store_non_linearity["x"].append(x.detach().numpy())
        store_non_linearity["output"].append(output.detach().numpy())

    for epoch in range(n_epochs):
        out_pred, model_variables = model(x)

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
                for count, neuron in enumerate(model_variables.keys()):
                    store_non_linearity[neuron + "-g"] = []
            for count, neuron in enumerate(model_variables.keys()):
                store_non_linearity[neuron + "-g"].append(torch.stack(model_variables[neuron]['g'][1:]).detach().numpy())

        if epoch % saveModelEvery == 0 and saveModel:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(DATA_PATH, "checkpoint_"+str(epoch)+".pt"))

    if saveParams:
        scipy.io.savemat(saveParamsName, store_non_linearity)
    if saveModel:
        torch.save(model, saveModelName)

    return x, output, out_pred, train_error


if __name__ == "__main__":
    x, go_signal_time_slots, go_signal_moments = create_input_go_add(batch_size=500)
    y = create_output_add(x, go_signal_time_slots, go_signal_moments)
    model = NetworkAdditive()
    dataset = (torch.from_numpy(x).float(), torch.from_numpy(y).float())
    x, output, out_pred, train_error = train(dataset, model, n_epochs=2010,
                        saveModelName=os.path.join(DATA_PATH, "model_additive_all_neurons_separate_linear.pt"))
    which_from_batch = plot_result_training_additive(x, output, out_pred.detach().numpy(), train_error, n_epochs=2010,
                                                      plot_every=100)

    plot_AF_separate_neurons_add(params_file=os.path.join(DATA_PATH, "params_additive.mat"), go_signal_time_slots=go_signal_time_slots,
            go_signal_moments=go_signal_moments, which_from_batch=which_from_batch)

    # TODO: see if there are differences for longer, shorter cue times
    # TODO: sample more from the results
    # TODO: how to save model over the course of learning? How is learning?