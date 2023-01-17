import torch
import torch.nn as nn
from src.data.common_vars import DATA_PATH
from scipy import io
import os


def train(dataset,
          model,
          n_epochs,
          save_params=True,
          save_model=True,
          verbose=True,
          data_path = DATA_PATH,
          save_params_name=os.path.join(DATA_PATH, "params.mat"),
          save_model_name=os.path.join(DATA_PATH, "model.pt"),
          checkpoints_file_name="model_checkpoints",
          save_loss_every=100,
          save_params_every=100,
          save_model_every=200):
    """

    :param dataset: tuple containing the input and desired output of the model, plus extra parameters
    that differ depending on the model type.
        if model == Factorial or Additive, dataset = (x,y, go_signal_time_slots, go_signal_moments)
        if model = OneNeuron, dataset = (x,y, cue_amp, go_start_time)
    :param model: desired network model to be trained
    :param n_epochs: total number of epochs for training
    :param save_params: if parameters, such as the gain, should be saved or not
    :param save_model: if the model should be saved or not
    :param data_path: path to results folder
    :param verbose: if the training error should be displayed during training
    :param save_params_name: file name for the parameters file
    :param save_model_name: file name for the model file
    :param checkpoints_file_name: file 'pre name' for the checkpoints
    :param save_loss_every: save the loss every x epochs
    :param save_params_every: save the params every x epochs
    :param save_model_every: save the model every x epochs
    :return: store_non_linearity, dictionary where input, output, non-linearity params and other params are stored
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    store_non_linearity = {"x": [], "y": [], "y_pred": [], "training_error": [], "n_epochs": n_epochs, 'n':[], 's':[]}

    if model.__class__.__name__ == 'NetworkOneNeuronGeadah':
        x, y, cue_amp, go_start_time = dataset
        store_non_linearity["go_start_time"] = go_start_time
        store_non_linearity["cue_amp"] = cue_amp

    elif model.__class__.__name__ == 'NetworkOneNeuron':
        x, y, cue_amp, go_start_time = dataset
        store_non_linearity["go_start_time"] = go_start_time
        store_non_linearity["cue_amp"] = cue_amp

    elif model.__class__.__name__ == 'NetworkFactorial' or model.__class__.__name__ == 'NetworkAdditive':
        x, y, go_signal_time_slots, go_signal_moments = dataset
        store_non_linearity["go_signal_time_slots"] = go_signal_time_slots
        store_non_linearity["go_signal_moments"] = go_signal_moments
    else:
        raise Exception("Model type not supported in this training script")

    store_non_linearity["x"].append(x.detach().numpy())
    store_non_linearity["y"].append(y.detach().numpy())

    for epoch in range(n_epochs):
        y_pred, model_variables = model(x)

        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % save_loss_every == 0:
            store_non_linearity["training_error"].append(loss.item())
            store_non_linearity["y_pred"].append(y_pred.detach().numpy())
            store_non_linearity["save_loss_every"] = save_loss_every
            if verbose:
                print("training:", epoch, loss.item())

        if epoch % save_params_every == 0 and save_params:
            if model.__class__.__name__ == 'NetworkOneNeuron':
                store_non_linearity["g"].append(model_variables.detach().numpy())
            if model.__class__.__name__ == 'NetworkOneNeuronGeadah':
                store_non_linearity["n"].append(model_variables[0].detach().numpy())
                store_non_linearity["s"].append(model_variables[1].detach().numpy())
            else:
                if epoch == 0:
                    for count, neuron in enumerate(model_variables.keys()):
                        store_non_linearity[neuron + "-g"] = []
                for count, neuron in enumerate(model_variables.keys()):
                    store_non_linearity[neuron + "-g"].append(
                        torch.stack(model_variables[neuron]['g'][1:]).detach().numpy())

        if epoch % save_model_every == 0 and save_model:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(data_path, checkpoints_file_name + "_" + str(epoch) + ".pt"))

    if save_params:
        io.savemat(save_params_name, store_non_linearity)
    if save_model:
        torch.save(model, save_model_name)

    return store_non_linearity
