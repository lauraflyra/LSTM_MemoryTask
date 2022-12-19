from src.data.create_input_output_additive import *
from src.models.network_additive import Network
import torch
import torch.nn as nn


def train(dataset, n_epochs = 1000, saveLossEvery = 100):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x, output = dataset

    for epoch in range(n_epochs):
        out_pred, model_variables = model(x)

        loss = loss_fn(out_pred, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % saveLossEvery == 0 :
            print("training:", epoch, loss.item())

    return x, out_pred

if __name__ == "__main__":
    model = Network()
    dataset = (torch.from_numpy(x).float(), torch.from_numpy(y).float())
    train(dataset)