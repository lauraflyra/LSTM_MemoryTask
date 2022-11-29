from src.data.create_input import *
from src.models.network import Network
import torch
x = torch.from_numpy(x)
model = Network()
model(x.float())