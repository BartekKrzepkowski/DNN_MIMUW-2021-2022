import torch
import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    """Network to train in given problem. It is based on network given in lab 4."""
    def __init__(self, dims):
        super(Net, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(h1, h2)
                                 for h1, h2 in zip(dims[:-1], dims[1:])])

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for l in self.fc[:-1]:
            x = l(x)
            x = F.relu(x)
        x = self.fc[-1](x)
        return F.log_softmax(x, dim=1)
