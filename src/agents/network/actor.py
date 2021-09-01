import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=32):
        super(Actor, self).__init__()
        self.l = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                               nn.LeakyReLU(),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.LeakyReLU(),
                               nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        out = self.l(x)
        pol = torch.softmax(out, dim=-1)
        return pol
