import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Qmixer(nn.Module):
    def __init__(self, n_ag, state_shape, embed_dim=32, hypernet_dim=64):
        super(Qmixer, self).__init__()

        self.n_agents = n_ag
        self.state_dim = int(np.prod(state_shape))
        self.embed_dim = embed_dim
        self.hypernet_dim = hypernet_dim

        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.hypernet_dim, self.embed_dim * self.n_agents))
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_dim),
                                           nn.ReLU(),
                                           nn.Linear(self.hypernet_dim, self.embed_dim))
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, qs, states):
        """
        :param qs: batched agents' q values of dimension (batch_size, num_ag, action_dim)
        :param states: batched states
        :return: q_tot of shape (batch_size, 1)
        """
        bs = qs.size(0)

        states = states.reshape(-1, self.state_dim)  # (batch x state_dim)
        agent_qs = qs.view(-1, 1, self.n_agents)  # (batch x 1 x n_ag)

        # First layer
        w1 = torch.abs(self.hyper_w_1(states))  # (batch x embed_dim*n_ag )
        b1 = self.hyper_b_1(states)  # (batch x embed_dim)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)  # (batch x n_ag x embed_dim )
        b1 = b1.view(-1, 1, self.embed_dim)  # (batch x 1 x embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)  # (batch x 1 x embed_dim)

        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))  # (batch x embed_dim)
        w_final = w_final.view(-1, self.embed_dim, 1)  # (batch x embed_dim x 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)  # (batch x 1 x 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v  # (batch x 1 x 1)
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
