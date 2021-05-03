import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam
from collections import namedtuple
from src.agents.baseagent import BaseAgent
from src.nn.qmixer import Qmixer
from src.nn.mlp import MultiLayeredPerceptron as MLP
from utils.torch_util import dn

Transition_base = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminated', 'avail_action'))


class QAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, n_ag, memory_len=10000, batch_size=5, train_start=1000, epsilon=1.0,
                 epsilon_decay=2 * 1e-5, gamma=0.99, hidden_dim=32, mixer=True, loss_ftn=nn.MSELoss(), lr=1e-4,
                 state_shape=(0, 0)):
        super(QAgent, self).__init__(state_dim, action_dim, memory_len, batch_size, train_start, gamma)

        self.critic = MLP(state_dim, action_dim, out_actiation=nn.Identity())
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.memory.transition = Transition_base

        self.mixer = None
        self.loss_ftn = loss_ftn

        params = list(self.critic.parameters())

        if mixer:
            assert int(np.prod(state_shape)) > 0
            self.mixer = Qmixer(n_ag, state_shape)
            params += list(self.mixer.parameters())

        self.optimizer = Adam(params, lr=5e-4)

    def get_qs(self, state):
        state_tensor = torch.Tensor(state).to(self.device)
        qs = self.critic(state_tensor)
        return qs

    def get_action(self, state, avail_actions, explore=True):
        qs = dn(self.get_qs(state))

        true_avail_action_mask = avail_actions[:, 1:]
        qs[true_avail_action_mask == 0] = -9999

        if explore:
            argmax_action = qs.argmax(axis=1)

            rand_q_val = np.random.random((qs.shape))
            rand_q_val[true_avail_action_mask == 0] = -9999
            rand_action = rand_q_val.argmax(axis=1)

            select_random = np.random.random((rand_action.shape)) < self.epsilon

            action = select_random * rand_action + ~select_random * argmax_action
        else:
            action = qs.argmax(axis=1)

        # anneal epsilon
        self.epsilon = max(self.epsilon - self.epsilon_decay, 0.05)

        return action

    def fit(self):
        samples = self.memory.sample(self.batch_size)

        s = []
        a = []
        r = []
        ns = []
        t = []
        avail_actions = []

        lst = [s, a, r, ns, t, avail_actions]

        for sample in samples:
            for sam, lst in zip(sample, lst):
                lst.append(sam)

        s_tensor = torch.Tensor(s).to(self.device)
        a_tensor = torch.tensor(a, dtype=int).reshape(-1, 1).to(self.device)
        r_tensor = torch.Tensor(r).to(self.device)
        ns_tensor = torch.Tensor(ns).to(self.device)
        t_tensor = torch.Tensor(t).to(self.device)

        curr_qs = self.get_qs(s_tensor).gather(dim=1,
                                               index=a_tensor)

        with torch.no_grad():
            next_qs = self.get_qs(ns_tensor).max(dim=1)[0]

        if self.mixer is not None:
            curr_qs = self.mixer(curr_qs, state=None)
            next_qs = self.mixer(next_qs, state=None)

        q_target = r_tensor + self.gamma * next_qs * (1 - t_tensor)

        loss_q = nn.MSELoss()(curr_qs, q_target)

        self.optimizer.zero_grad()
        loss_q.backward()
