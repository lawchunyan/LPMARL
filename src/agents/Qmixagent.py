import torch
import torch.nn as nn
import numpy as np
import wandb

from torch.optim import Adam
from collections import namedtuple
from src.agents.baseagent import BaseAgent
from src.nn.qmixer import Qmixer
from src.nn.mlp import MultiLayeredPerceptron as MLP
from utils.torch_util import dn

Transition_base = namedtuple('Transition', (
    'state', 'action', 'reward', 'next_state', 'terminated', 'avail_action', 'global_state_prev', 'global_state_next'))


class QAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, n_ag, memory_len=10000, batch_size=20, train_start=100, epsilon_start=1.0,
                 epsilon_decay=2 * 1e-5, gamma=0.99, hidden_dim=32, mixer=False, loss_ftn=nn.MSELoss(), lr=1e-4,
                 state_shape=(0, 0), memory_type='ep'):
        super(QAgent, self).__init__(state_dim, action_dim, memory_len, batch_size, train_start, gamma, memory_type=memory_type)

        self.critic = MLP(state_dim, action_dim, out_actiation=nn.Identity())
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay

        self.memory.transition = Transition_base

        self.mixer = None
        self.loss_ftn = loss_ftn

        params = list(self.critic.parameters())

        if mixer:
            assert int(np.prod(state_shape)) > 0
            self.mixer = Qmixer(n_ag, state_shape)
            params += list(self.mixer.parameters())

        self.optimizer = Adam(params, lr=lr)

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

        env_action = self.convert_nn_action_to_env_action(action, avail_actions)

        return env_action, action

    @staticmethod
    def convert_nn_action_to_env_action(action, avail_actions):
        dead_ag_loc = avail_actions[:, 0] == 1
        env_action = action + 1
        env_action[dead_ag_loc] = 0
        return env_action

    def fit(self):
        samples = self.memory.sample(self.batch_size)

        s = []
        a = []
        r = []
        ns = []
        t = []
        avail_actions = []
        gs = []
        gs_n = []

        lst = [s, a, r, ns, t, avail_actions, gs, gs_n]

        for sample in samples:
            for sam, llst in zip(sample, lst):
                llst.append(sam)

        s_tensor = torch.Tensor(s).to(self.device)
        a_tensor = torch.tensor(a, dtype=int).reshape(-1, 1).to(self.device)
        r_tensor = torch.Tensor(r).to(self.device)
        ns_tensor = torch.Tensor(ns).to(self.device)
        t_tensor = torch.Tensor(t).to(self.device)
        gs_tensor = torch.Tensor(gs).to(self.device)
        gs_n_tensor = torch.Tensor(gs_n).to(self.device)

        curr_qs = self.get_qs(s_tensor.reshape(-1, self.state_dim)).gather(dim=1,
                                                                           index=a_tensor).reshape(self.batch_size, -1,
                                                                                                   1)

        with torch.no_grad():
            next_qs = self.get_qs(ns_tensor.reshape(-1, self.state_dim)).max(dim=1)[0].reshape(self.batch_size, -1, 1)

        if self.mixer is not None:
            curr_qs = self.mixer(curr_qs, states=gs_tensor)
            next_qs = self.mixer(next_qs, states=gs_n_tensor)

        q_target = r_tensor + self.gamma * next_qs.detach() * (1 - t_tensor)

        loss_q = nn.MSELoss()(curr_qs, q_target)

        self.optimizer.zero_grad()
        loss_q.backward()

        wandb.log({'loss_critic': loss_q.item()})
