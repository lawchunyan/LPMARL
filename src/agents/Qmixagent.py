import torch
import torch.nn as nn
import numpy as np
import wandb

from torch.optim import Adam, RMSprop
from collections import namedtuple
from src.agents.baseagent import BaseAgent
from src.nn.qmixer import Qmixer
from src.nn.mlp import MultiLayeredPerceptron as MLP
from utils.torch_util import dn

Transition_base = namedtuple('Transition', (
    'state', 'action', 'reward', 'next_state', 'terminated', 'avail_action', 'global_state_prev', 'global_state_next'))


class QmixAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, n_ag, memory_len=10000, batch_size=20, train_start=100, epsilon_start=1.0,
                 epsilon_decay=2 * 1e-5, gamma=0.99, hidden_dim=32, mixer=False, loss_ftn=nn.MSELoss(), lr=1e-4,
                 state_shape=(0, 0), memory_type='ep', name='Qmix', target_update_interval=200, target_tau=0.5, **kwargs):
        super(QmixAgent, self).__init__(state_dim, action_dim, memory_len, batch_size, train_start, gamma,
                                        memory_type=memory_type, name=name)

        self.critic = MLP(state_dim, action_dim, hidden_dims=[], out_actiation=nn.Identity())
        self.target_critic = MLP(state_dim, action_dim, hidden_dims=[], out_actiation=nn.Identity())
        self.update_target_network(self.target_critic.parameters(), self.critic.parameters())

        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay

        self.memory.transition = Transition_base

        self.mixer = None
        self.loss_ftn = loss_ftn

        self.params = list(self.critic.parameters())

        if mixer:
            assert int(np.prod(state_shape)) > 0
            self.mixer = Qmixer(n_ag, state_shape)
            self.params += list(self.mixer.parameters())
            self.target_mixer = Qmixer(n_ag, state_shape)
            self.update_target_network(self.target_mixer.parameters(), self.mixer.parameters())

        self.optimizer = Adam(self.params, lr=lr)

        self.target_update_interval = target_update_interval
        self.tau = target_tau

    def get_qs(self, state):
        state_tensor = torch.Tensor(state).to(self.device)
        qs = self.critic(state_tensor)
        return qs

    def get_target_qs(self, state):
        state_tensor = torch.Tensor(state).to(self.device)
        qs = self.target_critic(state_tensor)
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

        env_action = self.convert_nn_action_to_sc2_action(action, avail_actions)

        return env_action, action

    @staticmethod
    def convert_nn_action_to_sc2_action(action, avail_actions):
        # dead agent index
        dead_ag_loc = avail_actions[:, 0] == 1
        # from env action, we mask out first element
        env_action = action + 1
        # dead agent only do 0th action
        env_action[dead_ag_loc] = 0
        return env_action

    def fit(self, e):
        samples = self.memory.sample(self.batch_size)
        num_samples = len(samples)

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
        avail_action_tensor = torch.Tensor(avail_actions).to(self.device)
        gs_tensor = torch.Tensor(gs).to(self.device)
        gs_n_tensor = torch.Tensor(gs_n).to(self.device)

        curr_qs = self.get_qs(s_tensor.reshape(-1, self.state_dim)).gather(dim=1,
                                                                           index=a_tensor).reshape(num_samples, -1,
                                                                                                   1)

        with torch.no_grad():
            next_qs_all = self.get_target_qs(ns_tensor.reshape(-1, self.state_dim))
            mask = torch.cat([avail_action_tensor[1:], avail_action_tensor[0].unsqueeze(0)], dim=0)
            mask = mask[:, :, 1:].reshape(-1, self.action_dim)
            next_qs_all[mask == 0] = -9999

            next_qs = next_qs_all.max(dim=1)[0].reshape(num_samples, -1, 1)

        if self.mixer is not None:
            curr_qs = self.mixer(curr_qs, states=gs_tensor).squeeze()
            next_qs = self.target_mixer(next_qs, states=gs_n_tensor).squeeze()

        q_target = r_tensor + self.gamma * next_qs.detach() * (1 - t_tensor)

        # mask = s_tensor.reshape(-1, self.state_dim)[:, 0]

        loss_q = ((curr_qs - q_target) ** 2).sum() / num_samples

        self.optimizer.zero_grad()
        loss_q.backward()
        nn.utils.clip_grad_norm_(self.params, 10)

        wandb.log({'loss_critic': loss_q.item(),
                   })

        if e % self.target_update_interval == 0:
            self.update_target()

    def update_target(self):
        self.update_target_network(self.target_critic.parameters(), self.critic.parameters(), self.tau)
        self.update_target_network(self.target_mixer.parameters(), self.mixer.parameters(), self.tau)
