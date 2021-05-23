import torch
import torch.nn as nn
import numpy as np
import wandb

from torch.optim import Adam
from collections import namedtuple
from src.agents.baseagent import BaseAgent
from utils.torch_util import dn

Transition_LP = namedtuple('Transition_DQN',
                           ('state', 'action', 'reward'))


class DQNAgent(BaseAgent):
    def __init__(self, state_dim, n_ag, n_en, action_dim=5, batch_size=5, memory_len=10000, epsilon_start=1.0,
                 epsilon_decay=2e-5, train_start=1000, gamma=0.99, hidden_dim=32, loss_ftn=nn.MSELoss(), lr=5e-4,
                 memory_type="ep", target_tau=1.0, name='DQN', target_update_interval=200, low_action=True, coeff=6,
                 **kwargs):
        super(DQNAgent, self).__init__(state_dim, action_dim, memory_len, batch_size, train_start, gamma,
                                       memory_type=memory_type, name=name)
        self.memory.transition = Transition_LP

        # layers
        self.critic_h = nn.Sequential(nn.Linear(state_dim * 2, hidden_dim),
                                      nn.LeakyReLU(),
                                      nn.Linear(hidden_dim, 1),
                                      nn.LeakyReLU())

        self.critic_h_target = nn.Sequential(nn.Linear(state_dim * 2, hidden_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(hidden_dim, 1),
                                             nn.LeakyReLU())

        self.update_target_network(self.critic_h_target.parameters(), self.critic_h.parameters())

        # src parameters
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.std = 0.5

        # optimizer
        # params = list(self.critic_h.parameters()) + list(
        #     self.critic_l.parameters())  # + list(self.optim_layer.parameters()) \
        self.optimizer = Adam(list(self.parameters()), lr=lr)

        self.loss_ftn = loss_ftn

        # other
        self.n_ag = n_ag
        self.n_en = n_en

        self.high_weight = 0.1
        self.target_tau = target_tau
        self.target_update_interval = target_update_interval
        self.low_action = low_action

    def get_action(self, obs, explore=True):
        agent_obs = obs[:self.n_ag]
        enemy_obs = obs[self.n_ag:]
        high_action = self.get_high_action(agent_obs, enemy_obs, self.n_ag, self.n_en, explore=explore)
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
        return high_action

    def get_high_qs(self, agent_obs, enemy_obs, num_ag=8, num_en=8):
        agent_side_input = np.concatenate([np.tile(agent_obs[i], (num_en, 1)) for i in range(num_ag)])
        enemy_side_input = np.tile(enemy_obs, (num_ag, 1))

        concat_input = torch.Tensor(np.concatenate([agent_side_input, enemy_side_input], axis=-1)).to(self.device)
        coeff = self.critic_h(concat_input)

        return coeff.reshape(num_ag, num_en)

    def get_high_qs_target(self, agent_obs, enemy_obs, num_ag=8, num_en=8):
        agent_side_input = np.concatenate([np.tile(agent_obs[i], (num_en, 1)) for i in range(num_ag)])
        enemy_side_input = np.tile(enemy_obs, (num_ag, 1))

        concat_input = torch.Tensor(np.concatenate([agent_side_input, enemy_side_input], axis=-1)).to(self.device)
        coeff = self.critic_h_target(concat_input)
        return coeff.reshape(num_ag, num_en)

    def get_high_action(self, agent_obs, enemy_obs, num_ag=8, num_en=8, explore=False, h_action=None):
        qs = dn(self.get_high_qs(agent_obs, enemy_obs, num_ag, num_en))

        if explore:
            argmax_action = qs.argmax(axis=1)
            rand_q_val = np.random.random((qs.shape))
            rand_action = rand_q_val.argmax(axis=1)

            select_random = np.random.random((rand_action.shape)) < self.epsilon

            chosen_h_action = select_random * rand_action + ~select_random * argmax_action
        else:
            chosen_h_action = torch.argmax(qs, dim=-1)

        return chosen_h_action

    def fit(self, e):

        samples = self.memory.sample(self.batch_size)
        s = []
        a = []
        r = []

        lst = [s, a, r]

        for sample in samples:
            for sam, llst in zip(sample, lst):
                llst.append(sam)

        loss_critic_h = []
        a = torch.tensor(a, dtype=int)

        for sample_idx in range(self.batch_size):
            agent_obs, enemy_obs = s[sample_idx][:self.n_ag], s[sample_idx][self.n_ag:]
            high_action_taken = a[sample_idx]
            r_h = r[sample_idx]

            qs = self.get_high_qs(agent_obs, enemy_obs, num_ag=self.n_ag, num_en=self.n_en)
            high_qs = qs.gather(dim=1, index=high_action_taken.reshape(-1, 1))

            high_q_target = r_h

            loss_critic_h.append(((high_qs - high_q_target) ** 2).mean())

            # low_action = self.get_low_action(agent_obs, high_action, high_en_feat, avail_action)
        loss_c_h = torch.stack(loss_critic_h).mean()

        self.optimizer.zero_grad()
        loss_c_h.backward()
        self.optimizer.step()

        wandb.log({'loss_c_h': loss_c_h.item()})

        # gradient on high / low action
        if e % self.target_update_interval == 0:
            self.update_target()

    def update_target(self):
        pass
