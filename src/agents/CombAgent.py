import torch
import torch.nn as nn
import numpy as np
import wandb

from torch.optim import Adam
from collections import namedtuple
from src.nn.comblayer import MatchingLayer
from src.agents.baseagent import BaseAgent
from utils.torch_util import dn

Transition_LP = namedtuple('Transition_LP',
                           ('state', 'action', 'reward'))


class RLAgent(BaseAgent):
    def __init__(self, state_dim, n_ag, n_en, action_dim=5, batch_size=5, memory_len=10000, epsilon_start=1.0,
                 epsilon_decay=2e-5, train_start=1000, gamma=0.99, hidden_dim=32, loss_ftn=nn.MSELoss(), lr=5e-4,
                 memory_type="ep", target_tau=1.0, name='LP', target_update_interval=200, low_action=True):
        super(RLAgent, self).__init__(state_dim, action_dim, memory_len, batch_size, train_start, gamma,
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
        self.actor_h = MatchingLayer(n_ag)

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
        high_action = self.get_high_action(agent_obs, enemy_obs, self.n_ag, self.n_en, explore=False)
        return dn(high_action)

    def get_high_qs(self, agent_obs, enemy_obs, num_ag=8, num_en=8):
        agent_side_input = np.concatenate([np.tile(agent_obs[i], (num_en, 1)) for i in range(num_ag)])
        enemy_side_input = np.tile(enemy_obs, (num_ag, 1))

        concat_input = torch.Tensor(np.concatenate([agent_side_input, enemy_side_input], axis=-1)).to(self.device)
        coeff = self.critic_h(concat_input)

        # dead_enemy = enemy_obs[:, -1] == 0

        # reshaped_coeff = coeff.reshape(num_ag, num_en)
        # reshaped_coeff[:, dead_enemy] = 0
        # coeff = reshaped_coeff.reshape(-1, 1)

        return coeff

    def get_high_qs_target(self, agent_obs, enemy_obs, num_ag=8, num_en=8):
        agent_side_input = np.concatenate([np.tile(agent_obs[i], (num_en, 1)) for i in range(num_ag)])
        enemy_side_input = np.tile(enemy_obs, (num_ag, 1))

        concat_input = torch.Tensor(np.concatenate([agent_side_input, enemy_side_input], axis=-1)).to(self.device)
        coeff = self.critic_h_target(concat_input)
        return coeff

    def get_high_action(self, agent_obs, enemy_obs, num_ag=8, num_en=8, explore=False, h_action=None):
        coeff = self.get_high_qs(agent_obs, enemy_obs, num_ag, num_en)

        # coeff = coeff_bef.reshape(num_ag, num_en)
        if explore:
            coeff = torch.normal(mean=coeff, std=self.std)

        solution = self.actor_h([coeff.squeeze()])

        # Sample from policy
        policy = solution.reshape(num_ag, num_en)  # to prevent - 0
        policy += 1e-4
        policy = policy / policy.sum(1, keepdims=True)

        if h_action is not None:
            chosen_h_action = h_action
        else:
            chosen_h_action = torch.distributions.categorical.Categorical(policy).sample()

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

        for sample_idx in range(self.batch_size):
            agent_obs, enemy_obs = s[sample_idx][:self.n_ag], s[sample_idx][self.n_ag:]
            high_action_taken = a[sample_idx]
            r_h = r[sample_idx]

            coeff = self.get_high_qs(agent_obs, enemy_obs, num_ag=self.n_ag, num_en=self.n_en)
            high_qs = coeff.reshape(self.n_ag, self.n_en).gather(dim=1, index=high_action_taken.reshape(-1, 1))

            high_q_target = r_h

            loss_critic_h.append(((high_qs - high_q_target) ** 2).mean())

            # low_action = self.get_low_action(agent_obs, high_action, high_en_feat, avail_action)
        loss_c_h = torch.stack(loss_critic_h).mean()

        self.optimizer.zero_grad()
        loss_c_h.backward()
        self.optimizer.step()

        wandb.log({'loss_c_h': loss_c_h.item()})

        self.high_weight = min(0.5, self.high_weight + 4e-4)

        # gradient on high / low action
        if e % self.target_update_interval == 0:
            self.update_target()

    def update_target(self):
        pass
