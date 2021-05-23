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

Transition_base = namedtuple('Transition', ('state', 'action', 'reward', 'global_state_prev'))


class QmixAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, n_ag, n_en, memory_len=10000, batch_size=20, train_start=100,
                 epsilon_start=1.0,
                 epsilon_decay=2 * 1e-5, gamma=0.99, hidden_dim=32, mixer=False, loss_ftn=nn.MSELoss(), lr=1e-4,
                 state_shape=(0, 0), memory_type='ep', name='Qmix', target_update_interval=200, target_tau=0.5,
                 **kwargs):
        super(QmixAgent, self).__init__(state_dim, action_dim, memory_len, batch_size, train_start, gamma,
                                        memory_type=memory_type, name=name)

        self.critic = MLP(state_dim * 2, 1, hidden_dims=[ ], hidden_activation=nn.LeakyReLU(),
                          out_actiation=nn.LeakyReLU())
        self.target_critic = MLP(state_dim * 2, 1, hidden_dims=[ ], hidden_activation=nn.LeakyReLU(),
                                 out_actiation=nn.LeakyReLU())
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

        self.n_ag = n_ag
        self.n_en = n_en

    def get_qs(self, agent_obs, enemy_obs, num_ag=8, num_en=8):
        agent_side_input = np.concatenate([np.tile(agent_obs[i], (num_en, 1)) for i in range(num_ag)])
        enemy_side_input = np.tile(enemy_obs, (num_ag, 1))

        concat_input = torch.Tensor(np.concatenate([agent_side_input, enemy_side_input], axis=-1)).to(self.device)
        qs = self.critic(concat_input).reshape(num_ag, num_en)
        return qs

    def get_target_qs(self, agent_obs, enemy_obs, num_ag=8, num_en=8):
        agent_side_input = np.concatenate([np.tile(agent_obs[i], (num_en, 1)) for i in range(num_ag)])
        enemy_side_input = np.tile(enemy_obs, (num_ag, 1))

        concat_input = torch.Tensor(np.concatenate([agent_side_input, enemy_side_input], axis=-1)).to(self.device)
        qs = self.critic(concat_input).reshape(num_ag, num_en)
        return qs

    def get_action(self, state, explore=True):

        agent_obs = state[:self.n_ag]
        enemy_obs = state[self.n_ag:]
        qs = dn(self.get_qs(agent_obs, enemy_obs, self.n_ag, self.n_en))

        if explore:
            argmax_action = qs.argmax(axis=1)
            rand_q_val = np.random.random((qs.shape))
            rand_action = rand_q_val.argmax(axis=1)

            select_random = np.random.random((rand_action.shape)) < self.epsilon

            action = select_random * rand_action + ~select_random * argmax_action
        else:
            action = qs.argmax(axis=1)

        # anneal epsilon
        self.epsilon = max(self.epsilon - self.epsilon_decay, 0.05)

        return action

    def fit(self, e):
        samples = self.memory.sample(self.batch_size)

        s = []
        a = []
        r = []
        gs = []

        lst = [s, a, r, gs]

        for sample in samples:
            for sam, llst in zip(sample, lst):
                llst.append(sam)

        s_tensor = torch.Tensor(s).to(self.device)
        a_tensor = torch.tensor(a, dtype=int).to(self.device)
        r_tensor = torch.Tensor(r).to(self.device)
        gs_tensor = torch.Tensor(gs).to(self.device)

        loss = []

        for sample_idx in range(self.batch_size):
            agent_obs, enemy_obs = s_tensor[sample_idx][:self.n_ag], s[sample_idx][self.n_ag:]
            action_taken = a_tensor[sample_idx]
            rwd = r_tensor[sample_idx]
            states = gs_tensor[sample_idx].reshape(-1, 1)

            qs = self.get_qs(agent_obs, enemy_obs, num_ag=self.n_ag, num_en=self.n_en)
            curr_qs = qs.gather(dim=1, index=action_taken.reshape(-1, 1))
            if self.mixer is not None:
                curr_qs = curr_qs.reshape(1, -1, 1)
                curr_qs = self.mixer(curr_qs, states=states)

            loss.append(((curr_qs - rwd) ** 2).mean())

        loss = torch.stack(loss).mean()

        self.optimizer.zero_grad()
        loss.backward()

        wandb.log({'loss_critic': loss.item(),
                   })

        if e % self.target_update_interval == 0:
            self.update_target()

    def update_target(self):
        self.update_target_network(self.target_critic.parameters(), self.critic.parameters(), self.tau)
        self.update_target_network(self.target_mixer.parameters(), self.mixer.parameters(), self.tau)

    def push(self, state, action, reward):

        global_state = state[:self.n_ag].reshape(-1)
        self.memory.push(state, action, reward, global_state)
