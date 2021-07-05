import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam
from collections import namedtuple

from src.nn.mlp import MultiLayeredPerceptron as MLP
from src.agents.baseagent import BaseAgent
from src.utils.torch_util import dn
from src.utils.OUNoise import OUNoise

Transition_LP = namedtuple('Transition_LP_hier',
                           ('state_ag', 'state_en', 'action', 'reward', 'n_state_ag', 'n_state_en'))


class IndependentDDPGAgent(BaseAgent):
    def __init__(self, state_dim, n_ag, n_en, action_dim=5, batch_size=5, memory_len=10000, epsilon_start=1.0,
                 epsilon_decay=2e-5, train_start=1000, gamma=0.99, hidden_dim=32, loss_ftn=nn.MSELoss(), lr=5e-4,
                 memory_type="ep", target_tau=1.0, name='indep', target_update_interval=200, sc2=True, en_feat_dim=None,
                 coeff=5, **kwargs):
        super(IndependentDDPGAgent, self).__init__(state_dim, action_dim, memory_len, batch_size, train_start, gamma,
                                                   memory_type, name)
        self.noise = [OUNoise(action_dim, epsilon_start=epsilon_start, epsilon_decay=epsilon_decay) for _ in
                      range(n_ag)]

        self.actor = [MLP(state_dim + en_feat_dim, action_dim, out_actiation=nn.Tanh()) for _ in range(n_ag)]
        self.actor_target = [MLP(state_dim + en_feat_dim, action_dim, out_actiation=nn.Tanh()) for _ in range(n_ag)]

        self.critic = [MLP(state_dim + en_feat_dim + action_dim, 1, out_actiation=nn.Identity()) for _ in range(n_ag)]
        self.critic_target = [MLP(state_dim + en_feat_dim + action_dim, 1, out_actiation=nn.Identity()) for _ in
                              range(n_ag)]

        for s, t in zip(self.critic, self.critic_target):
            self.update_target_network(t.parameters(), s.parameters(), tau=1.0)
        for s, t in zip(self.actor, self.actor_target):
            self.update_target_network(t.parameters(), s.parameters(), tau=1.0)

        self.memory.transition = Transition_LP

        self.n_ag = n_ag

        self.actor_optimizer = [Adam(actor.parameters(), lr=lr) for actor in self.actor]
        self.critic_optimizer = [Adam(critic.parameters(), lr=lr) for critic in self.critic]

        self.target_update_interval = target_update_interval
        self.target_tau = target_tau

    def get_action(self, state, landmark_state, explore=True, get_high_action=False):
        actor_input = np.concatenate([state, landmark_state], axis=-1)
        actor_input = torch.Tensor(actor_input).to(self.device)
        action = [actor(actor_input[i]).cpu().detach().numpy() for i, actor in enumerate(self.actor)]

        if explore:
            action = np.array([noise.get_action(action[i]) for i, noise in enumerate(self.noise)])

        return action, None, None

    def fit(self, e):
        samples = self.memory.sample(self.batch_size)
        state_ag = []
        state_en = []
        action = []
        reward = []
        n_state_ag = []
        n_state_en = []
        lst = [state_ag, state_en, action, reward, n_state_ag, n_state_en]

        for sample in samples:
            for s, l in zip(sample, lst):
                l.append(s)

        state_ag = torch.Tensor(state_ag).to(self.device)  # [batch x ag x dim]
        state_en = torch.Tensor(state_en).to(self.device)

        n_state_ag = torch.Tensor(n_state_ag).to(self.device)
        n_state_en = torch.Tensor(n_state_en).to(self.device)

        action = torch.Tensor(action).to(self.device)  # [batch * ag * dim]
        reward = torch.Tensor(reward).to(self.device)  # [batch * ag]

        critic_l = 0
        actor_l = 0

        for i in range(self.n_ag):
            s_ag = state_ag[:, i, :]
            s_en = state_en[:, i, :]

            ns_ag = n_state_ag[:, i, :]
            ns_en = n_state_en[:, i, :]

            a_taken = action[:, i, :]
            r = reward[:, i]

            critic_input = torch.cat([s_ag, s_en, a_taken], dim=-1)
            curr_q_val = self.critic[i](critic_input)

            with torch.no_grad():
                ns_input = torch.cat([ns_ag, ns_en], dim=-1)
                next_action = self.actor_target[i](ns_input)
                target_input = torch.cat([ns_ag, ns_en, next_action], dim=-1)
                next_q_val = self.critic_target[i](target_input)

            critic_loss = ((r + self.gamma * next_q_val - curr_q_val) ** 2).mean()

            self.critic_optimizer[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizer[i].step()
            critic_l += critic_loss.item()

            curr_s = torch.cat([s_ag, s_en], dim=-1)
            actor_loss = - self.critic_target[i](torch.cat([curr_s, self.actor[i](curr_s)], dim=-1)).mean()

            self.actor_optimizer[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizer[i].step()
            actor_l += actor_loss.item()

        if e % self.target_update_interval == 0:
            for s, t in zip(self.critic, self.critic_target):
                self.update_target_network(t.parameters(), s.parameters(), tau=self.target_tau)
            for s, t in zip(self.actor, self.actor_target):
                self.update_target_network(t.parameters(), s.parameters(), tau=self.target_tau)

        return {'actor_l': actor_l,
                'critic_l': critic_l}
