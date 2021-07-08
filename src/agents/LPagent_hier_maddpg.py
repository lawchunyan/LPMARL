import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam
from collections import namedtuple
from src.nn.optimlayer_backwardhook import EdgeMatching_autograd
from src.agents.LPagent_Hier import LPAgent
from src.utils.torch_util import dn
from src.utils.OUNoise import OUNoise

Transition_LP = namedtuple('Transition_LP_hier',
                           ('state_ag', 'state_en', 'high_action', 'low_action', 'low_reward', 'n_state_ag',
                            'n_state_en',
                            'terminated', 'avail_action', 'high_rwd'))


class DDPGLPAgent(LPAgent):
    def __init__(self, **kwargs):
        super(DDPGLPAgent, self).__init__(**kwargs, name='ddpg')
        self.memory.transition = Transition_LP

        en_feat_dim = kwargs['en_feat_dim']
        state_dim = kwargs['state_dim']
        action_dim = kwargs['action_dim']
        hidden_dim = kwargs['hidden_dim']

        if en_feat_dim is None:
            critic_in_dim = state_dim * 2
        else:
            critic_in_dim = state_dim + en_feat_dim

        # layers
        self.critic_l = nn.Sequential(nn.Linear(critic_in_dim + action_dim, hidden_dim),
                                      nn.LeakyReLU(),
                                      nn.Linear(hidden_dim, 1),
                                      )
        self.critic_l_target = nn.Sequential(nn.Linear(critic_in_dim + action_dim, hidden_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(hidden_dim, 1),
                                             )

        self.actor_l = nn.Sequential(nn.Linear(critic_in_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, action_dim),
                                     nn.Tanh()
                                     )
        self.actor_l_target = nn.Sequential(nn.Linear(critic_in_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, action_dim),
                                            nn.Tanh()
                                            )

        self.update_target_network(self.critic_h_target.parameters(), self.critic_h.parameters())
        self.update_target_network(self.critic_l_target.parameters(), self.critic_l.parameters())
        self.update_target_network(self.actor_l_target.parameters(), self.actor_l.parameters())

        # src parameters
        epsilon_start = kwargs['epsilon_start']
        epsilon_decay = kwargs['epsilon_decay']
        lr = kwargs['lr']

        self.noise = [OUNoise(action_dim, epsilon_start=epsilon_start, epsilon_decay=epsilon_decay) for _ in
                      range(kwargs['n_ag'])]

        self.critic_h_optimizer = Adam(list(self.critic_l.parameters()), lr=lr)
        self.critic_l_optimizer = Adam(list(self.critic_l.parameters()), lr=lr)
        self.actor_optimizer = Adam(self.actor_l.parameters(), lr=lr)

        self.actor_h = EdgeMatching_autograd()

        self.ag_indices = [i for i in range(kwargs['n_ag'])]
        self.en_indices = [i for i in range(kwargs['n_en'])]

    def get_action(self, agent_obs, enemy_obs, avail_actions=None, explore=True, get_high_action=True):
        if get_high_action:
            high_action, high_feat, chosen_action_logit_h = self.get_high_action(agent_obs, enemy_obs, self.n_ag,
                                                                                 self.n_en, explore=explore)
            self.high_action = high_action.squeeze()
        else:
            # high_action, high_feat, chosen_action_logit_h = self.get_high_action(agent_obs, enemy_obs, self.n_ag,
            #                                                                      self.n_en, explore=explore,
            #                                                                      h_action=self.high_action)
            high_action = self.high_action
            high_feat = enemy_obs[high_action]
        low_action = self.get_low_action(agent_obs, high_feat, explore=explore)
        out_action = low_action

        return out_action, dn(high_action.squeeze()), low_action

    def get_low_action(self, agent_obs, high_feat, avail_action_mask=None, explore=True):
        low_action = dn(self.actor_l(torch.Tensor(np.concatenate([agent_obs, high_feat], axis=-1)).to(self.device)))
        if explore:
            l_action = []
            for i in range(self.n_ag):
                low_action = self.noise[i].get_action(low_action[i])
                l_action.append(low_action)
            self.epsilon = self.noise[0].epsilon
            low_action = np.array(l_action)
        return low_action

    def get_high_qs(self, agent_obs, enemy_obs, num_ag, num_en):
        if type(agent_obs) == list:
            critic_in = [np.concatenate([agent_obs[i], enemy_obs[j]]) for i in self.ag_indices for j in self.en_indices]
            critic_in = torch.Tensor(critic_in).unsqueeze(0).to(self.device)
        else:
            critic_in = [torch.cat([agent_obs[:, i], enemy_obs[:, j]], dim=-1) for i in self.ag_indices for j in
                         self.en_indices]
            critic_in = torch.stack(critic_in, dim=1)
        critic_out = self.critic_h(critic_in)
        return critic_out

    def get_high_qs_target(self, agent_obs, enemy_obs, num_ag, num_en):
        if type(agent_obs) == list:
            critic_in = [np.concatenate([agent_obs[i], enemy_obs[j]]) for i in self.ag_indices for j in self.en_indices]
            critic_in = torch.Tensor(critic_in).unsqueeze(0).to(self.device)
        else:
            critic_in = [torch.cat([agent_obs[:, i], enemy_obs[:, j]], dim=-1) for i in self.ag_indices for j in
                         self.en_indices]
            critic_in = torch.stack(critic_in, dim=1)
        critic_out = self.critic_h_target(critic_in)
        return critic_out

    def get_high_action(self, agent_obs, enemy_obs, num_ag, num_en, explore=False, h_action=None):
        critic_out = self.get_high_qs(agent_obs, enemy_obs, num_ag, num_en)  # shape = (batch, n_ag x n_en, 1)

        n_batch = critic_out.shape[0]

        if n_batch == 1:
            solution = self.actor_h.apply(critic_out.squeeze())  # .to(self.device)
        else:
            solution = torch.stack([self.actor_h.apply(c.squeeze()) for c in critic_out])

        # Sample from policy
        policy = solution.reshape(n_batch, num_ag, num_en)  # to prevent - 0
        policy += 1e-4
        policy = policy / policy.sum(-1, keepdims=True)

        if h_action is not None:
            chosen_h_action = h_action
            p_logit = torch.log(policy)
            chosen_action_logit_h = p_logit.gather(-1, chosen_h_action.unsqueeze(-1))
            # chosen_action_logit_h = torch.log(policy).gather(dim=1, index=chosen_h_action.reshape(-1, 1))
        else:
            chosen_h_action = torch.distributions.categorical.Categorical(policy).sample()
            chosen_action_logit_h = None

        if n_batch == 1:
            chosen_h_en_feat = enemy_obs[chosen_h_action.squeeze().tolist()]
        else:
            chosen_h_en_feat = enemy_obs[torch.arange(n_batch)[:, None], chosen_h_action]

        return chosen_h_action, chosen_h_en_feat, chosen_action_logit_h

    def fit(self, e):
        samples = self.memory.sample(self.batch_size)
        ag_obs = []
        en_obs = []
        a_h = []
        a_l = []
        r = []
        n_ag_obs = []
        n_en_obs = []
        t = []
        avail_actions = []
        high_r = []

        lst = [ag_obs, en_obs, a_h, a_l, r, n_ag_obs, n_en_obs, t, avail_actions, high_r]

        for sample in samples:
            for s, l in zip(sample, lst):
                l.append(s)

        ag_obs = torch.Tensor(ag_obs).to(self.device)
        en_obs = torch.Tensor(en_obs).to(self.device)
        a_h = torch.tensor(a_h, dtype=torch.int64).to(self.device)
        a_l = torch.Tensor(a_l).to(self.device)

        n_ag_obs = torch.Tensor(n_ag_obs).to(self.device)
        n_en_obs = torch.Tensor(n_en_obs).to(self.device)
        r_h = torch.Tensor(high_r).to(self.device)

        high_qs = self.get_high_qs(ag_obs, en_obs, self.n_ag, self.n_en)
        high_qs = high_qs.squeeze().reshape(-1, self.n_ag, self.n_en)
        high_qs_taken = high_qs.gather(index=a_h.unsqueeze(-1), dim=-1)

        with torch.no_grad():
            next_high_qs = self.get_high_qs_target(n_ag_obs, n_en_obs, self.n_ag, self.n_en)
            next_high_qs = next_high_qs.squeeze().reshape(-1, self.n_ag, self.n_en)
            next_argmax_high_q, next_high_action = next_high_qs.max(dim=-1)
            high_q_target = self.gamma * next_argmax_high_q.squeeze() + r_h

        high_critic_loss = self.loss_ftn(high_qs_taken.squeeze(), high_q_target)

        self.critic_h_optimizer.zero_grad()
        high_critic_loss.backward()
        self.critic_h_optimizer.step()

        low_critic_in = torch.cat([ag_obs, en_obs[torch.arange(self.batch_size)[:, None], a_h], a_l], dim=-1)
        low_qs = self.critic_l(low_critic_in)

        with torch.no_grad():
            inp = torch.cat([n_ag_obs, n_en_obs[torch.arange(self.batch_size)[:, None], next_high_action]], dim=-1)
            next_action = self.actor_l_target(inp)
            next_low_critic_in = torch.cat([inp, next_action], dim=-1)
            next_low_q = self.critic_l_target(next_low_critic_in)
            low_q_target = next_low_q.squeeze() + torch.Tensor(r).to(self.device)

        low_q_loss = self.loss_ftn(low_qs.squeeze(), low_q_target)

        self.critic_l_optimizer.zero_grad()
        low_q_loss.backward()
        self.critic_l_optimizer.step()

        actor_inp = torch.cat([ag_obs, en_obs[torch.arange(self.batch_size)[:, None], a_h]], dim=-1)
        low_actor_loss = -self.critic_l(torch.cat([actor_inp, self.actor_l(actor_inp)], dim=-1))
        low_actor_loss = low_actor_loss.mean()
        self.actor_optimizer.zero_grad()
        low_actor_loss.backward()
        self.actor_optimizer.step()

        ret_dict = {'loss_c_h': high_critic_loss.item(),
                    'loss_c_l': low_q_loss.item(),
                    # 'loss_a_h': loss_a_h.item(),
                    'loss_a_l': low_actor_loss.item(),
                    'high_weight': self.high_weight
                    }

        # gradient on high / low action
        if e % self.target_update_interval == 0:
            self.update_target()

        return ret_dict

    def update_target(self):
        self.update_target_network(self.critic_l_target.parameters(), self.critic_l.parameters(), tau=self.target_tau)
        self.update_target_network(self.critic_h_target.parameters(), self.critic_h.parameters(), tau=self.target_tau)
        self.update_target_network(self.actor_l_target.parameters(), self.actor_l.parameters(), tau=self.target_tau)
