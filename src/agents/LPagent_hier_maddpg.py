import torch
import torch.nn as nn
import numpy as np
import wandb

from torch.optim import Adam
from collections import namedtuple
from src.nn.optimlayer_backwardhook import EdgeMatching_autograd
from src.agents.LPagent_Hier import LPAgent
from src.utils.torch_util import dn
from src.utils.OUNoise import OUNoise

Transition_LP = namedtuple('Transition_LP_hier',
                           ('state_ag', 'state_en', 'high_action', 'low_action', 'reward', 'n_state_ag', 'n_state_en',
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
            # critic_l_out_dim = action_dim + 1
        else:
            critic_in_dim = state_dim + en_feat_dim
            # critic_l_out_dim = action_dim

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

        self.critic_optimizer = Adam(list(self.critic_l.parameters()) + list(self.critic_h.parameters()), lr=lr)
        self.actor_optimizer = Adam(self.actor_l.parameters(), lr=lr)

        self.actor_h = EdgeMatching_autograd()

    def get_action(self, agent_obs, enemy_obs, avail_actions=None, explore=True, get_high_action=True):
        if get_high_action:
            high_action, high_feat, chosen_action_logit_h = self.get_high_action(agent_obs, enemy_obs, self.n_ag,
                                                                                 self.n_en, explore=explore)
            self.high_action = high_action
        else:
            high_action, high_feat, chosen_action_logit_h = self.get_high_action(agent_obs, enemy_obs, self.n_ag,
                                                                                 self.n_en, explore=explore,
                                                                                 h_action=self.high_action)
        low_action = self.get_low_action(agent_obs, high_feat, explore=explore)
        out_action = low_action

        return out_action, dn(high_action), low_action

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

    def get_high_action(self, agent_obs, enemy_obs, num_ag, num_en, explore=False, h_action=None):
        coeff = self.get_high_qs(agent_obs, enemy_obs, num_ag, num_en)

        if explore:
            coeff = torch.normal(mean=coeff, std=self.std)
            self.std = max(self.std - self.epsilon_decay, 0.05)

        solution = self.actor_h.apply(coeff.squeeze()).to(self.device)

        # Sample from policy
        policy = solution.reshape(num_ag, num_en)  # to prevent - 0
        policy += 1e-4
        policy = policy / policy.sum(1, keepdims=True)

        if h_action is not None:
            chosen_h_action = h_action.to(self.device)
        else:
            chosen_h_action = torch.distributions.categorical.Categorical(policy).sample().to(self.device)

        chosen_action_logit_h = torch.log(policy).gather(dim=1, index=chosen_h_action.reshape(-1, 1))
        chosen_h_en_feat = enemy_obs[chosen_h_action.tolist()]

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

        # ag_obs = torch.Tensor(ag_obs).to(self.device)
        # en_obs = torch.Tensor(en_obs).to(self.device)

        loss_critic_h = []
        loss_actor_h = []
        loss_critic_l = []

        actor_inputs = []
        for sample_idx in range(self.batch_size):
            agent_obs, enemy_obs = ag_obs[sample_idx], en_obs[sample_idx]
            high_action_taken = torch.tensor(a_h[sample_idx]).to(self.device)
            low_action = torch.Tensor(a_l[sample_idx]).to(self.device)

            r_l = torch.Tensor(r[sample_idx]).to(self.device)
            r_h = torch.Tensor(high_r[sample_idx]).to(self.device)
            terminated = torch.Tensor(t[sample_idx]).to(self.device)

            _, high_en_feat, h_logit = self.get_high_action(agent_obs, enemy_obs, explore=False,
                                                            h_action=high_action_taken,
                                                            num_ag=self.n_ag, num_en=self.n_en)

            coeff = self.get_high_qs(agent_obs, enemy_obs, num_ag=self.n_ag, num_en=self.n_en)
            high_qs = coeff.reshape(self.n_ag, self.n_en).gather(dim=1, index=high_action_taken.reshape(-1, 1)).squeeze()

            n_agent_obs, n_enemy_obs = n_ag_obs[sample_idx], n_en_obs[sample_idx]
            next_high_q_val, next_high_action = self.get_high_qs(n_agent_obs, n_enemy_obs, self.n_ag,
                                                                 self.n_en). \
                reshape(self.n_ag, self.n_en).max(dim=1)

            high_q_target = r_h + self.gamma * next_high_q_val.detach() * (1 - terminated)

            # low q update
            low_qs = self.critic_l(torch.cat(
                [torch.Tensor(agent_obs).to(self.device), torch.Tensor(enemy_obs).to(self.device), low_action], dim=-1)).squeeze()

            with torch.no_grad():
                inp = torch.Tensor(np.concatenate([n_agent_obs, n_enemy_obs[dn(next_high_action)]], axis=-1)).to(
                    self.device)
                next_low_q_val = self.critic_l_target(torch.cat([inp, self.actor_l_target(inp)], dim=-1)).squeeze()
                low_q_target = r_l + self.gamma * next_low_q_val * (1 - terminated)
                actor_inputs.append(inp)

            loss_critic_h.append(self.loss_ftn(high_qs, high_q_target))
            loss_critic_l.append(self.loss_ftn(low_qs, low_q_target))

            loss_actor_h.append(-h_logit * low_qs)
            # loss_actor_l.append(-self.critic_l(torch.cat([inp, self.actor_l(inp)], dim=-1)))

        loss_c_h = torch.stack(loss_critic_h).mean()
        loss_c_l = torch.stack(loss_critic_l).mean()
        loss_a_h = torch.stack(loss_actor_h).mean()

        self.critic_optimizer.zero_grad()
        # (loss_c_h * self.high_weight + loss_c_l + loss_a_h * 0.1).backward()

        (loss_c_h * self.high_weight + loss_c_l + loss_a_h * 0.1).backward()
        torch.nn.utils.clip_grad_norm_(self.critic_l.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic_h.parameters(), 0.5)
        self.critic_optimizer.step()

        actor_inp = torch.cat(actor_inputs, dim=0)
        loss_a_l = -self.critic_l_target(torch.cat([actor_inp, self.actor_l(actor_inp)], dim=-1)).mean()

        self.actor_optimizer.zero_grad()
        loss_a_l.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_l.parameters(), 0.5)
        self.actor_optimizer.step()

        ret_dict = {'loss_c_h': loss_c_h.item(),
                    'loss_c_l': loss_c_l.item(),
                    'loss_a_h': loss_a_h.item(),
                    'loss_a_l': loss_a_l.item(),
                    'high_weight': self.high_weight
                    }

        self.high_weight = min(0.7, self.high_weight + 1e-5)

        # gradient on high / low action
        if e % self.target_update_interval == 0:
            self.update_target()

        return ret_dict

    def update_target(self):
        self.update_target_network(self.critic_l_target.parameters(), self.critic_l.parameters(), tau=self.target_tau)
        self.update_target_network(self.critic_h_target.parameters(), self.critic_h.parameters(), tau=self.target_tau)
        self.update_target_network(self.actor_l_target.parameters(), self.actor_l.parameters(), tau=self.target_tau)
