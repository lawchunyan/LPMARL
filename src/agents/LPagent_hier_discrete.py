import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam
from collections import namedtuple
from src.nn.optimlayer_backwardhook import EdgeMatching_autograd
from src.agents.LPagent_Hier import LPAgent
from src.agents.network.actor import Actor
from src.nn.MultiLayeredPerceptron import MLP_sigmoid
from src.utils.torch_util import dn
from src.utils.OUNoise import OUNoise

Transition_LP = namedtuple('Transition_LP_hier',
                           ('state_ag', 'state_en', 'high_action', 'low_action', 'low_reward', 'n_state_ag',
                            'n_state_en',
                            'terminated', 'avail_action', 'high_rwd', 'h_transition'))


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

        # self.critic_batch = nn.BatchNorm1d(critic_in_dim, affine=False)

        # layers
        self.critic_l = nn.Sequential(nn.Linear(critic_in_dim + action_dim, hidden_dim),
                                      nn.LeakyReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.LeakyReLU(),
                                      nn.Linear(hidden_dim, 1),
                                      )
        self.critic_l_target = nn.Sequential(nn.Linear(critic_in_dim + action_dim, hidden_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(hidden_dim, 1),
                                             )

        self.coeff_layer = MLP_sigmoid(critic_in_dim, 1, hidden_dims=[hidden_dim, hidden_dim])

        self.actor_h = EdgeMatching_autograd()

        # self.actor_l = nn.ModuleList([Actor(critic_in_dim, action_dim) for _ in range(kwargs['n_ag'])])
        # self.actor_l_target = nn.ModuleList([Actor(critic_in_dim, action_dim) for _ in range(kwargs['n_ag'])])
        self.actor_l = Actor(critic_in_dim, action_dim)
        self.actor_l_target = Actor(critic_in_dim, action_dim)

        self.update_target_network(self.critic_h_target.parameters(), self.critic_h.parameters())
        self.update_target_network(self.critic_l_target.parameters(), self.critic_l.parameters())
        self.update_target_network(self.actor_l_target.parameters(), self.actor_l.parameters())

        # src parameters
        # epsilon_start = kwargs['epsilon_start']
        # epsilon_decay = kwargs['epsilon_decay']
        lr = kwargs['lr']

        # self.noise = [OUNoise(action_dim, epsilon_start=epsilon_start, epsilon_decay=epsilon_decay) for _ in
        #               range(kwargs['n_ag'])]

        self.actor_h_optimizer = Adam(self.coeff_layer.parameters(), lr=lr)
        self.critic_h_optimizer = Adam(list(self.critic_h.parameters()), lr=lr)
        self.critic_l_optimizer = Adam(list(self.critic_l.parameters()), lr=lr)
        self.actor_optimizer = Adam(self.actor_l.parameters(), lr=lr)

        self.ag_indices = [i for i in range(kwargs['n_ag'])]
        self.en_indices = [i for i in range(kwargs['n_en'])]

        self.batch_norm = False

    def get_action(self, agent_obs, enemy_obs, avail_actions=None, explore=True, get_high_action=True):
        if get_high_action:
            high_action, high_feat, chosen_action_logit_h, coeff = self.get_high_action(agent_obs, enemy_obs, self.n_ag,
                                                                                        self.n_en, explore=explore)
            self.high_action = high_action.squeeze().tolist()
            high_action = self.high_action
            self.coeff = coeff.mean()
        else:
            high_action = self.high_action
            high_feat = enemy_obs[high_action]
        low_action = self.get_low_action(agent_obs, high_feat, explore=explore)
        out_action = low_action

        return out_action, high_action, low_action

    def get_low_action(self, agent_obs, high_feat, avail_action_mask=None, explore=True):
        low_actor_input = np.concatenate([agent_obs, high_feat], axis=-1)
        low_actor_input = torch.Tensor(low_actor_input).to(self.device)
        low_policy = self.actor_l(low_actor_input)

        # if explore:
        #     random_qs = torch.rand_like(low_qs)
        #
        #     argmax_action = low_qs.argmax(-1)
        #     random_action = random_qs.argmax(-1)
        #
        #     random_val = torch.rand(argmax_action.shape).to(self.device)
        #     select_random = random_val < self.epsilon
        #
        #     out_action = select_random * random_action + ~select_random * argmax_action
        #
        # else:
        #     out_action = low_qs.argmax(-1)

        # self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
        # chosen_l_action = torch.distributions.categorical.Categorical(low_policy).sample()
        return low_policy

    def get_coeff(self, agent_obs, enemy_obs, num_ag, num_en):
        if type(agent_obs) == list:
            critic_in = [np.concatenate([agent_obs[i], enemy_obs[j]]) for i in self.ag_indices for j in self.en_indices]
            critic_in = torch.Tensor(critic_in).unsqueeze(0).to(self.device)
        else:
            critic_in = [torch.cat([agent_obs[:, i], enemy_obs[:, j]], dim=-1) for i in self.ag_indices for j in
                         self.en_indices]
            critic_in = torch.stack(critic_in, dim=1)
        coeff = self.coeff_layer(critic_in)
        return coeff

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
            if self.batch_norm:
                critic_in = self.critic_batch(critic_in.transpose(1, 2)).transpose(1, 2)

        critic_out = self.critic_h_target(critic_in)
        return critic_out

    def get_high_action(self, agent_obs, enemy_obs, num_ag, num_en, explore=False, return_logit_probs=False,
                        h_action=None):
        coeff = self.get_coeff(agent_obs, enemy_obs, num_ag, num_en)  # shape = (batch, n_ag x n_en, 1)

        n_batch = coeff.shape[0]
        # if torch.isnan(coeff.sum()):
        #     A = 0
        if n_batch == 1:
            solution = self.actor_h.apply(coeff.squeeze())  # .to(self.device)
        else:
            solution = torch.stack([self.actor_h.apply(c.squeeze()) for c in coeff])

        # Sample from policy
        policy = solution.reshape(n_batch, num_ag, num_en)  # to prevent - 0
        policy += 1e-4
        if explore:
            policy += self.epsilon
        policy = policy / policy.sum(-1, keepdims=True)

        if return_logit_probs:
            chosen_h_action = h_action
            p_logit = torch.log(policy)
            # chosen_action_logit_h = p_logit.gather(-1, chosen_h_action.unsqueeze(-1))
            # chosen_action_logit_h = torch.log(policy).gather(dim=1, index=chosen_h_action.reshape(-1, 1))
        else:
            chosen_h_action = torch.distributions.categorical.Categorical(policy).sample()
            # chosen_h_action = torch.arange(self.n_ag).reshape(1, -1).to(self.device)
            p_logit = None

        if n_batch == 1:
            chosen_h_en_feat = enemy_obs.squeeze()[chosen_h_action.squeeze().tolist()]
        else:
            chosen_h_en_feat = enemy_obs[torch.arange(n_batch)[:, None], chosen_h_action]

        return chosen_h_action, chosen_h_en_feat, p_logit, coeff

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
        h_transition = []

        lst = [ag_obs, en_obs, a_h, a_l, r, n_ag_obs, n_en_obs, t, avail_actions, high_r, h_transition]

        for sample in samples:
            for s, l in zip(sample, lst):
                l.append(s)

        ag_obs = torch.Tensor(ag_obs).to(self.device)
        en_obs = torch.Tensor(en_obs).to(self.device)
        a_h = torch.tensor(a_h, dtype=torch.int64).to(self.device)
        a_l = torch.stack(a_l).to(self.device)

        n_ag_obs = torch.Tensor(n_ag_obs).to(self.device)
        n_en_obs = torch.Tensor(n_en_obs).to(self.device)
        r_h = torch.Tensor(high_r).to(self.device)
        r_l = torch.Tensor(r).to(self.device)

        t = torch.Tensor(t).to(self.device)
        h_transition = torch.BoolTensor(h_transition).to(self.device)

        if r_h.sum() != 0:

            high_qs = self.get_high_qs(ag_obs, en_obs, self.n_ag, self.n_en)
            high_qs = high_qs.squeeze().reshape(-1, self.n_ag, self.n_en)
            high_qs_taken = high_qs.gather(index=a_h.unsqueeze(-1), dim=-1)

            with torch.no_grad():
                next_high_qs = self.get_high_qs_target(ag_obs, en_obs, self.n_ag, self.n_en)
                next_high_qs = next_high_qs.squeeze().reshape(-1, self.n_ag, self.n_en)
                next_argmax_high_q, next_high_action = next_high_qs.max(dim=-1)
                high_q_target = self.gamma * next_argmax_high_q.squeeze().sum(-1) + r_h[:, 0] * (
                        1 - t[:, 0])

            loss_c_h = self.loss_ftn(high_qs_taken.squeeze().sum(-1), high_q_target.squeeze())

            self.critic_h_optimizer.zero_grad()
            loss_c_h.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_h.parameters(), 0.5)
            self.critic_h_optimizer.step()

            # high actor: Note: WIP
            with torch.no_grad():
                high_qs = self.get_high_qs(ag_obs, en_obs, self.n_ag, self.n_en)

            loss_a_h = []
            for i, transition in enumerate(h_transition):
                if transition:
                    _, _, logits, coeff = self.get_high_action(ag_obs, en_obs, self.n_ag, self.n_en,
                                                               return_logit_probs=True,
                                                               h_action=a_h)

                    value = (high_qs.reshape(logits.shape[0], logits.shape[1], -1) * logits).sum(-1).detach()
                    taken_q = high_qs.reshape(-1, self.n_ag, self.n_en).gather(-1, a_h.unsqueeze(-1)).detach()
                    pol_target = taken_q.squeeze() - value

                    logits_taken = logits.gather(-1, a_h.unsqueeze(-1)).squeeze()
                    loss_a_h.append((-pol_target * logits_taken).mean())

            if len(loss_a_h) > 0:
                loss_a_h = torch.stack(loss_a_h).mean()
                self.actor_h_optimizer.zero_grad()
                loss_a_h.backward()
                torch.nn.utils.clip_grad_norm_(self.coeff_layer.parameters(), 0.5)
                self.actor_h_optimizer.step()
            else:
                loss_a_h = torch.Tensor([0])

        else:
            loss_c_h = torch.Tensor([0])
            loss_a_h = torch.Tensor([0])

        # low critic
        low_critic_in = torch.cat([ag_obs, en_obs[torch.arange(self.batch_size)[:, None], a_h], a_l], dim=-1).detach()

        low_qs = self.critic_l(low_critic_in).squeeze()

        with torch.no_grad():
            next_input = torch.cat([n_ag_obs, n_en_obs], dim=-1)  #
            next_low_q = self.critic_l_target(torch.cat([next_input, self.actor_l_target(next_input)], dim=-1))
            low_q_target = next_low_q.squeeze() * self.gamma * (1 - t) + r_l

        loss_c_l = self.loss_ftn(low_qs, low_q_target)

        self.critic_l_optimizer.zero_grad()
        loss_c_l.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_l.parameters(), 1)
        self.critic_l_optimizer.step()

        # low actor
        actor_inp = torch.cat([ag_obs, en_obs[torch.arange(self.batch_size)[:, None], a_h]], dim=-1)
        loss_a_l = -self.critic_l(torch.cat([actor_inp, self.actor_l(actor_inp)], dim=-1))

        loss_a_l = loss_a_l.mean()
        self.actor_optimizer.zero_grad()
        loss_a_l.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_l.parameters(), 1)
        self.actor_optimizer.step()

        ret_dict = {
            'loss_c_h': loss_c_h.item(),
            'loss_c_l': loss_c_l.item(),
            'loss_a_h': loss_a_h.item(),
            'loss_a_l': loss_a_l.item(),
            # 'high_weight': self.high_weight
        }

        # gradient on high / low action
        if e % self.target_update_interval == 0:
            self.update_target()

        return ret_dict

    def update_target(self):
        self.update_target_network(self.critic_l_target.parameters(), self.critic_l.parameters(), tau=self.target_tau)
        self.update_target_network(self.critic_h_target.parameters(), self.critic_h.parameters(), tau=self.target_tau)
        self.update_target_network(self.actor_l_target.parameters(), self.actor_l.parameters(), tau=self.target_tau)
