import torch
import torch.nn as nn
import numpy as np
import wandb

from torch.optim import Adam
from collections import namedtuple
from src.nn.optimlayer_backwardhook import EdgeMatching_autograd
from src.agents.baseagent import BaseAgent
from src.agents.network.actor import Actor
from src.nn.MultiLayeredPerceptron import MultiLayeredPerceptron as MLP
from src.utils.torch_util import dn

Transition_LP = namedtuple('Transition_LP_hier',
                           ('state_ag', 'state_en', 'high_action', 'low_action', 'reward', 'n_state_ag', 'n_state_en',
                            'terminated',
                            'avail_action', 'high_rwd', 'h_transition'))


class LPAgent(BaseAgent):
    def __init__(self, state_dim, n_ag, n_en, action_dim=5, batch_size=5, memory_len=10000, epsilon_start=1.0,
                 epsilon_decay=2e-5, train_start=1000, gamma=0.99, hidden_dim=32, loss_ftn=nn.MSELoss(), lr=5e-4,
                 memory_type="ep", target_tau=1.0, name='LP', target_update_interval=200, sc2=True, en_feat_dim=None,
                 coeff=5, **kwargs):
        super(LPAgent, self).__init__(state_dim, action_dim, memory_len, batch_size, train_start, gamma,
                                      memory_type=memory_type, name=name)
        self.memory.transition = Transition_LP

        if en_feat_dim is None:
            critic_in_dim = state_dim * 2
            critic_l_out_dim = action_dim + 1
        else:
            critic_in_dim = state_dim + en_feat_dim
            critic_l_out_dim = action_dim

        # layers
        self.critic_h = MLP(critic_in_dim, 1, hidden_dims=[hidden_dim, hidden_dim])
        self.critic_h_target = MLP(critic_in_dim, 1, hidden_dims=[hidden_dim, hidden_dim])

        self.coeff_layer = MLP(critic_in_dim, 1, hidden_dims=[hidden_dim, hidden_dim])
        self.actor_h = EdgeMatching_autograd()

        self.critic_l = MLP(critic_in_dim, critic_l_out_dim, hidden_dims=[hidden_dim, hidden_dim])
        self.critic_l_target = MLP(critic_in_dim, critic_l_out_dim, hidden_dims=[hidden_dim, hidden_dim])

        self.update_target_network(self.critic_l_target.parameters(), self.critic_l.parameters())
        self.update_target_network(self.critic_h_target.parameters(), self.critic_h.parameters())

        # src parameters
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.std = 0.5

        # optimizer
        # self.optimizer = Adam(list(self.parameters()), lr=lr)

        self.actor_h_optimizer = Adam(self.coeff_layer.parameters(), lr=lr)
        self.critic_h_optimizer = Adam(self.critic_h.parameters(), lr=lr)
        self.critic_l_optimizer = Adam(self.critic_l.parameters(), lr=lr)

        self.loss_ftn = loss_ftn

        # other
        self.n_ag = n_ag
        self.n_en = n_en

        self.high_weight = 0.1
        self.target_tau = target_tau
        self.target_update_interval = target_update_interval
        self.sc2 = sc2
        self.ag_indices = [i for i in range(n_ag)]
        self.en_indices = [i for i in range(n_en)]

        self.fixed_action = False

    def get_action(self, agent_obs, enemy_obs, avail_actions=None, explore=True, high_action=None):
        if high_action is None:
            high_action, high_feat, chosen_action_logit_h = self.get_high_action(agent_obs, enemy_obs, self.n_ag,
                                                                                 self.n_en, explore=explore)
            self.high_feat = high_feat

        high_feat = self.high_feat

        if self.sc2:
            avail_action_mask = self.get_sc2_low_action_mask(avail_actions, high_action)
            low_action = self.get_low_action(agent_obs, high_feat, avail_action_mask, explore=explore)
            out_action = self.hier_action_to_sc2_action(low_action, high_action, avail_actions)

        else:
            avail_action_mask = None
            low_action = self.get_low_action(agent_obs, high_feat, None, explore=explore)
            out_action = low_action

        # anneal epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

        return out_action, high_action, low_action

    def get_coeff(self, agent_obs, enemy_obs, num_ag, num_en):
        if len(agent_obs.shape) < 3:
            critic_in = [np.concatenate([agent_obs[i], enemy_obs[j]]) for i in self.ag_indices for j in self.en_indices]
            critic_in = torch.Tensor(critic_in).unsqueeze(0).to(self.device)
        else:
            # critic_in = [torch.cat([agent_obs[:, i], enemy_obs[:, j]], dim=-1) for i in self.ag_indices for j in
            #              self.en_indices]
            # critic_in = torch.stack(critic_in, dim=1)
            pass
        coeff = self.coeff_layer(critic_in)
        return coeff

    def get_high_qs(self, agent_obs, enemy_obs, num_ag, num_en):
        # high_q_input = self.get_bipartite_state(agent_obs, enemy_obs, num_ag, num_en).to(self.device)
        # coeff = self.critic_h(high_q_input)
        #
        # if self.sc2:
        #     dead_agent = agent_obs[:, -1] == 0
        #     dead_enemy = enemy_obs[:, -1] == 0
        #     reshaped_coeff = coeff.reshape(num_ag, num_en)
        #     reshaped_coeff[dead_agent] = 0
        #     reshaped_coeff[:, dead_enemy] = 0
        #     coeff = reshaped_coeff.reshape(-1, 1)

        if True:
            critic_in = [np.concatenate([agent_obs[i], enemy_obs[j]]) for i in self.ag_indices for j in self.en_indices]
            critic_in = torch.Tensor(critic_in).unsqueeze(0).to(self.device)
        else:
            critic_in = [torch.cat([agent_obs[:, i], enemy_obs[:, j]], dim=-1) for i in self.ag_indices for j in
                         self.en_indices]
            critic_in = torch.stack(critic_in, dim=1)
        critic_out = self.critic_h(critic_in)
        return critic_out

    def get_high_qs_target(self, agent_obs, enemy_obs, num_ag, num_en):
        # high_q_input = self.get_bipartite_state(agent_obs, enemy_obs, num_ag, num_en).to(self.device)
        # coeff = self.critic_h_target(high_q_input)
        #
        # if self.sc2:
        #     dead_agent = agent_obs[:, -1] == 0
        #     dead_enemy = enemy_obs[:, -1] == 0
        #     reshaped_coeff = coeff.reshape(num_ag, num_en)
        #     reshaped_coeff[dead_agent] = 0
        #     reshaped_coeff[:, dead_enemy] = 0
        #     coeff = reshaped_coeff.reshape(-1, 1)
        if True:
            critic_in = [np.concatenate([agent_obs[i], enemy_obs[j]]) for i in self.ag_indices for j in self.en_indices]
            critic_in = torch.Tensor(critic_in).unsqueeze(0).to(self.device)
        else:
            critic_in = [torch.cat([agent_obs[:, i], enemy_obs[:, j]], dim=-1) for i in self.ag_indices for j in
                         self.en_indices]
            critic_in = torch.stack(critic_in, dim=1)

        critic_out = self.critic_h_target(critic_in)
        return critic_out

    def get_high_action(self, agent_obs, enemy_obs, num_ag, num_en, explore=False, h_action=None, h_transition=True):
        if h_transition:
            coeff = self.get_coeff(agent_obs, enemy_obs, num_ag, num_en)

            if explore:
                coeff = torch.normal(mean=coeff, std=self.std)
                self.std = max(self.std - self.epsilon_decay, 0.05)

            # solution = self.actor_h([coeff.squeeze()]).to(self.device)
            solution = self.actor_h.apply(coeff.squeeze()).to(self.device)

            # Sample from policy
            policy = solution.reshape(num_ag, num_en)  # to prevent - 0
            policy += 1e-4
            policy = policy / policy.sum(1, keepdims=True)

            if h_action is not None:
                chosen_h_action = h_action.to(self.device)
            else:
                chosen_h_action = torch.distributions.categorical.Categorical(policy).sample().to(self.device)

            logit_h = torch.log(policy)
            # chosen_action_logit_h = None
            chosen_h_en_feat = enemy_obs[chosen_h_action.tolist()]
        else:
            chosen_h_action = None
            chosen_h_en_feat = enemy_obs[h_action.tolist()]
            logit_h = torch.zeros((num_ag, num_en))

        return chosen_h_action, chosen_h_en_feat, logit_h

    def get_low_qs(self, agent_obs, high_feat):
        action_inp = torch.Tensor(np.concatenate([agent_obs, high_feat], axis=-1)).to(self.device)
        low_action_val = self.critic_l(action_inp)
        return low_action_val

    def get_low_qs_target(self, agent_obs, high_feat, *args, **kwargs):
        action_inp = torch.Tensor(np.concatenate([agent_obs, high_feat], axis=-1)).to(self.device)
        low_action_val = self.critic_l_target(action_inp)
        return low_action_val

    def get_low_action(self, agent_obs, high_feat, avail_action_mask=None, explore=True):
        low_action_val = self.get_low_qs(agent_obs, high_feat).cpu().detach().numpy()
        low_action = self.exploration_using_q(low_action_val, self.epsilon, explore, avail_action_mask)
        return low_action

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
            for sam, llst in zip(sample, lst):
                llst.append(sam)

        next_avail_actions = np.stack(avail_actions[1:] + [avail_actions[0]])

        loss_critic_h = []
        loss_actor_h = []
        loss_critic_l = []
        loss_actor_l = []

        for sample_idx in range(len(samples)):
            agent_obs, enemy_obs = ag_obs[sample_idx], en_obs[sample_idx]
            high_action_taken = a_h[sample_idx].to(self.device)
            low_action = a_l[sample_idx]  # .to(self.device)
            next_avail_action = next_avail_actions[sample_idx]  # .to(self.device)
            r_l = torch.Tensor(r[sample_idx])  # .to(self.device)
            r_h = high_r[sample_idx]  # .to(self.device)
            h_trans = h_transition[sample_idx]
            terminated = t[sample_idx]  # .to(self.device)

            # high critic update
            _, high_en_feat, h_logit = self.get_high_action(agent_obs, enemy_obs,
                                                            num_ag=self.n_ag, num_en=self.n_en,
                                                            h_action=high_action_taken,
                                                            explore=False,
                                                            h_transition=h_trans
                                                            )

            high_qs = self.get_high_qs(agent_obs, enemy_obs, num_ag=self.n_ag, num_en=self.n_en)
            high_qs = high_qs.reshape(self.n_ag, self.n_en)
            high_qs_taken = high_qs.gather(dim=1, index=high_action_taken.to(self.device).reshape(-1, 1))

            with torch.no_grad():
                n_agent_obs, n_enemy_obs = n_ag_obs[sample_idx], n_en_obs[sample_idx]
                next_high_q_val, next_high_action = self.get_high_qs(n_agent_obs, n_enemy_obs, self.n_ag,
                                                                     self.n_en).reshape(self.n_ag, self.n_en).max(dim=1)

                high_q_target = r_h + self.gamma * next_high_q_val * (1 - terminated)

            loss_critic_h.append(self.loss_ftn(high_qs_taken, high_q_target))

            # high actor update
            value = (high_qs * h_logit).sum(-1).detach()
            pol_target = (high_qs - value).detach()
            logits_taken = h_logit.gather(-1, high_action_taken.unsqueeze(-1)).squeeze()
            loss_actor_h.append(-pol_target * logits_taken)

            # low critic update
            low_qs = self.get_low_qs(agent_obs, high_en_feat).gather(dim=1,
                                                                     index=torch.tensor(low_action,
                                                                                        dtype=int).reshape(-1,
                                                                                                           1))

            with torch.no_grad():
                next_low_q_val = self.get_low_qs_target(n_agent_obs, n_enemy_obs[next_high_action])
                dead_mask = None
                if self.sc2:
                    next_low_action_mask = self.get_sc2_low_action_mask(next_avail_action, high_action_taken)
                    next_low_q_val[torch.tensor(next_low_action_mask == 0)] = -9999
                    dead_mask = next_avail_action[:, 0]
                selected_low_q_target = next_low_q_val.max(dim=1)[0]

                low_q_target = r_l + self.gamma * selected_low_q_target * (1 - terminated)

            low_qs[dead_mask == 1] = 0
            low_q_target[dead_mask == 1] = 0
            loss_critic_l.append(self.loss_ftn(low_qs, low_q_target))

            # loss_actor_h.append(-h_logit * high_qs)

            # low_action = self.get_low_action(agent_obs, high_action, high_en_feat, avail_action)
        loss_c_h = torch.stack(loss_critic_h).mean()
        loss_c_l = torch.stack(loss_critic_l).mean()
        loss_a_h = torch.stack(loss_actor_h).mean()

        self.critic_h_optimizer.zero_grad()
        loss_c_h.backward()
        self.critic_h_optimizer.step()

        self.actor_h_optimizer.zero_grad()
        loss_a_h.backward()
        self.actor_h_optimizer.step()

        self.critic_l_optimizer.zero_grad()
        loss_c_l.backward()
        self.critic_l_optimizer.step()

        # self.optimizer.zero_grad()
        # (loss_c_h * self.high_weight + loss_c_l + loss_a_h * 0.1).backward()
        # # (loss_c_l).backward()
        # self.optimizer.step()

        wandb.log({'loss_c_h': loss_c_h.item(),
                   'loss_c_l': loss_c_l.item(),
                   'loss_a_h': loss_a_h.item(),
                   # 'high_weight': self.high_weight
                   })

        # self.high_weight = min(0.5, self.high_weight + 4e-4)

        # gradient on high / low action
        if e % self.target_update_interval == 0:
            self.update_target()

    def update_target(self):
        self.update_target_network(self.critic_l_target.parameters(), self.critic_l.parameters(), tau=self.target_tau)
        self.update_target_network(self.critic_h_target.parameters(), self.critic_h.parameters(), tau=self.target_tau)
