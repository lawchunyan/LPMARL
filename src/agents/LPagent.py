import torch
import torch.nn as nn
import numpy as np
import wandb

from torch.optim import Adam
from collections import namedtuple
from src.nn.optimlayer import MatchingLayer
from src.agents.baseagent import BaseAgent
from utils.torch_util import dn

Transition_LP = namedtuple('Transition_LP',
                           ('state', 'high_action', 'action', 'reward', 'next_state', 'terminated', 'avail_action',
                            'high_rwd'))


class RLAgent(BaseAgent):
    def __init__(self, state_dim, n_ag, n_en, action_dim=5, batch_size=5, memory_len=10000, epsilon_start=1.0,
                 epsilon_decay=2e-5, train_start=1000, gamma=0.99, hidden_dim=32, loss_ftn=nn.MSELoss(), lr=5e-4,
                 memory_type="ep", target_tau=1.0, name='LP'):
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
        self.actor_h = MatchingLayer(n_ag, n_en)
        self.critic_l = nn.Sequential(nn.Linear(state_dim * 2, hidden_dim),
                                      nn.LeakyReLU(),
                                      nn.Linear(hidden_dim, action_dim + 1))
        self.critic_l_target = nn.Sequential(nn.Linear(state_dim * 2, hidden_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(hidden_dim, action_dim + 1))

        self.update_target_network(self.critic_l_target.parameters(), self.critic_l.parameters())
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

    def get_action(self, obs, avail_actions):
        agent_obs = obs[:self.n_ag]
        enemy_obs = obs[self.n_ag:]
        high_action, high_feat, chosen_action_logit_h = self.get_high_action(agent_obs, enemy_obs, self.n_ag,
                                                                             self.n_en)
        low_action = self.get_low_action(agent_obs, high_action, high_feat, avail_actions)
        out_action = self.convert_low_action(low_action, high_action, avail_actions)

        # anneal epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

        return out_action, high_action, low_action

    def get_high_qs(self, agent_obs, enemy_obs, num_ag=8, num_en=8):
        agent_side_input = np.concatenate([np.tile(agent_obs[i], (num_en, 1)) for i in range(num_ag)])
        enemy_side_input = np.tile(enemy_obs, (num_ag, 1))

        concat_input = torch.Tensor(np.concatenate([agent_side_input, enemy_side_input], axis=-1)).to(self.device)
        coeff = self.critic_h(concat_input)
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

        # self.optimizer.zero_grad()
        # solution.sum().backward()
        # self.optimizer.step()

        # selecting max value
        # chosen_h_action = solution.reshape(num_ag, num_en).max(dim=1)[1]

        # Sample from policy
        policy = solution.reshape(num_ag, num_en)  # to prevent - 0
        policy += 1e-4
        policy = policy / policy.sum(1, keepdims=True)

        if h_action is not None:
            chosen_h_action = h_action
        else:
            chosen_h_action = torch.distributions.categorical.Categorical(policy).sample()

        chosen_action_logit_h = torch.log(policy).gather(dim=1, index=chosen_h_action.reshape(-1, 1))
        chosen_h_en_feat = enemy_obs[chosen_h_action]

        return chosen_h_action, chosen_h_en_feat, chosen_action_logit_h

    def get_low_qs(self, agent_obs, high_feat):
        action_inp = torch.Tensor(np.concatenate([agent_obs, high_feat], axis=-1)).to(self.device)
        low_action_val = self.critic_l(action_inp)
        return low_action_val

    def get_low_qs_target(self, agent_obs, high_feat):
        action_inp = torch.Tensor(np.concatenate([agent_obs, high_feat], axis=-1)).to(self.device)
        low_action_val = self.critic_l_target(action_inp)
        return low_action_val

    def get_low_action(self, agent_obs, high_action, high_feat, avail_actions, explore=True):
        low_action_val = self.get_low_qs(agent_obs, high_feat).cpu().detach().numpy()

        # making new action mask using
        # moving action
        avail_action_mask_move = avail_actions[:, 1:6]
        # attacking action
        avail_action_mask_high = np.take_along_axis(avail_actions[:, 6:], dn(high_action.reshape(-1, 1)), axis=1)
        avail_action_mask = np.concatenate([avail_action_mask_move, avail_action_mask_high], axis=-1)

        # masking out unavailable action
        low_action_val[avail_action_mask == 0] = -9999

        if explore:
            argmax_low_action = low_action_val.argmax(axis=1)
            rand_low_action_val = np.random.random((low_action_val.shape))
            rand_low_action_val[avail_action_mask == 0] = -9999
            rand_low_action = rand_low_action_val.argmax(axis=1)

            # making random prob of shape = (num_ag,)
            random_indicator = np.random.random((rand_low_action.shape))
            # boolean value of selecting random
            select_random = random_indicator < self.epsilon

            low_action = select_random * rand_low_action + ~select_random * argmax_low_action
        else:
            low_action = low_action_val.argmax(axis=1)

        return low_action

    def convert_low_action(self, low_action, high_action, avail_actions):

        out_action = low_action + 1
        out_action[low_action == 5] = high_action[low_action == 5] + 6

        dead_indicator = avail_actions[:, 0] == 1
        out_action[dead_indicator] = 0

        return out_action

    def fit(self):

        samples = self.memory.sample(self.batch_size)
        s = []
        a_h = []
        a_l = []
        r = []
        ns = []
        t = []
        avail_actions = []
        high_r = []

        lst = [s, a_h, a_l, r, ns, t, avail_actions, high_r]

        for sample in samples:
            for sam, llst in zip(sample, lst):
                llst.append(sam)

        next_avail_actions = np.stack(avail_actions[1:] + [avail_actions[0]])

        loss_critic_h = []
        loss_actor_h = []
        loss_critic_l = []
        loss_actor_l = []

        for sample_idx in range(self.batch_size):
            agent_obs, enemy_obs = s[sample_idx][:self.n_ag], s[sample_idx][self.n_ag:]
            high_action_taken = a_h[sample_idx]
            low_action = a_l[sample_idx]
            next_avail_action = next_avail_actions[sample_idx]
            r_l = r[sample_idx]
            r_h = high_r[sample_idx]

            _, high_en_feat, h_logit = self.get_high_action(agent_obs, enemy_obs, explore=False,
                                                            h_action=high_action_taken,
                                                            num_ag=self.n_ag, num_en=self.n_en)

            coeff = self.get_high_qs(agent_obs, enemy_obs, num_ag=self.n_ag, num_en=self.n_en)
            high_qs = coeff.reshape(self.n_ag, self.n_en).gather(dim=1, index=high_action_taken.reshape(-1, 1))

            n_agent_obs, n_enemy_obs = ns[sample_idx][:self.n_ag], ns[sample_idx][self.n_ag:]
            next_high_q_val, next_high_action = self.get_high_qs(n_agent_obs, n_enemy_obs, self.n_ag,
                                                                 self.n_en). \
                reshape(self.n_ag, self.n_en).max(dim=1)

            high_q_target = r_h + self.gamma * next_high_q_val.detach() * (1 - t[sample_idx])

            # low q update
            low_qs = self.get_low_qs(agent_obs, high_en_feat).gather(dim=1,
                                                                     index=torch.tensor(low_action,
                                                                                        dtype=int).reshape(-1,
                                                                                                           1))

            with torch.no_grad():
                next_low_q_val = self.get_low_qs_target(n_agent_obs, n_enemy_obs[next_high_action])
                next_move_mask = next_avail_action[:, 1:1 + 5]  # shape (n_agent x n_action)
                next_attack_mask = next_avail_action[:, 1 + 5:]

                next_action_mask_attack = np.take_along_axis(next_attack_mask,
                                                             high_action_taken.detach().numpy().reshape(-1, 1),
                                                             -1)
                next_action_mask = np.concatenate([next_move_mask, next_action_mask_attack], axis=-1)
                next_low_q_val[torch.tensor(next_action_mask == 0)] = -9999
                selected_low_q_target = next_low_q_val.max(dim=1)[0]

            low_q_target = r_l + self.gamma * selected_low_q_target * (1 - t[sample_idx])

            dead_mask = next_avail_action[:, 0]
            low_qs[dead_mask == 1] = 0
            low_q_target[dead_mask == 1] = 0

            loss_critic_h.append(self.loss_ftn(high_qs, high_q_target))
            loss_critic_l.append(self.loss_ftn(low_qs, low_q_target))

            loss_actor_h.append(-h_logit * low_qs)

            # low_action = self.get_low_action(agent_obs, high_action, high_en_feat, avail_action)
        loss_c_h = torch.stack(loss_critic_h).mean()
        loss_c_l = torch.stack(loss_critic_l).mean()
        loss_a_h = torch.stack(loss_actor_h).mean()

        self.optimizer.zero_grad()
        (loss_c_h * self.high_weight + loss_c_l + loss_a_h * 0.1).backward()
        self.optimizer.step()

        wandb.log({'loss_c_h': loss_c_h.item(),
                   'loss_c_l': loss_c_l.item(),
                   'loss_a_h': loss_a_h.item(),
                   'high_weight': self.high_weight
                   })

        self.high_weight = min(0.5, self.high_weight + 4e-4)

        # gradient on high / low action

    def update_target(self):
        self.update_target_network(self.critic_l_target.parameters(), self.critic_l.parameters(), tau=self.target_tau)
        self.update_target_network(self.critic_h_target.parameters(), self.critic_h.parameters(), tau=self.target_tau)
