import dgl
import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam
from collections import namedtuple
from src.nn.optimlayer_backwardhook import EdgeMatching_autograd
from src.agents.baseagent import BaseAgent
from src.utils.torch_util import dn
from src.utils.OUNoise import OUNoise
from src.utils.graph_utils import get_filtered_node_idx, get_filtered_edge_idx
from src.nn.MultiLayeredPerceptron import MultiLayeredPerceptron as MLP
from src.nn.GraphNeuralNetwork import GraphNeuralNet

Transition_LP = namedtuple('Transition_LP_hier', (
    'g', 'state', 'en_state', 'high_action', 'low_action', 'reward', 'n_state', 'n_en_state', 'terminated',
    'avail_action', 'high_rwd'))


class DDPGLPAgent(BaseAgent):
    def __init__(self, **kwargs):
        super(DDPGLPAgent, self).__init__(**kwargs, name='ddpg')

        self.n_ag = kwargs['n_ag']
        self.n_en = kwargs['n_en']

        self.memory.transition = Transition_LP

        state_dim = kwargs['state_dim']
        action_dim = kwargs['action_dim']
        hidden_dim = kwargs['hidden_dim']
        self.hidden_dim = hidden_dim

        # embedding layers
        self.ag_embedding = MLP(state_dim, hidden_dim, hidden_dims=[32])
        self.en_embedding = MLP(2, hidden_dim, hidden_dims=[32])

        # layers
        self.gnn = GraphNeuralNet(hidden_dim, hidden_dim)
        self.critic_h = MLP(hidden_dim * 2, 1)
        self.actor_h = EdgeMatching_autograd()
        self.critic_l = MLP(hidden_dim * 2 + action_dim, action_dim)
        self.actor_l = MLP(state_dim + 2, action_dim, out_activation=nn.Tanh())

        self.gnn_target = GraphNeuralNet(hidden_dim, hidden_dim)
        self.critic_h_target = MLP(hidden_dim * 2, 1)
        self.critic_l_target = MLP(hidden_dim * 2 + action_dim, action_dim)
        self.actor_l_target = MLP(state_dim + 2, action_dim, out_activation=nn.Tanh())

        self.update_target_network(self.gnn_target.parameters(), self.gnn.parameters())
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

    def get_gnn_embedding(self, g, state, landmark_state):
        state = torch.Tensor(state).to(self.device)
        landmark_state = torch.Tensor(landmark_state).to(self.device)

        s_feature = self.ag_embedding(state)
        en_feature = self.en_embedding(landmark_state)

        ag_node_idx = get_filtered_node_idx(g, 0)
        en_node_idx = get_filtered_node_idx(g, 1)

        full_feat = torch.Tensor(g.number_of_nodes(), self.hidden_dim).to(self.device)
        full_feat[ag_node_idx] = s_feature
        full_feat[en_node_idx] = en_feature

        out_feat = self.gnn(g, full_feat)
        return out_feat

    def get_high_qs(self, g, embedding):
        bipartite_edges = get_filtered_edge_idx(g, 0)
        _from, _to = g.find_edges(bipartite_edges)
        bipartite_feat = torch.cat([embedding[_from], embedding[_to]], dim=-1)

        high_qs = self.critic_h(bipartite_feat)
        return high_qs

    def get_low_action(self, ag_obs, high_feat, explore=True):
        low_action = dn(self.actor_l(torch.Tensor(np.concatenate([ag_obs, high_feat], axis=-1)).to(self.device)))
        if explore:
            l_action = []
            for i in range(self.n_ag):
                low_action = self.noise[i].get_action(low_action[i])
                l_action.append(low_action)
            self.epsilon = self.noise[0].epsilon
            low_action = np.array(l_action)
        return low_action

    def get_high_action(self, high_qs, explore=False, h_action=None):
        if explore:
            high_qs = torch.normal(mean=high_qs, std=self.std)
            self.std = max(self.std - self.epsilon_decay, 0.05)

        solution = self.actor_h.apply(high_qs.squeeze()).to(self.device)

        # Sample from policy
        policy = solution.reshape(self.n_ag, self.n_en)  # to prevent - 0
        policy += 1e-4
        policy = policy / policy.sum(1, keepdims=True)

        if h_action is not None:
            chosen_h_action = h_action.to(self.device)
        else:
            chosen_h_action = torch.distributions.categorical.Categorical(policy).sample().to(self.device)

        return chosen_h_action

    def get_low_qs(self, ag_feat, en_feat):
        action_inp = torch.Tensor(np.concatenate([ag_feat, en_feat], axis=-1)).to(self.device)
        low_qs = self.critic_l(action_inp)
        return low_qs

    def get_action(self, g, state, landmark_state, avail_actions=None, explore=True):
        embedding = self.get_gnn_embedding(g, state, landmark_state)
        high_qs = self.get_high_qs(g, embedding)
        high_action = self.get_high_action(high_qs)
        chosen_high_feat = landmark_state[high_action]
        low_action = self.get_low_action(state, chosen_high_feat)
        return high_action, low_action

    def fit(self, e):
        samples = self.memory.sample(self.batch_size)

        g = []
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

        lst = [g, ag_obs, en_obs, a_h, a_l, r, n_ag_obs, n_en_obs, t, avail_actions, high_r]

        for sample in samples:
            for s, l in zip(sample, lst):
                l.append(s)

        g = dgl.batch(g)
        ag_obs = np.stack(ag_obs)
        en_obs = np.stack(en_obs)
        a_h = torch.Tensor(a_h)
        a_l = torch.Tensor(a_l)
        r = torch.Tensor(r)
        n_ag_obs = torch.Tensor(n_ag_obs)
        n_en_obs = torch.Tensor(n_en_obs)
        t = torch.Tensor(t)
        high_r = torch.Tensor(high_r)

        embeddings = self.get_gnn_embedding(dgl.batch(g), ag_obs, en_obs)


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
            high_qs = coeff.reshape(self.n_ag, self.n_en).gather(dim=1, index=high_action_taken.reshape(-1, 1))

            n_agent_obs, n_enemy_obs = n_ag_obs[sample_idx], n_en_obs[sample_idx]
            next_high_q_val, next_high_action = self.get_high_qs(n_agent_obs, n_enemy_obs, self.n_ag,
                                                                 self.n_en). \
                reshape(self.n_ag, self.n_en).max(dim=1)

            high_q_target = r_h + self.gamma * next_high_q_val.detach() * (1 - terminated)

            # low q update
            low_qs = self.critic_l(torch.cat(
                [torch.Tensor(agent_obs).to(self.device), torch.Tensor(enemy_obs).to(self.device), low_action], dim=-1))

            with torch.no_grad():
                inp = torch.Tensor(np.concatenate([n_agent_obs, n_enemy_obs[dn(next_high_action)]], axis=-1)).to(
                    self.device)
                next_low_q_val = self.critic_l_target(torch.cat([inp, self.actor_l_target(inp)], dim=-1))
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
        self.critic_optimizer.step()

        actor_inp = torch.cat(actor_inputs, dim=0)
        loss_a_l = -self.critic_l_target(torch.cat([actor_inp, self.actor_l(actor_inp)], dim=-1)).mean()

        self.actor_optimizer.zero_grad()
        loss_a_l.backward()
        self.actor_optimizer.step()

        ret_dict = {'loss_c_h': loss_c_h.item(),
                    'loss_c_l': loss_c_l.item(),
                    'loss_a_h': loss_a_h.item(),
                    'loss_a_l': loss_a_l.item(),
                    'high_weight': self.high_weight
                    }

        self.high_weight = min(0.5, self.high_weight + 4e-4)

        # gradient on high / low action
        if e % self.target_update_interval == 0:
            self.update_target()

        return ret_dict

    def update_target(self):
        self.update_target_network(self.critic_l_target.parameters(), self.critic_l.parameters(), tau=self.target_tau)
        self.update_target_network(self.critic_h_target.parameters(), self.critic_h.parameters(), tau=self.target_tau)
        self.update_target_network(self.actor_l_target.parameters(), self.actor_l.parameters(), tau=self.target_tau)
