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

    def get_high_qs_target(self, g, embedding):
        bipartite_edges = get_filtered_edge_idx(g, 0)
        _from, _to = g.find_edges(bipartite_edges)
        bipartite_feat = torch.cat([embedding[_from], embedding[_to]], dim=-1)

        high_qs = self.critic_h_target(bipartite_feat)
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
        ag_obs = np.concatenate(ag_obs)  # batch X n_ag X state_dim
        en_obs = np.concatenate(en_obs)  # batch X n_en X en_dim
        a_h = torch.stack(a_h).reshape(-1, 1)  # batch * n_ag
        a_l = torch.Tensor(a_l)  # batch X n_ag X n_action
        r = torch.Tensor(r)  # batch X n_ag
        n_ag_obs = np.concatenate(n_ag_obs)  # batch X n_ag X state_dim
        n_en_obs = np.concatenate(n_en_obs)  # batch X n_en X en_dim
        t = torch.Tensor(t)  # batch X n_ag
        high_r = torch.Tensor(high_r).reshape(-1)  # batch * n_ag


        # high_qs
        embeddings = self.get_gnn_embedding(g, ag_obs, en_obs)
        high_qs = self.get_high_qs(g, embeddings).reshape(-1, self.n_en)
        high_qs_taken = high_qs.gather(dim=1, index=a_h)

        with torch.no_grad():
            next_embeddings = self.get_gnn_embedding(g, n_ag_obs, n_en_obs)
            next_high_qs = self.get_high_qs(g, next_embeddings).reshape(-1, self.n_en)
            next_q_target = high_r + self.gamma * next_high_qs.max(dim=1)[0]

        loss_c_h = ((next_q_target - high_qs_taken) ** 2).mean()
        loss_a_h = self.get


        # loss_c_h = torch.stack(loss_critic_h).mean()
        # loss_c_l = torch.stack(loss_critic_l).mean()
        # loss_a_h = torch.stack(loss_actor_h).mean()

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
