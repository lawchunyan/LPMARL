import torch
import torch.nn as nn
import numpy as np
from src.utils.replaymemory import ReplayMemory, ReplayMemory_episode
from src.utils.torch_util import dn
from datetime import date


class BaseAgent(nn.Module):
    def __init__(self, state_dim, action_dim, memory_len, batch_size, train_start, gamma=0.99, memory_type="sample",
                 name=None, **kwargs):
        super(BaseAgent, self).__init__()

        if memory_type == 'sample':
            self.memory = ReplayMemory(memory_len)
        elif memory_type == 'ep':
            self.memory = ReplayMemory_episode(memory_len)
        else:
            raise NotImplementedError("other than 'sample' or 'ep' memory not implemented")

        assert name is not None

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.train_start = train_start
        self.gamma = gamma
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.name = name
        self.epsilon_min = 0.05

    def get_action(self, *args):
        raise NotImplementedError

    def can_fit(self):
        return True if len(self.memory) > self.train_start else False

    def push(self, *args):
        self.memory.push(*args)

    def save(self, dirname, e):
        torch.save(self.state_dict(), dirname + "/{}.th".format(e))

    @staticmethod
    def update_target_network(target_params, source_params, tau=1.0):
        for t, s in zip(target_params, source_params):
            t.data.copy_(tau * s.data + (1.0 - tau) * t.data)

    @staticmethod
    def exploration_using_q(low_action_val, epsilon, explore, avail_action_mask=None):
        # masking out unavailable action
        low_action_val[avail_action_mask == 0] = -9999

        if explore:
            argmax_low_action = low_action_val.argmax(axis=1)
            rand_low_action_val = np.random.random(low_action_val.shape)
            rand_low_action_val[avail_action_mask == 0] = -9999
            rand_low_action = rand_low_action_val.argmax(axis=1)

            # making random prob of shape = (num_ag,)
            random_indicator = np.random.random(rand_low_action.shape)
            # boolean value of selecting random
            select_random = random_indicator < epsilon

            low_action = select_random * rand_low_action + ~select_random * argmax_low_action
        else:
            low_action = low_action_val.argmax(axis=1)

        return low_action

    @staticmethod
    def hier_action_to_sc2_action(low_action, high_action, avail_actions):
        high_action = high_action.squeeze()
        out_action = low_action + 1
        out_action[low_action == 5] = high_action[low_action == 5] + 6

        dead_indicator = avail_actions[:, 0] == 1
        out_action[dead_indicator] = 0

        return out_action

    @staticmethod
    def get_bipartite_state(agent_obs, enemy_obs, num_ag, num_en):
        agent_side_input = np.concatenate([np.tile(agent_obs[i], (num_en, 1)) for i in range(num_ag)])
        enemy_side_input = np.tile(enemy_obs, (num_ag, 1))
        out = torch.Tensor(np.concatenate([agent_side_input, enemy_side_input], axis=-1))
        return out

    @staticmethod
    def get_sc2_low_action_mask(avail_actions, high_action):
        # making new action mask using existing avail action mask and taken high action
        # moving action
        avail_action_mask_move = avail_actions[:, 1:6]
        # attacking action
        avail_action_mask_high = np.take_along_axis(avail_actions[:, 6:], dn(high_action.reshape(-1, 1)), axis=1)
        avail_action_mask = np.concatenate([avail_action_mask_move, avail_action_mask_high], axis=-1)
        return avail_action_mask




if __name__ == '__main__':
    A = torch.nn.Linear(6, 7)
    B = torch.nn.Linear(6, 7)
