import numpy as np
import math
from functools import partial


class ZeroShotGroupingEnv(object):
    def __init__(self, n_ag, n_en, enemy_obs=False, global_reward=False, num_optimal_groups=10):
        self.n_agents = n_ag
        self.n_enemies = n_en

        assert n_ag == n_en, "n_ag > n_en not implemented yet"

        self.n_actions = n_en
        self.enemy_obs = enemy_obs
        self.state_dim = n_ag

        self.correct_action = np.arange(n_en)
        self.global_reward = global_reward

        self.max_reward = 10
        self.min_reward = -10

        if num_optimal_groups == n_ag:
            self.reward_ftn = self.compute_reward_assign
        else:
            self.reward_ftn = partial(self.compute_reward_split, n_groups=num_optimal_groups)

    def get_obs(self):
        ag_state = self.generate_state(self.n_agents)
        if self.enemy_obs:
            return np.concatenate([ag_state, ag_state], axis=0)
        else:
            return ag_state

    def get_obs_size(self):
        return self.n_agents

    def get_avail_actions(self):
        return np.ones(shape=(self.n_agents, self.n_enemies))

    def step(self, actions):
        actions = np.array(actions).reshape(-1)
        reward = self.reward_ftn(actions, self.global_reward)
        return reward, True, {}

    def reset(self):
        return

    def compute_reward_split(self, actions, global_reward=False, n_groups=None):
        setA = set(actions)
        if global_reward:
            max_distance = max(self.n_agents - n_groups, n_groups, self.n_agents)

            min_distance = 0
            curr_distance = abs(len(setA) - n_groups)

            accuracy = (max_distance - curr_distance) / max_distance
            reward = accuracy * (self.max_reward - self.min_reward) + self.min_reward


        else:
            raise Exception

        return reward

    def compute_reward_assign(self, actions, global_reward=False):
        setA = set(actions)
        if global_reward:
            accuracy = len(setA) / self.n_actions
            reward = accuracy * (self.max_reward - self.min_reward) + self.min_reward

        else:
            reward = []
            for a in actions:
                num_overlap = (a == actions).sum()
                reward.append(self.max_reward if num_overlap == 1 else self.min_reward)

        return reward

    @staticmethod
    def generate_state(n_ag):
        return np.identity(n_ag)

    def close(self):
        return
