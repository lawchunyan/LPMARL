import numpy as np


class fake_env():
    def __init__(self, n_ag=10, n_en=10, state_dim=15, enemy_obs=False):
        self.state_dim = 15
        self.n_ag = n_ag
        self.n_en = n_en
        self.enemy_obs = enemy_obs
        self.n_actions = 6 + n_en

    @staticmethod
    def reset():
        return

    def get_state(self):
        return np.random.random(())

    def get_avail_actions(self):
        return np.random.randint(2, size=(self.n_ag, self.n_actions))

    def get_obs(self):
        if self.enemy_obs:
            out_size = self.n_ag + self.n_en
        else:
            out_size = self.n_ag
        return np.random.random((out_size, self.state_dim))

    def step(self):
        reward = np.random.random()
        terminated = False
        info = dict()
        return reward, terminated, info

    def get_state_size(self):
        return 3 * self.n_en + (4 + self.n_actions) * self.n_ag

    def get_obs_size(self):
        return self.state_dim
