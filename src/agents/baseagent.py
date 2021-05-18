import torch
import torch.nn as nn
from src.utils.replaymemory import ReplayMemory, ReplayMemory_episode
from datetime import date


class BaseAgent(nn.Module):
    def __init__(self, state_dim, action_dim, memory_len, batch_size, train_start, gamma=0.99, memory_type="sample",
                 name=None):
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

    def save(self, e):
        torch.save(self.state_dict(), 'result/{}/{}.th'.format(date.today().strftime("%Y%m%d") + "_" + self.name, e))

    @staticmethod
    def update_target_network(target_params, source_params, tau=1.0):
        for t, s in zip(target_params, source_params):
            t.data.copy_(tau * s.data + (1.0 - tau) * t.data)


if __name__ == '__main__':
    A = torch.nn.Linear(6, 7)
    B = torch.nn.Linear(6, 7)
