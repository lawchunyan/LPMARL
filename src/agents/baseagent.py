import torch
import torch.nn as nn
from src.utils.replaymemory import ReplayMemory


class BaseAgent(nn.Module):
    def __init__(self, state_dim, action_dim, memory_len, batch_size, train_start, gamma=0.99):
        super(BaseAgent, self).__init__()

        self.memory = ReplayMemory(memory_len)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.train_start = train_start
        self.gamma = gamma
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_action(self, *args):
        raise NotImplementedError

    def can_fit(self):
        return True if len(self.memory) > self.train_start else False

    def push(self, *args):
        self.memory.push(*args)

    def save(self):
        torch.save(self.state_dict(), 'happy.th')
