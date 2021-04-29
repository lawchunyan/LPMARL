from collections import namedtuple
import random

Transition = namedtuple('Transition', ('state', 'high_action', 'action', 'reward', 'next_state', 'terminated', 'avail_action', 'logit_h'))

# Transition_g = namedtuple('Transition', ('graph', 'next_graph', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.transition = Transition

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
