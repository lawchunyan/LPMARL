import itertools

from collections import namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'high_action', 'action', 'reward', 'next_state', 'terminated', 'avail_action'))
Transition_base = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminated', 'avail_action'))


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


class ReplayMemory_episode(object):
    def __init__(self, capacity, max_ep_len=120):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.max_ep_len = max_ep_len
        self.transition = Transition

        self.ep_transitions = []

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        # self.memory[self.position] = self.transition(*args)
        # self.position = (self.position + 1) % self.capacity

        curr_sample = self.transition(*args)
        self.ep_transitions.append(curr_sample)

        if curr_sample.terminated:
            self.push_episodes()

    def push_episodes(self):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.ep_transitions

        self.position = (self.position + 1) % self.capacity
        self.ep_transitions = []

    def sample(self, batch_size):
        episode_samples = random.sample(self.memory, batch_size)

        return list(itertools.chain(*episode_samples))
