import torch.nn as nn
import torch

from torch.optim import Adam
from src.utils.agent_utils import update_target_network
from src.utils.replaymemory import ReplayMemory
from src.agents.commnet.network import CommNetWork_Actor, CommNetWork_Critic
from collections import namedtuple

Transition_base = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminated'))


class CommnetAgent(nn.Module):
    def __init__(self, n_ag, state_dim, action_dim, batch_size, lr=0.0001, gamma=0.99):
        super(CommnetAgent, self).__init__()

        self.memory = ReplayMemory(10000)
        self.memory.transition = Transition_base
        self.noise = 1.0
        self.batch_size = batch_size
        self.gamma = gamma

        self.actor = CommNetWork_Actor(n_ag, state_dim, action_dim)
        self.actor_target = CommNetWork_Actor(n_ag, state_dim, action_dim)

        self.critic = CommNetWork_Critic(n_ag, state_dim, action_dim)
        self.critic_target = CommNetWork_Critic(n_ag, state_dim, action_dim)

        self.actor_optimizer = Adam(self.actor.parameters(), lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr)

        self.epsilon = 1.0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        update_target_network(self.actor_target.parameters(), self.actor.parameters())
        update_target_network(self.actor_target.parameters(), self.actor.parameters())

        self.n_fit = 0
        self.n_ag = n_ag

    def get_action(self, state, explore=True):
        tensor_state = torch.Tensor(state)
        action = self.actor(tensor_state)
        if explore:
            action = action + torch.rand_like(action) * self.epsilon
            action = action.clip(-1, 1)

            if self.epsilon > 0.05:
                self.epsilon -= 1e-6

        return action.cpu().detach().numpy()

    def can_fit(self):
        return len(self.memory) > self.batch_size

    def fit(self):
        samples = self.memory.sample(self.batch_size)
        s = []
        a = []
        r = []
        ns = []
        t = []
        l = [s, a, r, ns, t]
        for sample in samples:
            for ss, lst in zip(sample, l):
                lst.append(ss)

        state_batch = torch.Tensor(s).to(self.device)
        action_batch = torch.Tensor(a).to(self.device)
        reward_batch = torch.Tensor(r).to(self.device)
        next_state_batch = torch.Tensor(ns).to(self.device)
        terminated_batch = torch.Tensor(t).int().to(self.device)

        current_q = self.critic(state_batch, action_batch).squeeze()
        with torch.no_grad():
            next_action = self.actor_target(next_state_batch)
            target_q = self.critic_target(next_state_batch, next_action).squeeze()

        q_target = reward_batch + self.gamma * target_q * (1 - terminated_batch)
        critic_loss = ((current_q - q_target) ** 2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        action = self.actor(state_batch).reshape(self.batch_size, self.n_ag, -1)
        actor_loss = -self.critic_target(state_batch, action).mean() * 100
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()

        ret_dict = {'actor_loss': actor_loss.item(),
                    'critic_loss': critic_loss.item()}

        self.n_fit += 1

        if self.n_fit % 200 == 0:
            update_target_network(self.actor_target.parameters(), self.actor.parameters(), tau=0.001)
            update_target_network(self.actor_target.parameters(), self.actor.parameters(), tau=0.001)

        return ret_dict

    def push(self, *args):
        self.memory.push(*args)


if __name__ == '__main__':
    agent = CommnetAgent()
