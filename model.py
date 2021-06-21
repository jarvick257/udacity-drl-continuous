import pdb
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class Actor(nn.Module):
    def __init__(
        self,
        n_actions,
        n_inputs,
        chkptr_dir="tmp/ppo",
    ):
        super(Actor, self).__init__()
        self.checkpoint_file = os.path.join(chkptr_dir, "actor_torch_ppo")

        self.act1 = nn.Linear(n_inputs, 64)
        self.mu = nn.Linear(64, n_actions)
        self.sigma = nn.Linear(64, n_actions)

    def forward(self, state):
        x = F.relu(self.act1(state))
        mu = F.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x))
        dist = Normal(mu, sigma)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Critic(nn.Module):
    def __init__(
        self,
        n_inputs,
        chkptr_dir="tmp/ppo",
    ):
        super(Critic, self).__init__()
        self.checkpoint_file = os.path.join(chkptr_dir, "critic_torch_ppo")

        self.crit1 = nn.Linear(n_inputs, 128)
        self.crit2 = nn.Linear(128, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.crit1(state))
        x = F.relu(self.crit2(x))
        value = self.value(x)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
