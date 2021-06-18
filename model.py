import pdb
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class PPOModel(nn.Module):
    def __init__(
        self,
        n_actions,
        n_inputs,
        chkptr_dir="tmp/ppo",
    ):
        super(PPOModel, self).__init__()
        self.checkpoint_file = os.path.join(chkptr_dir, "actor_torch_ppo")
        hidden_dims = 128
        self.fc1 = nn.Linear(n_inputs, 256)
        self.fc2 = nn.Linear(256, hidden_dims)
        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(hidden_dims, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        dist = F.softmax(self.actor(x), dim=1)
        dist = Categorical(dist)

        x = F.relu(self.fc2(x))
        val = self.critic(x)
        return dist, val

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
