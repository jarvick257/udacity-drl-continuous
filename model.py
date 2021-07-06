import pdb
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    def __init__(self, n_inputs, n_actions, seed):
        super(Actor, self).__init__()
        self.seed = T.manual_seed(seed)

        self.fc1 = nn.Linear(n_inputs, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softsign(self.fc3(x))

    def save_checkpoint(self, path):
        T.save(self.state_dict(), path)

    def load_checkpoint(self, path, device=T.device("cpu")):
        self.load_state_dict(T.load(path, map_location=device))


class Critic(nn.Module):
    def __init__(self, n_inputs, n_actions, seed):
        super(Critic, self).__init__()
        self.seed = T.manual_seed(seed)

        self.fcs1 = nn.Linear(n_inputs, 256)
        self.fc2 = nn.Linear(256 + n_actions, 128)
        self.fc3 = nn.Linear(128, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        xs = F.relu(self.fcs1(state))
        x = T.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save_checkpoint(self, path):
        T.save(self.state_dict(), path)

    def load_checkpoint(self, path, device=T.device("cpu")):
        self.load_state_dict(T.load(path, map_location=device))
