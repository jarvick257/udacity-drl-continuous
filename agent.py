import os
import pdb
import time
import torch as T
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam

from memory import ReplayBuffer
from noise import OUNoise
from model import Actor, Critic

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 256
LEARN_EVERY = 10
GAMMA = 0.90
TAU = 1e-2
LR_ACTOR = 0.0001
LR_CRITIC = 0.0001


class Agent:
    def __init__(self, n_inputs, n_actions, n_agents, random_seed):
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        # Actor
        self.target_actor = Actor(n_inputs, n_actions, random_seed).to(self.device)
        self.local_actor = Actor(n_inputs, n_actions, random_seed).to(self.device)
        self.optim_actor = Adam(self.local_actor.parameters(), lr=LR_ACTOR)

        # Critic
        self.target_critic = Critic(n_inputs, n_actions, random_seed).to(self.device)
        self.local_critic = Critic(n_inputs, n_actions, random_seed).to(self.device)
        self.optim_critic = Adam(self.local_critic.parameters(), lr=LR_CRITIC)

        self.memory = ReplayBuffer(BUFFER_SIZE, random_seed)
        self.noise = OUNoise((n_agents, n_actions), random_seed)
        self.num_steps = 0

    def step(self, states, actions, rewards, next_states, dones):
        self.num_steps += 1
        self.memory.remember(states, actions, rewards, next_states, dones)
        if len(self.memory) > BATCH_SIZE and self.num_steps % LEARN_EVERY == 0:
            for _ in range(states.shape[0]):
                experiences = self.memory.sample(BATCH_SIZE)
                self.learn(experiences)

    def act(self, states, add_noise=True):
        states = T.from_numpy(states).float().to(self.device)
        self.local_actor.eval()
        with T.no_grad():
            action = self.local_actor(states).cpu().data.numpy()
        self.local_actor.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def save_checkpoint(self, path):
        print("Saving checkpoint...")
        self.local_actor.save_checkpoint(os.path.join(path, "actor.pth"))
        self.local_critic.save_checkpoint(os.path.join(path, "critic.pth"))

    def load_checkpoint(self, path):
        print("Loading checkpoint...")
        self.local_actor.load_checkpoint(
            os.path.join(path, "actor.pth"), device=self.device
        )
        self.local_critic.load_checkpoint(
            os.path.join(path, "critic.pth"), device=self.device
        )

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma=GAMMA):
        states, actions, rewards, next_states, dones = experiences
        states = T.tensor(states, dtype=T.float).to(self.device)
        actions = T.tensor(actions, dtype=T.float).to(self.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.device)
        next_states = T.tensor(next_states, dtype=T.float).to(self.device)
        dones = T.tensor(dones.astype(np.float32)).to(self.device)

        action_next = self.target_actor(next_states)
        Q_targets_next = self.target_critic(next_states, action_next)

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.local_critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()

        actions_pred = self.local_actor(states)
        actor_loss = -self.local_critic(states, actions_pred).mean()
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()

        self.soft_update(self.local_critic, self.target_critic)
        self.soft_update(self.local_actor, self.target_actor)

    def soft_update(self, local, target, tau=TAU):
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )
