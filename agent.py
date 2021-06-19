import pdb
import os
import torch as T
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from memory import PPOMemory


class Agent:
    def __init__(
        self,
        model,
        optimizer,
        device,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        sequence_size=64,
        n_epochs=10,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs

        self.device = device
        self.model = model
        self.optim = optimizer
        self.memory = PPOMemory(sequence_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        # pdb.set_trace()
        state = T.tensor([observation], dtype=T.float).to(self.device)
        dist, value = self.model(state)
        action = dist.sample()
        action_prob = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()
        return action, action_prob, value

    def learn(self):
        (
            state_arr,
            action_arr,
            old_prob_arr,
            val_arr,
            reward_arr,
            done_arr,
            batches,
        ) = self.memory.generate_batches()
        values = val_arr
        advantage = np.zeros(len(reward_arr), dtype=np.float32)
        # pdb.set_trace()
        for t in range(len(reward_arr) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr) - 1):
                a_t += discount * (
                    reward_arr[k]
                    + self.gamma * values[k + 1] * (1 - int(done_arr[k]))
                    - values[k]
                )
                discount *= self.gamma * self.gae_lambda
            advantage[t] = a_t
        advantage = T.tensor(advantage)
        values = T.tensor(values)
        for epoch in range(self.n_epochs):
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.device)
                actions = T.tensor(action_arr[batch]).to(self.device)
                dist, critic_value = self.model(states)
                critic_value = T.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = (
                    T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * advantage[batch]
                )
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5 * critic_loss
                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()
        self.memory.clear_memory()
