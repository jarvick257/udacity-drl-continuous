import random
from collections import deque

import numpy as np


class Experience:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class ReplayBuffer:
    def __init__(self, buffer_size, seed):
        self.seed = random.seed
        self.memory = deque(maxlen=buffer_size)

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None])
        return states, actions, rewards, next_states, dones

    def remember(self, states, actions, rewards, next_states, dones):
        for i in range(states.shape[0]):
            self.memory.append(
                Experience(states[i], actions[i], rewards[i], next_states[i], dones[i])
            )

    def reset(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)
