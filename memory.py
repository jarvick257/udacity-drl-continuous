import pdb
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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))

    def reset(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)


if __name__ == "__main__":
    import pdb
    from unityagents import UnityEnvironment

    env = UnityEnvironment("./Reacher_Linux/Reacher.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    states = []
    for i in range(10):
        env_info = env.reset(train_mode=True)[brain_name]
        states.append(env_info.vector_observations)
    pdb.set_trace()
