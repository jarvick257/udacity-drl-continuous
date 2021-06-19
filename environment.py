import numpy as np
from torch.multiprocessing import Queue
from unityagents import UnityEnvironment


class Environment:
    def __init__(self, idx, action_q, state_q, num_inputs, num_actions):
        self.idx = idx
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.action_q = action_q
        self.state_q = state_q

    def reset(self):
        return self.state_q.get()

    def step(self, action):
        self.action_q.put((self.idx, action))
        return self.state_q.get()


class UnityEnv:
    def __init__(self, path):
        self.env = UnityEnvironment(path)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.num_agents = len(env_info.agents)
        self.num_actions = self.brain.vector_action_space_size
        self.num_inputs = env_info.vector_observations.shape[1]

        self.num_workers = 0
        self.action_q = Queue()
        self.state_qs = [Queue() for _ in range(self.num_agents)]
        self.action = np.zeros((self.num_agents, self.num_actions))

    def get_env(self):
        idx = self.num_workers
        self.num_workers += 1
        return Environment(
            idx, self.action_q, self.state_qs[idx], self.num_inputs, self.num_actions
        )

    def run(self):
        done = False
        score = 0
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        observations = env_info.vector_observations
        for q, obs in zip(self.state_qs, observations):
            q.put(obs)
        while not done:
            for _ in range(self.num_workers):
                i, action = self.action_q.get()
                self.action[i, :] = action
            env_info = self.env.step(self.action)[self.brain_name]
            observations = env_info.vector_observations
            rewards = env_info.rewards
            done = env_info.local_done[0]
            for i, q in enumerate(self.state_qs):
                q.put((observations[i], rewards[i], done))
            score += np.mean(rewards[: self.num_workers])
        return score

    def stop(self):
        [q.put(None) for q in self.state_qs]
        self.env.close()
