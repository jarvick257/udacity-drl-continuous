import numpy as np


class PPOMemory:
    def __init__(self, sequence_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.sequence_size = sequence_size

    def generate_batches(self):
        n_sequences = len(self.states) // self.sequence_size
        indices = np.arange(n_sequences * self.sequence_size, dtype=np.int64)
        sequences = indices.reshape([-1, self.sequence_size])
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
            sequences,
        )

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []


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
