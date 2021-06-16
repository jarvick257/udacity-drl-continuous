import pdb
import os
import torch as T
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


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


class ActorNetwork(nn.Module):
    def __init__(
        self,
        n_actions,
        inputs,
        lr,
        fc1_dims=256,
        chkptr_dir="tmp/ppo",
    ):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkptr_dir, "actor_torch_ppo")
        self.net = nn.Sequential(
            nn.Linear(inputs, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, n_actions),
            nn.Softmax(dim=-1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x = self.net(state)
        dist = Categorical(x)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, inputs, lr, fc1_dims=256, fc2_dims=256, chkptr_dir="tmp/ppo"):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkptr_dir, "critic_torch_ppo")
        self.critic = nn.Sequential(
            nn.Linear(inputs, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(
        self,
        n_actions,
        inputs,
        gamma=0.99,
        lr=0.0003,
        gae_lambda=0.95,
        policy_clip=0.2,
        sequence_size=64,
        n_epochs=10,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.actor = ActorNetwork(n_actions, inputs, lr).to(self.device)
        self.critic = CriticNetwork(inputs, lr).to(self.device)
        self.memory = PPOMemory(sequence_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        # pdb.set_trace()
        state = T.tensor([observation], dtype=T.float).to(self.device)
        dist = self.actor(state)
        value = self.critic(state)
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
        advantage = T.tensor(advantage).to(self.device)
        values = T.tensor(values).to(self.device)
        for epoch in range(self.n_epochs):
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.device)
                actions = T.tensor(action_arr[batch]).to(self.device)
                dist = self.actor(states)
                critic_value = self.critic(states)
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
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        self.memory.clear_memory()
