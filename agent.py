import pdb
import torch as T
import numpy as np
from torch.optim import Adam

from memory import PPOMemory
from model import Actor, Critic


class Agent:
    def __init__(
        self,
        shared_actor,
        shared_critic,
        n_inputs,
        n_actions,
        lr=0.0005,
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

        # self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.device = T.device("cpu")
        self.local_actor = Actor(n_actions=n_actions, n_inputs=n_inputs)
        self.optim_actor = Adam(self.local_actor.parameters(), lr=lr)
        self.local_critic = Critic(n_inputs=n_inputs)
        self.optim_critic = Adam(self.local_critic.parameters(), lr=lr)
        self.memory = PPOMemory(sequence_size)
        self.shared_actor = shared_actor
        self.shared_critic = shared_critic

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        dist = self.local_actor(state)
        value = self.local_critic(state)
        action = dist.sample()
        action_prob = T.sum(dist.log_prob(action)).item()
        action = action.detach().cpu().numpy()
        value = T.squeeze(value).item()
        return action, action_prob, value

    def sync_model(self):
        self.local_actor.load_state_dict(self.shared_actor.state_dict())
        self.local_critic.load_state_dict(self.shared_critic.state_dict())

    def share_grads(self):
        for param, shared_param in zip(
            self.local_actor.parameters(), self.shared_actor.parameters()
        ):
            shared_param._grad = param.grad
        for param, shared_param in zip(
            self.local_critic.parameters(), self.shared_critic.parameters()
        ):
            shared_param._grad = param.grad

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
        for t in range(len(reward_arr) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr) - 1):
                a_t += discount * (
                    reward_arr[k]
                    + self.gamma * values[k + 1] * (1 - done_arr[k])
                    - values[k]
                )
                discount *= self.gamma * self.gae_lambda
            advantage[t] = a_t
        advantage = T.tensor(advantage)
        values = T.tensor(values, dtype=T.float32)
        # advantage = (advantage - T.mean(advantage)) / T.std(advantage)
        for epoch in range(self.n_epochs):
            for batch in batches:
                states = T.tensor(state_arr[batch]).to(self.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.device)
                actions = T.tensor(action_arr[batch]).to(self.device)
                dist = self.local_actor(states)
                critic_value = self.local_critic(states)
                critic_value = T.squeeze(critic_value)
                new_probs = T.sum(dist.log_prob(actions))
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = prob_ratio * advantage[batch]
                weighted_clipped_probs = (
                    T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * advantage[batch]
                )
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5 * critic_loss
                self.optim_actor.zero_grad()
                self.optim_critic.zero_grad()
                total_loss.backward()
                self.share_grads()
                self.optim_actor.step()
                self.optim_critic.step()
                self.sync_model()
        self.memory.clear_memory()
