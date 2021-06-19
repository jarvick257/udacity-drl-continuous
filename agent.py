import torch as T
import numpy as np

from memory import PPOMemory
from model import PPOModel


class Agent:
    def __init__(
        self,
        shared_model,
        n_inputs,
        n_actions,
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
        self.shared_model = shared_model
        self.model = PPOModel(n_actions=n_actions, n_inputs=n_inputs)
        self.model.to(self.device)
        self.optim = optimizer
        self.memory = PPOMemory(sequence_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.device)
        dist, value = self.model(state)
        action = dist.sample()
        action = T.clamp(action, -1.0, 1.0)
        action = T.squeeze(action).detach().cpu().numpy()
        action_prob = T.squeeze(dist.log_prob(action)).item()
        value = T.squeeze(value).item().cpu().numpy()
        return action, action_prob, value

    def sync_model(self):
        self.model.load_state_dict(self.shared_model.state_dict())

    def share_grads(self):
        for param, shared_param in zip(
            self.model.parameters(), self.shared_model.parameters()
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
                prob_ratio = (new_probs.exp() / old_probs.exp()).mean(dim=1)
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
                self.share_grads()
                self.optim.step()
                self.sync_model()
        self.memory.clear_memory()
