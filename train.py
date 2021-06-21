import pdb
import gym
import numpy as np

from agent import Agent
from utils import plot_learning_curve

seed = 42
num_games = 1000
max_t = 300

env = gym.make("Pendulum-v0")
# env.seed(seed)

agent = Agent(n_inputs=3, n_actions=1)

scores = []
best_score = -np.inf
for i in range(num_games):
    state = env.reset()
    agent.reset()
    score = 0
    done = False
    t = 0
    while not done and t < max_t:
        action = agent.act(state)
        state_, reward, done, _ = env.step(action)
        agent.step(state, action, reward, state_, done)
        state = state_
        score += reward
        t += 1
    scores.append(score)
    avg_score = np.mean(scores[-100:])
    print(f"Eps {i+1:5d}: score {score:10.2f}, avg {avg_score:10.2f}")
    if avg_score > best_score:
        agent.save_checkpoint("tmp")
        best_score = avg_score

plot_learning_curve(np.arange(len(scores)), scores, "tmp/progress.png")
