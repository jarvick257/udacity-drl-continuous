import sys
import gym
import numpy as np
from ppo import Agent
import matplotlib.pyplot as plt
from utils import plot_learning_curve


try:
    render_interval = int(sys.argv[1])
except IndexError:
    render_interval = 0

# env = gym.make("CartPole-v0")
env = gym.make("LunarLander-v2")
N = 20
memory_size = 50
sequence_size = 5
n_epochs = 4
alpha = 0.0005
agent = Agent(
    n_actions=env.action_space.n,
    sequence_size=sequence_size,
    lr=alpha,
    n_epochs=n_epochs,
    inputs=env.observation_space.shape[0],
)

n_games = 300
best_score = env.reward_range[0]
score_history = []

learn_iters = 0
avg_score = 0
n_steps = 0

for i in range(n_games):
    if i == n_games - 1:
        render = True
    elif render_interval > 0:
        render = (i + 1) % render_interval == 0
    else:
        render = False
    observation = env.reset()
    if render:
        env.render()
    done = False
    score = 0
    while not done:
        action, prob, val = agent.choose_action(observation)
        if render:
            env.render()
        observation_, reward, done, info = env.step(action)
        n_steps += 1
        score += reward
        agent.remember(observation, action, prob, val, reward, done)
        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    if render:
        env.close()

    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()

    print(
        f"Eps {i}, score {score:0.1f}, Avg Score {avg_score:0.1f}, time steps {n_steps}, learning_steps {learn_iters}"
    )

x = [i + 1 for i in range(len(score_history))]
plot_learning_curve(x, score_history, "progress.png")
