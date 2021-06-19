import sys
import numpy as np
from ppo import Agent
import matplotlib.pyplot as plt
from utils import plot_learning_curve

from unityagents import UnityEnvironment

try:
    render_interval = int(sys.argv[1])
except IndexError:
    render_interval = 0

env = UnityEnvironment("./Reacher_Linux/Reacher.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# number of agents
num_agents = len(env_info.agents)
print("Number of agents:", num_agents)
# size of each action
action_size = brain.vector_action_space_size
print("Size of each action:", action_size)
# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print(
    "There are {} agents. Each observes a state with length: {}".format(
        states.shape[0], state_size
    )
)
print("The state for the first agent looks like:", states[0])

N = 100
sequence_size = 50
n_epochs = 10
alpha = 0.0001
agent = Agent(
    inputs=state_size,
    n_actions=action_size,
    sequence_size=sequence_size,
    lr=alpha,
    n_epochs=n_epochs,
    gamma=1.0,
)

n_games = 300
best_score = 0
score_history = []

learn_iters = 0
avg_score = 0
n_steps = 0

for i in range(n_games):
    score = 0
    done = False
    env_info = env.reset(train_mode=True)[brain_name]
    observation = env_info.vector_observations[0]

    while not done:
        action, prob, val = agent.choose_action(observation)
        env_info = env.step(action)[brain_name]
        observation_ = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        agent.remember(observation, action, prob, val, reward, done)
        n_steps += 1
        score += reward
        if n_steps % N == 0:
            # agent.learn()
            learn_iters += 1
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()

    print(
        f"Eps {i}, score {score:0.1f}, Avg Score {avg_score:0.1f}, time steps {n_steps}, learning_steps {learn_iters}"
    )

env.close()
x = [i + 1 for i in range(len(score_history))]
plot_learning_curve(x, score_history, "progress.png")
