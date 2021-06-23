import pdb
import numpy as np

from unityagents import UnityEnvironment

from agent import Agent
from utils import plot_learning_curve

seed = 123
num_games = 1000

env = UnityEnvironment("Reacher_Linux_single/Reacher.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
num_actions = brain.vector_action_space_size
num_inputs = env_info.vector_observations.shape[1]

agent = Agent(n_inputs=num_inputs, n_actions=num_actions, random_seed=seed)

scores = []
avg_over_30 = 0
best_score = -np.inf
game = 0
try:
    while avg_over_30 < 100:
        game += 1
        score = 0
        t = 0
        done = False
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        agent.reset()
        while not done:
            action = agent.act(state, add_noise=True)
            env_info = env.step(action)[brain_name]
            state_ = env_info.vector_observations
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, state_, done)
            state = state_
            score += reward
            t += 1
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        if avg_score >= 30.0:
            avg_over_30 += 1
        else:
            avg_over_30 = 0
        print(
            f"Eps {game:5d}: theta: {agent.noise.theta:0.2f}, score {score:6.2f}, avg {avg_score:6.2f}, num over 30: {avg_over_30}"
        )
        if avg_score > best_score and game > 10:
            agent.save_checkpoint("tmp")
            best_score = avg_score
except KeyboardInterrupt:
    pass

env.close()
plot_learning_curve(np.arange(len(scores)), scores, "tmp/progress.png")
