import pdb
import numpy as np

from unityagents import UnityEnvironment

from agent import Agent
from utils import plot_learning_curve

seed = 123
num_games = 1000

env = UnityEnvironment("Reacher_Linux_multi/Reacher.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
num_actions = brain.vector_action_space_size
num_inputs = env_info.vector_observations.shape[1]

agent = Agent(
    n_inputs=num_inputs,
    n_actions=num_actions,
    n_agents=num_agents,
    random_seed=seed,
)

scores = []
thetas = []
avg_over_30 = 0
best_score = -np.inf
game = 0
try:
    while avg_over_30 < 100:
        game += 1
        score = 0
        t = 0
        dones = [False] * num_agents
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        while not np.any(dones):
            action = agent.act(states, add_noise=True)
            env_info = env.step(action)[brain_name]
            states_ = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, action, rewards, states_, dones)
            states = states_
            score += np.mean(rewards)
            t += 1
            print(t, end="\r")
        scores.append(score)
        thetas.append(agent.noise.theta)
        if score > 30.0:
            avg_over_30 += 1
        else:
            avg_over_30 = 0
        print(
            f"Eps {game:5d}: theta: {agent.noise.theta:0.2f}, score {score:6.2f}, games over 30: {avg_over_30}"
        )
        avg = np.mean(scores[-10:])
        if avg > best_score and game > 10:
            agent.save_checkpoint("checkpoints")
            best_score = avg
except KeyboardInterrupt:
    pass

env.close()
plot_learning_curve(
    scores, thetas, "checkpoints/progress.png", "Learning Progess (Avg over agents)"
)
