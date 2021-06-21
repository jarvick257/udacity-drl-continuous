import pdb
import numpy as np
from unityagents import UnityEnvironment
import torch.multiprocessing as mp

from agent import Agent
from model import Actor, Critic

N = 30
SEQUENCE_SIZE = 1
N_EPOCHS = 10
LR = 0.0005
NUM_GAMES = 100
GAMMA = 1.0


def worker(idx, obs_q, action_q, shared_actor, shared_critic):
    agent = Agent(
        shared_actor,
        shared_critic,
        33,
        4,
        lr=LR,
        gamma=GAMMA,
        sequence_size=SEQUENCE_SIZE,
        n_epochs=N_EPOCHS,
    )
    n_steps = 0
    while True:
        done = False
        obs = obs_q.get()
        while not done:
            action, prob, val = agent.choose_action(obs)
            action_q.put((idx, action))
            obs_, reward, done = obs_q.get()
            if obs_ is None:
                return
            agent.remember(obs, action, prob, val, reward, done)
            n_steps += 1
            if n_steps % N == 0:
                agent.learn()
            obs = obs_


# Env
env = UnityEnvironment("./Reacher_Linux/Reacher.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
num_actions = brain.vector_action_space_size
num_inputs = env_info.vector_observations.shape[1]

# Prepare shared instances
actor = Actor(num_actions, num_inputs)
actor.share_memory()
critic = Critic(num_inputs)
critic.share_memory()

action_q = mp.Queue()
state_qs = [mp.Queue() for _ in range(num_agents)]
workers = [
    mp.Process(target=worker, args=(i, state_qs[i], action_q, actor, critic))
    for i in range(num_agents)
]
[w.start() for w in workers]

# Start workers and keep track of progress
best_score = 0
score_history = []
action = np.zeros((num_agents, num_actions))
for game in range(NUM_GAMES):
    env_info = env.reset(train_mode=True)[brain_name]
    obs = env_info.vector_observations
    [q.put(obs[i]) for i, q in enumerate(state_qs)]
    dones = [False] * num_agents
    score = 0
    while not np.any(dones):
        # Get action from workers
        for _ in range(len(workers)):
            idx, act = action_q.get()
            action[idx] = act
        # Step env
        np.clip(action, -1.0, 1.0)
        env_info = env.step(action)[brain_name]
        rewards = env_info.rewards
        dones = env_info.local_done
        observations = env_info.vector_observations
        # Send new state to workers
        for i, q in enumerate(state_qs):
            q.put((observations[i], rewards[i], dones[i]))
        score += np.mean(rewards)
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    saved = ""
    if avg_score > best_score and game > 20:
        best_score = avg_score
        actor.save_checkpoint()
        critic.save_checkpoint()
        saved = " <-"
    print(f"{game:5d}: score {score:8.1f} - " f"avg {avg_score:8.1f}")
[q.put((None,) * 3) for q in state_qs]
env.close()

# Plot progress (late import because importing matplotlib takes so long)
from utils import plot_learning_curve

x = [i + 1 for i in range(len(score_history))]
plot_learning_curve(x, score_history, "progress.png")
