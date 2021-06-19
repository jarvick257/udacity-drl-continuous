import copy

import gym
import numpy as np
import matplotlib.pyplot as plt

import torch.multiprocessing as mp
import torch as T
from torch.optim import Adam

from model import PPOModel
from agent import Agent
from utils import plot_learning_curve


def train(id_, model, device, env, n_games, report_q):
    N = 20

    optim = Adam(model.parameters(), lr=0.0005)
    agent = Agent(
        model=model,
        device=device,
        optimizer=optim,
        sequence_size=5,
        n_epochs=4,
        gamma=0.99,
    )

    learn_iters = 0
    n_steps = 0
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        report_q.put((id_, i, score))
    report_q.put((None, None, None))


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    n_games = 1200
    n_workers = 8

    device = T.device("cpu" if not T.cuda.is_available() else "cuda:0")

    model = PPOModel(
        n_actions=env.action_space.n,
        n_inputs=env.observation_space.shape[0],
    )
    model.to(device)
    model.share_memory()
    report_q = mp.Queue()

    workers = [
        mp.Process(
            target=train,
            args=(i, model, device, copy.deepcopy(env), n_games // n_workers, report_q),
        )
        for i in range(n_workers)
    ]

    best_score = env.reward_range[0]
    avg_score = 0
    score_history = []
    [w.start() for w in workers]
    total_eps = 0
    while n_workers > 0:
        id_, game, score = report_q.get()
        if id_ is None:
            n_workers -= 1
            continue
        total_eps += 1
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        saved = ""
        if avg_score > best_score:
            best_score = avg_score
            model.save_checkpoint()
            saved = " <-"
        print(
            f"{total_eps}: {id_} - {game} - score {score:0.1f} - avg {avg_score:0.1f}{saved}"
        )
    [w.join() for w in workers]
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, "progress.png")
