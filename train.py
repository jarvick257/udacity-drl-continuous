import argparse
import copy

import gym
import numpy as np

import torch as T
from torch.optim import Adam
import torch.multiprocessing as mp

from agent import Agent
from model import PPOModel


def train(id_, global_model, device, env, n_games, report_q):
    N = 20
    optimizer = Adam(global_model.parameters(), lr=0.0005)
    agent = Agent(
        n_inputs=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        device=device,
        optimizer=optimizer,
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
            agent.remember(observation, action, prob, val, reward, done)
            n_steps += 1
            score += reward
            if n_steps % N == 0 or done:
                agent.learn(global_model)
                learn_iters += 1
            observation = observation_
        report_q.put((id_, i, score))
    report_q.put((None, None, None))


if __name__ == "__main__":
    # CLI Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", "-n", default=mp.cpu_count(), type=int)
    parser.add_argument("--num_games", "-g", default=1200, type=int)
    parser.add_argument("--device", "-d", default="cpu", type=str)
    parser.add_argument("--lr", default=0.0005, type=float)
    args = parser.parse_args()

    # Env
    env = gym.make("LunarLander-v2")

    # Device
    if args.device != "cpu":
        assert T.cuda.is_available()
    device = T.device(args.device)

    # Prepare shared instances
    model = PPOModel(
        n_actions=env.action_space.n,
        n_inputs=env.observation_space.shape[0],
    )
    model.to(device)
    model.share_memory()

    # Set up workers
    report_q = mp.Queue()
    workers = [
        mp.Process(
            target=train,
            args=(
                i,
                model,
                device,
                copy.deepcopy(env),
                args.num_games // args.num_workers,
                report_q,
            ),
        )
        for i in range(args.num_workers)
    ]

    # Start workers and keep track of progress
    best_score = env.reward_range[0]
    avg_score = 0
    score_history = []
    [w.start() for w in workers]
    total_eps = 0
    active_workers = args.num_workers
    while active_workers > 0:
        id_, game, score = report_q.get()
        if id_ is None:
            active_workers -= 1
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
            f"{total_eps:5d}: {id_:2d} - {game:4d} - score {score:8.1f} - avg {avg_score:8.1f}{saved}"
        )

    # Stop workers
    [w.join() for w in workers]

    # Plot progress (late import because importing matplotlib takes so long)
    from utils import plot_learning_curve

    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, "progress.png")
