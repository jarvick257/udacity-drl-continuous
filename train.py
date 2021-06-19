import argparse
import copy

import numpy as np

import torch as T
from torch.optim import Adam
import torch.multiprocessing as mp

from agent import Agent
from model import PPOModel
from environment import UnityEnv

N = 500
SEQUENCE_SIZE = 100
N_EPOCHS = 10


def train(id_, global_model, device, env):
    agent = Agent(
        shared_model=global_model,
        n_inputs=env.num_inputs,
        n_actions=env.num_actions,
        device=device,
        optimizer=Adam(global_model.parameters(), lr=0.0005),
        sequence_size=SEQUENCE_SIZE,
        n_epochs=N_EPOCHS,
        gamma=0.99,
    )

    n_steps = 0
    while True:
        observation = env.reset()
        if observation is None:
            break
        done = False
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            agent.remember(observation, action, prob, val, reward, done)
            n_steps += 1
            if n_steps % N == 0 or done:
                agent.learn()
            observation = observation_


if __name__ == "__main__":
    # CLI Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", "-n", default=mp.cpu_count(), type=int)
    parser.add_argument("--num_games", "-g", default=1200, type=int)
    parser.add_argument("--device", "-d", default="cpu", type=str)
    parser.add_argument("--lr", default=0.0005, type=float)
    args = parser.parse_args()

    # Env
    unity = UnityEnv("./Reacher_Linux/Reacher.x86_64")

    # Device
    if args.device != "cpu":
        assert T.cuda.is_available()
    device = T.device(args.device)

    # Prepare shared instances
    model = PPOModel(n_actions=unity.num_actions, n_inputs=unity.num_inputs)
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
                unity.get_env(),
            ),
        )
        for i in range(args.num_workers)
    ]

    # Start workers and keep track of progress
    best_score = 0
    score_history = []
    [w.start() for w in workers]
    for game in range(args.num_games):
        score = unity.run()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        saved = ""
        if avg_score > best_score and game > 20:
            best_score = avg_score
            model.save_checkpoint()
            saved = " <-"
        print(f"{game:5d}: score {score:8.1f} - avg {avg_score:8.1f}{saved}")

    # Stop workers
    unity.stop()
    [w.join() for w in workers]

    # Plot progress (late import because importing matplotlib takes so long)
    from utils import plot_learning_curve

    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, "progress.png")
