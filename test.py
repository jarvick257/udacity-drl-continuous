import sys
import gym
from agent import Agent

env = gym.make("Pendulum-v0")

agent = Agent(3, 1)
agent.load_checkpoint(sys.argv[1])

for i in range(50):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.act(observation, add_noise=False)
        observation, reward, done, info = env.step(action)
        score += reward
        env.render()
    print(i + 1, score)
env.close()
