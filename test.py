import gym
from model import PPOModel
from agent import Agent


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")

    model = PPOModel(
        n_actions=env.action_space.n,
        n_inputs=env.observation_space.shape[0],
    )
    model.load_checkpoint()
    model.eval()
    agent = Agent(model, optimizer=None)

    for i in range(50):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, _, _ = agent.choose_action(observation)
            observation, reward, done, info = env.step(action)
            score += reward
            env.render()
        print(i + 1, score)
    env.close()
