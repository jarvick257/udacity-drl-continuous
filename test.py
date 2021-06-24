import time
import sys
from unityagents import UnityEnvironment
from agent import Agent

env = UnityEnvironment("Reacher_Linux_single/Reacher.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
num_actions = brain.vector_action_space_size
num_inputs = env_info.vector_observations.shape[1]

seed = int(time.time() * 1000 % 1000)
agent = Agent(
    n_inputs=num_inputs,
    n_actions=num_actions,
    n_agents=num_agents,
    random_seed=seed,
)
agent.load_checkpoint(sys.argv[1])

for i in range(10):
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations
    done = False
    score = 0
    while not done:
        action = agent.act(state, add_noise=False)
        env_info = env.step(action)[brain_name]
        state = env_info.vector_observations
        score += env_info.rewards[0]
        done = env_info.local_done[0]
    print(i + 1, score)
env.close()
