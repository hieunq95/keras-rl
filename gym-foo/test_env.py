import gym
import gym_foo
import random as random
import numpy as np
import matplotlib.pyplot as plt

# ENV_NAME = 'OneRoundNondeterministicReward-v0'
ENV_NAME = 'mcml-v0'
env = gym.make(ENV_NAME)
print(env.observation_space, env.action_space, env.action_space.nvec)
env.reset()

nb_actions = 1
for i in env.action_space.nvec:
    nb_actions = nb_actions * i
# Test step()
for _ in range(5):
    state, reward, done, info = env.step(env.action_space.sample())
    env.reset()
    print(state, reward, done, info)

print(nb_actions)
nprandom = np.random.RandomState()
reset_state = nprandom.randint(low=0, high=10, size=env.observation_space.shape)
print(reset_state)



