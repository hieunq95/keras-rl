# Solving frozen-lake using Q-learning
# @author: Hieu Nguyen

import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
# env = gym.make('FrozenLake8x8-v0')
env.seed(1000)
print(env.action_space, env.observation_space)
env.reset()
for i in range(10):
    env.step(1)
    env.render()

