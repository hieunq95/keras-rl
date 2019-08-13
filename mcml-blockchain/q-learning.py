# Q-learning agent for MCML-Block chain
# @author: Hieu Nguyen

import numpy as np
import gym
import matplotlib.pyplot as plt
import xlsxwriter
import random
from environment import Environment
from writer_v1 import MCMLWriter

mempool = []
workbook = xlsxwriter.Workbook('./build/qlearning-result.xlsx')
writer = MCMLWriter(workbook)

env = Environment(mempool, writer)
env.reset()
print(env.observation_space, env.action_space)

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1
# Initialize q-table
state_space_size = 2560  # 4 x 4 x 4 x 4 x 10
action_space_size = 1024  # 4 x 4 x 4 x 4 x 4
q_table = np.zeros((state_space_size, action_space_size))
print(q_table.shape)

# Training parameters
NB_EPISODES = 1000

for i in range(NB_EPISODES):
    state = env.reset()
    episode_reward = 0

    episode, reward = 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state,action] = new_value

        state = next_state

        episode_reward += reward

        print('Episode: {}, Ep_reward: {},'.format(episode, reward))


