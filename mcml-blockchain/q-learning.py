# Q-learning agent for MCML-Block chain
# @author: Hieu Nguyen

import numpy as np
import gym
import matplotlib.pyplot as plt
import xlsxwriter
import random
from environment import Environment, MyProcessor
from writer_v1 import MCMLWriter
from config import MEMPOOL_MAX, CPU_SHARES, CAPACITY_MAX, ENERGY_MAX, DATA_MAX, MINING_RATE, NB_DEVICES

mempool = []
workbook = xlsxwriter.Workbook('./build/qlearning-result.xlsx')
writer = MCMLWriter(workbook)

env = Environment(mempool, writer)
env.reset()
processor = MyProcessor()
print(env.observation_space, env.action_space)

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1
# Initialize q-table
state_space_size = (CPU_SHARES ** NB_DEVICES) * (CAPACITY_MAX ** NB_DEVICES) * MEMPOOL_MAX  # 4 x 4 x 4 x 4 x MEMPOOL_MAX
action_space_size = (ENERGY_MAX ** NB_DEVICES) * (DATA_MAX ** NB_DEVICES) * MINING_RATE  # 4 x 4 x 4 x 4 x 4
q_table = np.zeros((state_space_size, action_space_size))
print(q_table.shape)

# Training parameters
NB_EPISODES = 10000
episode = 0
for i in range(NB_EPISODES):
    state = env.reset()
    episode_reward = 0

    steps, reward = 0, 0
    done = False
    # print('Ep: {}, q_table shape: {}, state: {}'.format(episode+1, q_table.shape, state))
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
            if action > action_space_size:
                print('invalid action: {}'.format(action))
            action = processor.process_action(action)

        next_state, reward, done, info = env.step(action)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state
        episode_reward += reward
        steps += 1

    episode += 1

    print('Episode: {}, Ep_reward: {}, Steps_per_ep: {}'.format(episode, episode_reward, steps))


