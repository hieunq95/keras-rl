# Q-learning agent for MCML-Block chain
# @author: Hieu Nguyen

import numpy as np
import gym
import matplotlib.pyplot as plt
import xlsxwriter
import random
import copy
from environment import Environment, MyProcessor
from writer_v1 import MCMLWriter
from config import MEMPOOL_MAX, CPU_SHARES, CAPACITY_MAX, ENERGY_MAX, DATA_MAX, MINING_RATE, NB_DEVICES
from policy_epgreedy import MyEpsGreedy

def array_to_scalar(arr, base1, base2):
    """
    Convert state array to scalar to feed q-table

    :param a: input state array

    :return: state scalar
    """
    base1 = base1
    base2 = base2
    scalar_state = 0
    state_size = len(arr)
    for i in range(state_size):
        if i < state_size:
            scalar_state += arr[i] * (base1 ** (state_size - 1 - i))
        else:
            scalar_state += arr[i] * (base2 ** (state_size - 1 - i))
    return scalar_state

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon_max = 0.9
epsilon_min = 0.05
epsilon_trained = 2000
epsilon_decay = (epsilon_max - epsilon_min) / epsilon_trained

# Training parameters
NB_EPISODES = 4000
TEST_ID = 10
episode = 0
epsilon = epsilon_max

# Statistic
mempool = []
workbook = xlsxwriter.Workbook('./build/q_learning-result-{}.xlsx'.format(TEST_ID))
writer = MCMLWriter(workbook)

# Initialize q-table
state_space_size = (CPU_SHARES ** NB_DEVICES) * (CAPACITY_MAX ** NB_DEVICES) * MEMPOOL_MAX  # 4 x 4 x 4 x 4 x MEMPOOL_MAX
action_space_size = (ENERGY_MAX ** NB_DEVICES) * (DATA_MAX ** NB_DEVICES) * MINING_RATE  # 4 x 4 x 4 x 4 x 4
q_table = np.zeros((state_space_size, action_space_size))
print(q_table.shape)

# Environment set up
env = Environment(mempool, writer)
env.reset()
processor = MyProcessor()
print(env.observation_space, env.action_space)

# Training begins
print('********************* Start Q-Learning test-id: {} ***********************'.format(TEST_ID))
for i in range(NB_EPISODES):
    # state = env.reset()
    state = array_to_scalar(env.reset(), CPU_SHARES, MEMPOOL_MAX)
    episode_reward = 0
    steps, reward = 0, 0
    done = False
    if epsilon >= epsilon_min + epsilon_decay:
        epsilon -= epsilon_decay
    else:
        epsilon = epsilon_min
    # print('Ep: {}, q_table shape: {}, state: {}'.format(episode+1, q_table.shape, state))
    while not done:
        action_scalar = 0
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
            action_scalar = array_to_scalar(action, ENERGY_MAX, ENERGY_MAX)
        else:
            action_scalar = np.argmax(q_table[state])
            if action_scalar > action_space_size:
                print('invalid action: {}, q_table[state].shape: {}'.format(action_scalar, q_table[state].shape))
            action = processor.process_action(action_scalar)
            # print(action_scalar, action)
        next_state, reward, done, info = env.step(action)
        next_state_scalar = array_to_scalar(next_state, CPU_SHARES, MEMPOOL_MAX)
        old_value = q_table[state, action_scalar]
        next_max = np.max(q_table[next_state_scalar])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action_scalar] = new_value

        state = next_state_scalar
        episode_reward += reward
        steps += 1

    episode += 1

    print('Episode: {}, Epsilon: {}, Total reward: {}, Steps: {}, Average reward: {}'
          .format(episode, epsilon, episode_reward, steps, episode_reward / steps))
# End of training
print('********************* End Q-Learning test-id: {} ***********************'.format(TEST_ID))
workbook.close()
