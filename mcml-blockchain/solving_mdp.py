# Solving MDP problem
# @author: Hieu Nguyen
# https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287

import numpy as np
import gym
import matplotlib.pyplot as plt

# actions = []
# states = []
#
env = gym.make('FrozenLake-v0')
# env = gym.make('FrozenLake8x8-v0')
env.seed(1000)
print(env.action_space, env.observation_space)
env.render()
#
# for i in range(1000):
#     action = env.action_space.sample()
#     state = env.observation_space.sample()
#     actions.append(action)
#     states.append(state)
#     # print(action)
#
# target = states
# histogram = plt.hist(target)
# print environmental info
# print(histogram[0], histogram[1], len(histogram[1]))
# plt.show()

HOLE = b'H'
GOAL = b'G'
nrow = 4
ncol = 4

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Value iteration: update value_matrix according to Bellman equation
value_matrix = np.zeros((nrow, ncol))
print(env.desc)
print(value_matrix)

def next_state(row, col, a, p):
    # random action
    if a == LEFT:
        a = np.random.choice([LEFT, DOWN, UP], p=[p, (1-p)/2, (1-p)/2])
    if a == DOWN:
        a = np.random.choice([DOWN, LEFT, RIGHT], p=[p, (1 - p) / 2, (1 - p) / 2])
    if a == RIGHT:
        a = np.random.choice([RIGHT, DOWN, UP], p=[p, (1 - p) / 2, (1 - p) / 2])
    if a == UP:
        a = np.random.choice([UP, RIGHT, LEFT], p=[p, (1 - p) / 2, (1 - p) / 2])

    if a == LEFT:
        col = max(col-1, 0)
    elif a == DOWN:
        row = min(row+1, nrow-1)
    elif a == RIGHT:
        col = min(col+1, ncol-1)
    elif a == UP:
        row = max(row-1, 0)
    return (row, col)

def update_value(state, gamma=0.9, p=0.8):
    """
    Bellman update

    :param state: position of the player

    :param gamma: discount factor

    :param p: probability of taking correct action

    :return:
    """
    x_pos = state[0]
    y_pos = state[1]
    bellman_values = []
    possible_next_states = []

    if env.desc[x_pos][y_pos] == GOAL:
        reward = 1
    elif env.desc[x_pos][y_pos] == HOLE:
        reward = -1
    else:
        reward = 0

    for a in range(8):
        possible_next_states.append(next_state(x_pos, y_pos, a, p))
    # assume that env is not slippery
    for s in possible_next_states:
        s_row = s[0]
        s_col = s[1]
        # if env.desc[s_row][s_col] == GOAL:
        #     reward = 1
        # elif env.desc[x_pos][y_pos] == HOLE:
        #     reward = -1
        # else:
        #     reward = 0
        bellman_values.append(reward + gamma * value_matrix[s_row][s_col])

    new_value = max(bellman_values)
    value_matrix[x_pos][y_pos] = new_value
    # print(new_value)
    # value_matrix[x_pos][y_pos] = reward + gamma *
    # print(state, gamma, p)

for i in range(ncol):
    for j in range(nrow):
        if env.desc[i][j] == HOLE:
            value_matrix[i][j] = -1
        elif env.desc[i][j] == GOAL:
            value_matrix[i][j] = 1
        else:
            value_matrix[i][j] = 0

print(value_matrix)

optimal_values = []
for k in range(100):
    for row in range(nrow):
        for col in range(ncol):
            update_value([row, col], p=0.8)
    optimal_values.append(value_matrix[-1][-1])
    # print(value_matrix)
print(value_matrix)
plt.plot(np.arange(0, len(optimal_values)), optimal_values)
plt.show()
