# Solving MDP problem
# @author: Hieu Nguyen
# https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287

import numpy as np
import gym
import matplotlib.pyplot as plt

# actions = []
# states = []
#
# env = gym.make('FrozenLake-v0')
env = gym.make('FrozenLake8x8-v0')
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
nrow = 8
ncol = 8

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

reward_matrix = []
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Value iteration: update value_matrix according to Bellman equation
value_matrix = np.zeros((nrow, ncol))
print(env.desc)
print(value_matrix)

def q_reward(s):
    row = s[0]
    col = s[1]
    if env.desc[row][col] == HOLE:
        reward = -1
    elif env.desc[row][col] == GOAL:
        reward = 1
    else:
        reward = 0
    return reward

def qlearning_update(s, a, lr=0.1, gamma=0.9):
    row = s // ncol
    col = s - row * ncol
    # state_offset = row * ncol + col
    q_values = []
    new_state = next_state(row, col, a, 0.8)
    state_offset = new_state[0] * ncol + new_state[1]
    for action in range(0, env.action_space.n):
        tmp_q_value = (1-lr)*q_table[row][col] + lr*(q_reward(new_state) + gamma*q_table[state_offset][action])
        q_values.append(tmp_q_value)
    new_q_value, taken_action = np.max(q_values), np.argmax(q_values)
    q_table[state_offset][taken_action] = new_q_value
    # return  q_table

def get_reward(next_state):
    row = next_state[0]
    col = next_state[1]
    return reward_matrix[row][col]

def optimal_policy(row, col):
    nbr_values = []
    right_nbr = value_matrix[row][col+1] if col + 1 < ncol else 0
    left_nbr = value_matrix[row][col-1] if col - 1 > 0 else 0
    up_nbr = value_matrix[row-1][col] if row - 1 > 0 else 0
    down_nbr = value_matrix[row+1][col] if row + 1 < nrow else 0
    nbr_values.append(left_nbr)
    nbr_values.append(down_nbr)
    nbr_values.append(right_nbr)
    nbr_values.append(up_nbr)

    action = np.argmax(nbr_values)
    return action

def next_state(row, col, a, p):
    # random action according to transition probability P
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

    for a in range(env.action_space.n):
        possible_next_states.append(next_state(x_pos, y_pos, a, p))
    reward = get_reward(state)
    for s in possible_next_states:
        s_row = s[0]
        s_col = s[1]
        # reward = get_reward(s)
        bellman_values.append(reward + gamma * value_matrix[s_row][s_col])

    new_value = max(bellman_values)
    value_matrix[x_pos][y_pos] = new_value


for i in range(ncol):
    for j in range(nrow):
        if env.desc[i][j] == HOLE:
            value_matrix[i][j] = -1
        elif env.desc[i][j] == GOAL:
            value_matrix[i][j] = 1
        else:
            value_matrix[i][j] = 0

# reward_matrix = value_matrix
reward_matrix = np.copy(value_matrix)
print(reward_matrix)
optimal_values = []
# Value iteration algorithm
for k in range(50000):
    for row in range(nrow):
        for col in range(ncol):
            update_value([row, col], p=0.8)
    optimal_values.append(value_matrix[-1][-1])
    # print(value_matrix)
print(value_matrix)
# plt.plot(np.arange(0, len(optimal_values)), optimal_values)
# plt.show()

# After value iteration
for ep in range(20):
    curr_position = (0,0)
    step_counter = 0
    policy = []
    for step in range(10):
        step_counter += 1
        # row = curr_position // ncol
        # col = curr_position - row * ncol
        row = curr_position[0]
        col = curr_position[1]
        action = optimal_policy(row, col)
        if action == 0:
            policy.append('LEFT')
        elif action == 1:
            policy.append('DOWN')
        elif action == 2:
            policy.append('RIGHT')
        elif action == 3:
            policy.append('UP')

        next_position = next_state(row, col, action,p=1)
        curr_position = next_position
        if curr_position == (nrow-1, ncol-1):
            break
        # if reward_matrix[curr_position[0]][curr_position[1]] == -1:
        #     print('Hole')
        #     break
    print(policy)
