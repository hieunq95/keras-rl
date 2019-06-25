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

q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Value iteration: update value_matrix according to Bellman equation
value_matrix = np.zeros((nrow, ncol))
print(env.desc)

def q_reward(s,a):
    row = s // ncol
    col = s - row * ncol
    new_state = next_state(row,col,a,1)
    new_row = new_state[0]
    new_col = new_state[1]

    if env.desc[new_row][new_col] == HOLE:
        reward = -1
    elif env.desc[new_row][new_col] == GOAL:
        reward = 1
    else:
        reward = 0
    return reward

def qlearning_update(s, lr=0.1, gamma=0.9):
    row = s // ncol
    col = s - row * ncol
    optimal_action = optimal_policy(q_table, row, col)
    new_state = next_state(row, col, optimal_action, 1)
    new_state = new_state[0] * ncol + new_state[1]
    for a in range(env.action_space.n):
        q_table[s][a] = (1-lr)*q_table[s][a] + lr*(q_reward(s,a) + gamma*np.max(q_table[new_state]))

    return q_table

def get_reward(next_state):
    row = next_state[0]
    col = next_state[1]
    return reward_matrix[row][col]

def optimal_policy(state_matrix,row, col):
    nbr_values = []
    right_nbr = state_matrix[row][col+1] if col + 1 < ncol else 0
    left_nbr = state_matrix[row][col-1] if col - 1 > 0 else 0
    up_nbr = state_matrix[row-1][col] if row - 1 > 0 else 0
    down_nbr = state_matrix[row+1][col] if row + 1 < nrow else 0
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

def init_value_matrix():
    for i in range(ncol):
        for j in range(nrow):
            if env.desc[i][j] == HOLE:
                value_matrix[i][j] = -1
            elif env.desc[i][j] == GOAL:
                value_matrix[i][j] = 1
            else:
                value_matrix[i][j] = 0

def value_iteration(iterator=500):
    # Value iteration algorithm
    for k in range(iterator):
        for row in range(nrow):
            for col in range(ncol):
                update_value([row, col], p=1)
        optimal_values.append(value_matrix[-1][-1])

def test(episode=20):
    for ep in range(episode):
        curr_position = (0, 0)
        step_counter = 0
        policy = []
        for step in range(30):
            step_counter += 1
            # row = curr_position // ncol
            # col = curr_position - row * ncol
            row = curr_position[0]
            col = curr_position[1]
            action = optimal_policy(value_matrix,row, col)
            if action == 0:
                policy.append('LEFT')
            elif action == 1:
                policy.append('DOWN')
            elif action == 2:
                policy.append('RIGHT')
            elif action == 3:
                policy.append('UP')

            next_position = next_state(row, col, action, p=1)
            curr_position = next_position
            if curr_position == (nrow - 1, ncol - 1):
                break
            # if reward_matrix[curr_position[0]][curr_position[1]] == -1:
            #     print('Hole')
            #     break
        print(policy)


optimal_values = []
init_value_matrix()
reward_matrix = np.copy(value_matrix)
value_iteration(500)
print(reward_matrix)
print(value_matrix)
test(2)



