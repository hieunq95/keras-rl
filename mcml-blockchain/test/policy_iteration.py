# @author: Hieu Nguyen
# Policy iteration implementation

import numpy as np
import gym
import math

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
DISCOUNT = 0.9

As = [LEFT, DOWN, RIGHT, UP]

def action_nb_to_str(a):
    if a == LEFT:
        return 'LEFT'
    elif a == DOWN:
        return 'DOWN'
    elif a == RIGHT:
        return 'RIGHT'
    elif a == UP:
        return 'UP'

def to_s(row, col):
    ncol = env.ncol
    return row * ncol + col

def get_state_value(s):
    return state_value_matrix[s]

def set_state_value(s, v):
    state_value_matrix[s] = v

def get_policy(s):
    return policy_matrix[s]

def state_value_update(s):
    value = 0
    nb_next_states = 3

    a = get_policy(s)
    observation = P[s][a]

    for i in range(nb_next_states):
        prob = observation[i][0]
        next_state = observation[i][1]
        reward = observation[i][2]
        value += prob * (reward + DISCOUNT * get_state_value(next_state))
        print(i, observation[i], value)
    return value

def policy_update(s):
    policy_values = []
    nb_next_states = 3

    for a in As:
        value = 0
        observation = P[s][a]
        for i in range(nb_next_states):
            prob = observation[i][0]
            next_state = observation[i][1]
            reward = observation[i][2]
            value += prob * (reward + DISCOUNT * get_state_value(next_state))
        policy_values.append(value)

    return np.argmax(policy_values)

env = gym.make('FrozenLake-v0')
env.reset()
env.render(mode='human')

# Transition
P = env.P
# initialize the states value
state_value_matrix = [k * 0 for k in range(env.ncol * env.nrow)]
# initialize policy for all states
policy_matrix = [np.random.choice(As) for k in range(env.ncol * env.nrow)]

# print(P[0][0][1][0])

if __name__ == "__main__":
    # initilization
    # policy evaluation
    delta = 1
    while delta > 0.1:
        delta = 0
        for row in range(env.nrow):
            for col in range(env.ncol):
                s = to_s(row, col)
                v = get_state_value(s)
                v_update = state_value_update(s)
                set_state_value(s, v_update)
                delta = max(delta, math.fabs(v - v_update))
        # print(state_value_matrix)
    # policy improvement

