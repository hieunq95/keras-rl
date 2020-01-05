#  Q-learning implementation

from __future__ import division
from environment import AV_Environment
from config import test_parameters, state_space_size, action_space_size
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import json

learning_parameters = {
    'alpha': 0.1,
    'gamma': 0.9,
    'eps_max': 1,
    'eps_min': 0.1,
    'nb_epsilons': 2500,
    'linear_epsilons': 1500,
}
q_table = {
    'state': spaces.MultiDiscrete(
            [state_space_size['data_size'], state_space_size['channel_size'], state_space_size['road_size'],
             state_space_size['weather_size'], state_space_size['speed_size'], state_space_size['object_size']]),
    'action': spaces.Discrete(2),
    'q-values': [],
}
q_state_size = state_space_size['data_size'] * state_space_size['channel_size'] * state_space_size['road_size'] \
               * state_space_size['weather_size'] * state_space_size['speed_size'] * state_space_size['object_size']
print(q_state_size)
state_retrieval = {}  # table for retrieval actual state values to decimal state value
# adding key-value pairs to the state_retrieval dictionary, key: decimal state, value: actual state
for i in range(q_state_size):
    # arrange state vector s = {d, c, r, w, v, m}
    c, r, w, v, m = 0, 0, 0, 0, 0
    c_offset = int(i / 11) % 32
    r_offset = int(i / 11) % 16
    w_offset = int(i / 11) % 8
    v_offset = int(i / 11) % 4
    m_offset = int(i / 11) % 2

    d = i % 11
    if m_offset == 0:
        m = 0
    else:
        m = 1
    if v_offset in np.arange(0, 2):
        v = 0
    if v_offset in np.arange(2, 4):
        v = 1
    if w_offset in np.arange(0, 4):
        w = 0
    if w_offset in np.arange(4, 8):
        w = 1
    if r_offset in np.arange(0, 16):
        r = 0
    if r_offset in np.arange(16, 32):
        r = 1
    if c_offset in np.arange(0, 16):
        c = 0
    if c_offset in np.arange(16, 32):
        c = 1

    actual_state = [d, c, r, w, v, m]
    state_retrieval['{}'.format(i)] = actual_state

print(state_retrieval)
#  Initialize Q-table
print(13 / 11, int(13 / 11), 11 % 11)

print(np.arange(2, 4))
if 5 in np.arange(2, 5):
    print('FOUND')
