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

#  Initialize Q-table
q_table['q-values'].append([q_table['state'], 0.0, 0.0])
print(q_table)
while len(q_table['q-values']) <= q_state_size:
    sampled_state = q_table['state'].sample()
    for s in q_table['q-values']:
        if (sampled_state == s[0]).all():
            print('duplicate {}'.format(s[0]), len(q_table['q-values']))
            break
        else:
            q_table['q-values'].append([sampled_state, 0.0, 0.0])

print(q_table['q-values'])
print(len(q_table['q-values']))