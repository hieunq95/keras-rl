# Win-stay, lose-shift implementation

from __future__ import division
from environment import AV_Environment
from config import test_parameters
import matplotlib.pyplot as plt
import numpy as np
import json
import array as arr

import h5py
filename = './logs/dqn_AV_Radar-v1_weights_67.h5f'

with h5py.File(filename, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    key_list = list(f.keys())
    for l in key_list:
        print(l)
    # Get the data
    # data = list(f[key_list[0]])

    # print(data)

env = AV_Environment()
json_data = {}
json_data['episode'] = []
json_data['episode_reward'] = []
json_data['nb_unexpected_ev'] = []
json_data['nb_episode_steps'] = []
json_data['mean_action'] = []
json_data['wrong_mode_actions'] = []
json_data['throughput'] = []

x_array, y_array = [], []


def win_stay(a, r):
    if r > 0:
        return a
    else:
        return (a + 1) % 2

x = [1, 2, 3 ,4, 7, 5]
print(x)
x1 = arr.array([1,1,1,1,1,1])
print(x1)

# for e in range(1, 2501):
#     episode_reward = 0
#     actions = []
#     wrong_mode_actions = 0
#     steps = 0
#     action = env.action_space.sample()
#     for t in range(1000):
#         steps += 1
#         next_state, reward, done, info = env.step(action)
#         action = win_stay(action, reward)
#         episode_reward += reward
#         actions.append(action)
#         if reward < 0:
#             wrong_mode_actions += 1
#         if done:
#             # Save data to json file
#             json_data['episode'].append(e)
#             json_data['episode_reward'].append(int(episode_reward))
#             json_data['nb_unexpected_ev'].append(env.episode_observation['unexpected_ev_counter'])
#             json_data['nb_episode_steps'].append(t)
#             json_data['mean_action'].append(np.mean(actions))
#             json_data['wrong_mode_actions'].append(wrong_mode_actions)
#             json_data['throughput'].append(env.episode_observation['throughput'] / steps)
#             print('Episode: {}, Total reward: {}, Steps: {}, Average reward: {}'
#                   .format(e, episode_reward, steps, episode_reward / steps))
#             env.reset()
#             break
#
# with open('./logs/win_stay_log_{}.json'.format(test_parameters['test_id']), 'w') as outfile:
#     json.dump(json_data, outfile)