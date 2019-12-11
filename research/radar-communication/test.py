from __future__ import division
from environment import AV_Environment
from config import test_parameters
import matplotlib.pyplot as plt
import numpy as np
import json

env = AV_Environment()
json_data = {}
json_data['episode'] = []
json_data['episode_reward'] = []
json_data['nb_unexpected_ev'] = []
json_data['nb_episode_steps'] = []
json_data['mean_action'] = []
json_data['wrong_mode_actions'] = []
json_data['throughput'] = []

def alternative_switch_action(t):
    if t % 2 == 0:
        return 0
    else:
        return 1

print(env.observation_space, env.action_space)
histogram = []
x_array = []
y_array = []
y_value = 0
for e in range(1, 2501):
    plot_target = 0
    cumulative_reward = 0
    x_array.append(e)
    y_value = 0
    actions = []
    wrong_mode_actions = 0
    for t in range(1000):
        # action = env.action_space.sample()
        action = alternative_switch_action(t)
        next_state, reward, done, info = env.step(action)
        cumulative_reward += reward
        plot_target = next_state[3]
        histogram.append(plot_target)
        actions.append(action)
        if reward < 0:
            wrong_mode_actions += 1
        # y_value += plot_target

        # print(t, next_state, reward, done, info)
        if done:
            # print('Break at t = {}'.format(t))
            # plot_target = next_state[1]
            # histogram.append(plot_target)
            y_value = env.episode_observation['unexpected_ev_counter']
            # y_value = cumulative_reward
            y_array.append(y_value)
            # Save data to json file
            json_data['episode'].append(e)
            json_data['episode_reward'].append(int(cumulative_reward))
            json_data['nb_unexpected_ev'].append(env.episode_observation['unexpected_ev_counter'])
            json_data['nb_episode_steps'].append(t)
            json_data['mean_action'].append(np.mean(actions))
            json_data['wrong_mode_actions'].append(wrong_mode_actions)
            json_data['throughput'].append(env.episode_observation['throughput'] / 400)
            env.reset()
            break

with open('./logs/switch_AV_Radar_log_{}.json'.format(test_parameters['test_id']), 'w') as outfile:
    json.dump(json_data, outfile)

print(np.mean(y_array))
plt.plot(x_array, y_array)
plt.show()
hist = plt.hist(histogram)
nb_occurrence = hist[0]
value_occurrence = hist[1]
occurrence_percentile = nb_occurrence / np.sum(nb_occurrence)
nonzero_value_occurrence = []
nonzero_occurrence_percentile = []
for i in range(len(occurrence_percentile)):
    if occurrence_percentile[i]:
        nonzero_occurrence_percentile.append(occurrence_percentile[i])
        nonzero_value_occurrence.append(value_occurrence[i])

print(nonzero_occurrence_percentile, nonzero_value_occurrence)
plt.show()