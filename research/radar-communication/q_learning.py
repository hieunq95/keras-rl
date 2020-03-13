#  Q-learning implementation

from __future__ import division
from environment import AV_Environment
from config import test_parameters, state_space_size, action_space_size, transition_probability, unexpected_ev_prob
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import json

def generate_retrieval_table(state_size):
    state_retrieval = {}  # table for retrieval actual state values to decimal state value
    for i in range(state_size):
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
        if r_offset in np.arange(0, 8):
            r = 0
        if r_offset in np.arange(8, 16):
            r = 1
        if c_offset in np.arange(0, 16):
            c = 0
        if c_offset in np.arange(16, 32):
            c = 1

        actual_state = [d, c, r, w, v, m]
        state_retrieval['{}'.format(i)] = actual_state

    return state_retrieval

def get_key_from_value(dict, value):
    """
    :param dict: retrieval table
    :param value: current state in array
    :return:
    """
    key = ''
    for k, v in dict.items():
        if (v == value).all():
            key = k
    # if key < 0:
    #     raise Exception('Invalid key value')
    return key


learning_parameters = {
    'alpha': test_parameters['alpha'],
    'gamma': test_parameters['gamma'],
    'eps_max': 1,
    'eps_min': 0.1,
    'nb_episodes': int(test_parameters['nb_steps'] / 400),
    'linear_episodes': int(test_parameters['nb_epsilon_linear'] / 400),
    'eps_decay': (1 - 0.1) / (test_parameters['nb_epsilon_linear'] / 400),
}
# Environment set up
env = AV_Environment()
# For statistic
json_data = {}
json_data['episode'] = []
json_data['episode_reward'] = []
json_data['nb_unexpected_ev'] = []
json_data['nb_episode_steps'] = []
json_data['mean_action'] = []
json_data['wrong_mode_actions'] = []
json_data['throughput'] = []
json_data['mean_q'] = []
json_data['mean_eps'] = []
json_data['avg_reward'] = []
json_data['avg_sim_reward'] = []
json_data['avg_reward_error'] = []

histogram = []
x_array = []
y_array = []
y_value = 0

q_state_size = state_space_size['data_size'] * state_space_size['channel_size'] * state_space_size['road_size'] \
               * state_space_size['weather_size'] * state_space_size['speed_size'] * state_space_size['object_size']
print('State space: {}'.format(q_state_size))

# Adding key-value pairs to the state_retrieval dictionary, key: decimal state, value: actual state
retrieval_table = generate_retrieval_table(q_state_size)
# print(retrieval_table)

#  Initialize Q-table
q_table = {}
for i in range(q_state_size):
    q_table['{}'.format(i)] = [0, 0]
# print(q_table)

# Training begins
TEST_ID = test_parameters['test_id']
print('********************* Start Q-Learning test-id: {} ***********************'.format(TEST_ID))
print('************************************************************************** \n '
      '**************************** Simulation parameters*********************** \n'
      '{} \n {} \n {} \n {} \n {} \n'.format(transition_probability, unexpected_ev_prob, state_space_size,
                                          action_space_size, learning_parameters)
      + '*************************************************************************** \n')

# epsilon = learning_parameters['eps_max']
epsilon = learning_parameters['eps_min']

for e in range(1, learning_parameters['nb_episodes'] + 1):
    actions = []
    wrong_mode_actions = 0
    max_q_array = []

    actual_state = env.reset()
    q_state = get_key_from_value(retrieval_table, actual_state)
    steps, reward, episode_reward = 0, 0, 0
    done = False
    # if epsilon >= learning_parameters['eps_min'] + learning_parameters['eps_decay']:
    #     epsilon -= learning_parameters['eps_decay']
    # else:
    #     epsilon = learning_parameters['eps_min']

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table['{}'.format(q_state)])
            if action > action_space_size['action_size']:
                raise Exception('Invalid action')

        next_state, reward, done, info = env.step(action)
        next_q_state = get_key_from_value(retrieval_table, next_state)
        old_value = q_table['{}'.format(q_state)][action]
        next_max = np.max(q_table['{}'.format(next_q_state)])

        new_value = (1 - learning_parameters['alpha']) * old_value \
                     + learning_parameters['alpha'] * (reward + learning_parameters['gamma'] * next_max)
        q_table['{}'.format(q_state)][action] = new_value

        q_state = next_q_state
        episode_reward += reward
        steps += 1
        actions.append(action)
        if reward < 0:
            wrong_mode_actions += 1

    # Save data to json file
    json_data['episode'].append(e)
    json_data['episode_reward'].append(int(episode_reward))
    json_data['nb_unexpected_ev'].append(env.episode_observation['unexpected_ev_counter'])
    json_data['nb_episode_steps'].append(steps)
    json_data['mean_action'].append(np.mean(actions))
    json_data['wrong_mode_actions'].append(wrong_mode_actions)
    json_data['throughput'].append(env.episode_observation['throughput'] / steps)
    json_data['mean_eps'].append(epsilon)
    json_data['avg_reward'].append(episode_reward / steps)

    for k in range(q_state_size):
        max_q_array.append(np.amax(q_table['{}'.format(k)]))
    json_data['mean_q'].append(np.mean(max_q_array))

    print('Episode: {}, Epsilon: {}, Total reward: {}, Steps: {}, Average reward: {}, Mean action: {}, Mean_q: {}'
          .format(e, epsilon, episode_reward, steps, episode_reward / steps, np.mean(actions),
                  json_data['mean_q'][- 1]))

    if test_parameters['add_simulation']:
        # Averaging total reward of different scenarios using current Q-table
        s0 = get_key_from_value(retrieval_table, env.state)
        r_array = []
        simulation_steps = q_state_size * 10
        for i in range(simulation_steps):
            a0 = np.argmax(q_table['{}'.format(s0)])
            s0, r, d, _ = env.step(a0)
            s0 = get_key_from_value(retrieval_table, s0)
            r_array.append(r)
        ave_r = np.mean(r_array)
        total_r = np.sum(r_array)
        # Saving into json_data
        json_data['avg_sim_reward'].append(ave_r)
        json_data['avg_reward_error'].append(ave_r - episode_reward / steps)

        print(' ********* Episode: {}, Simulation steps: {}, Total reward: {}, Average total reward: {}, Average error: {} '
              '*********'.format(e, simulation_steps, total_r, ave_r, json_data['avg_reward_error'][-1]))

    with open('./logs/q_learning_AV_Radar_log_{}.json'.format(TEST_ID), 'w') as outfile:
        json.dump(json_data, outfile)

# End of training
print('********************* End Q-Learning test-id: {} ***********************'.format(TEST_ID))
