#  Testing environment for an Autonomous Vehicle
from __future__ import division
import json
import matplotlib.pyplot as plt
import pandas
import numpy as np
from matplotlib import rc

def learning_curve(files, evaluated_value, ewmw):
    """
    Input: files - an string array of files' path
    """
    for f in files:
        x_array = []
        y_array = []
        with open(f) as json_file:
            data = json.load(json_file)
            ewm_data = pandas.read_json(f)[evaluated_value].ewm(span=ewmw, adjust=False).mean()
            i = 0
            for p in data[evaluated_value]:
                x_array.append(i)
                y_array.append(ewm_data[i])
                i += 1
        if f.find('dqn') > 0:
            plt.plot(x_array, y_array, label='DQN')
        elif f.find('q_learning') > 0:
            plt.plot(x_array, y_array, label='Q-learning')
        else:
            plt.plot(x_array, y_array, label='Alternative switching')

    plt.xlabel('Episode', fontsize=15)
    plt.ylabel('Total reward', fontsize=15)
    plt.legend()
    plt.savefig('./results/{}.pdf'.format(EVALUATED_VALUE))
    plt.show()

def line_graph(x_axis, nb_lines, files, evaluated_value, averaged_point, axis_titles, output):
    """
    @input: y_axis, files, averaged_point
    """
    for l in range(nb_lines):
        line_values = []
        for k in range(len(x_axis)):
            value_array = []
            with open(files[l][k]) as json_file:
                data = json.load(json_file)
                # for p, q in zip(data[evaluated_value], data['nb_unexpected_ev']):
                #     value_array.append(p / q)
                # for p in data[evaluated_value]:
                    # value_array.append(p)
                line_values.append(np.mean(value_array[averaged_point:]))
        print(line_values)
        if files[l][0].find('dqn') > 0:
            plt.plot(x_asis, line_values, '-*', label='DQN')
        elif files[l][1].find('q_learning') > 0:
            plt.plot(x_asis, line_values, '-^', label='Q-learning')
        else:
            plt.plot(x_asis, line_values, '-o', label='Alternative switching')
    # plt.rc('text', usetex=True)
    plt.xlabel(axis_titles[0], fontsize=15)
    plt.ylabel(axis_titles[1], fontsize=15)
    plt.xticks(x_asis)
    plt.grid(b=True, which='both', axis='both', linestyle='-.', linewidth=0.2)
    plt.legend()
    plt.savefig('./results/{}.pdf'.format(output_file))
    plt.show()
"""
loss mae mean_q mean_eps nb_episode_steps nb_steps episode duration nb_unexpected_ev
#  episode_reward mean_action throughput wrong_mode_actions 
"""
EWM_WINDOW = 20
EVALUATED_VALUE = 'episode_reward'
DATA = './logs/dqn_AV_Radar-v1_log_50.json'
# DATA_REF = './logs/switch_AV_Radar_log_50.json'
# DATA = './logs/switch_AV_Radar_log_50.json'
DATA_REF = './logs/q_learning_AV_Radar_log_50.json'
DATA_REF2 = './logs/switch_AV_Radar_log_50.json'
files = [DATA, DATA_REF, DATA_REF2]
learning_curve(files, EVALUATED_VALUE, EWM_WINDOW)

x_asis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

files_array_1 = ['./logs/dqn_AV_Radar-v1_log_{}.json'.format(k) for k in range(36, 46)]
files_array_2 = ['./logs/q_learning_AV_Radar_log_{}.json'.format(k) for k in range(36, 46)]
files_array_3 = ['./logs/switch_AV_Radar_log_{}.json'.format(k) for k in range(36, 46)]


files_line_graph = [files_array_1, files_array_2, files_array_3]
axis_titles = [r'$\rho_1^c$', 'Miss detection probability']
output_file = EVALUATED_VALUE + '_vs_channel'
# line_graph(x_asis, 3, files_line_graph, EVALUATED_VALUE, 1500, axis_titles, output_file)

