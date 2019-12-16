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
                for p, q in zip(data[evaluated_value], data['nb_unexpected_ev']):
                    value_array.append(p / q)
                # for p in data[evaluated_value]:
                #     value_array.append(p)
                line_values.append(np.mean(value_array[averaged_point:]))
        print(line_values)
        if files[l][0].find('dqn') > 0:
            plt.plot(x_asis, line_values, '-*', label='DQN')
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
EWM_WINDOW = 10
EVALUATED_VALUE = 'wrong_mode_actions'
DATA = './logs/dqn_AV_Radar-v1_log_50.json'
DATA_REF = './logs/switch_AV_Radar_log_50.json'

files = [DATA, DATA_REF]
# learning_curve(files, EVALUATED_VALUE, EWM_WINDOW)

x_asis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
f_1 = './logs/dqn_AV_Radar-v1_log_36.json'
f_2 = './logs/dqn_AV_Radar-v1_log_37.json'
f_3 = './logs/dqn_AV_Radar-v1_log_38.json'
f_4 = './logs/dqn_AV_Radar-v1_log_39.json'
f_5 = './logs/dqn_AV_Radar-v1_log_40.json'
f_6 = './logs/dqn_AV_Radar-v1_log_41.json'
f_7 = './logs/dqn_AV_Radar-v1_log_42.json'
f_8 = './logs/dqn_AV_Radar-v1_log_43.json'
f_9 = './logs/dqn_AV_Radar-v1_log_44.json'
f_10 = './logs/dqn_AV_Radar-v1_log_45.json'

f_11 = './logs/switch_AV_Radar_log_36.json'
f_12 = './logs/switch_AV_Radar_log_37.json'
f_13 = './logs/switch_AV_Radar_log_38.json'
f_14 = './logs/switch_AV_Radar_log_39.json'
f_15 = './logs/switch_AV_Radar_log_40.json'
f_16 = './logs/switch_AV_Radar_log_41.json'
f_17 = './logs/switch_AV_Radar_log_42.json'
f_18 = './logs/switch_AV_Radar_log_43.json'
f_19 = './logs/switch_AV_Radar_log_44.json'
f_20 = './logs/switch_AV_Radar_log_45.json'

files_line_graph = [[f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9, f_10],
                    [f_11, f_12, f_13, f_14, f_15, f_16, f_17, f_18, f_19, f_20]]
axis_titles = [r'$\rho_1^c$', 'Miss detection probability']
output_file = EVALUATED_VALUE + '_vs_channel'
line_graph(x_asis, 2, files_line_graph, EVALUATED_VALUE, 1500, axis_titles, output_file)

