#  Testing environment for an Autonomous Vehicle
from __future__ import division
import json
import matplotlib.pyplot as plt
import pandas
import numpy as np

#  loss mae mean_q mean_eps episode_reward nb_episode_steps nb_steps episode duration nb_unexpected_ev
#  mean_action wrong_mode_actions

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
            plt.plot(x_array, y_array, label=evaluated_value + ' - DQN with EWM={}'.format(ewmw))
        else:
            plt.plot(x_array, y_array, label=evaluated_value + ' - Alt_switch with EWM={}'.format(ewmw))

    plt.xlabel('Episode')
    plt.ylabel(EVALUATED_VALUE.upper())
    plt.legend()
    # plt.savefig('./results/{}.png'.format(EVALUATED_VALUE))
    plt.show()

def line_graph(x_axis, files, evaluated_value, averaged_point):
    """
    @input: y_axis, files, averaged_point
    """
    for f in files:
        y_array = []
        with open(f) as json_file:
            data = json.load(json_file)
            i = 0
            for k in x_axis:
                y_array.append(np.mean(data))


EWM_WINDOW = 100
EVALUATED_VALUE = 'wrong_mode_actions'
DATA = './logs/dqn_AV_Radar-v1_log_26.json'
DATA_REF = './logs/switch_AV_Radar_log.json'

files = [DATA, DATA_REF]
learning_curve(files, EVALUATED_VALUE, EWM_WINDOW)

