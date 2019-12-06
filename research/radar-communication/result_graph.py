#  Testing environment for an Autonomous Vehicle
from __future__ import division
import json
import matplotlib.pyplot as plt
import pandas
import numpy as np

#  loss mae mean_q mean_eps episode_reward nb_episode_steps nb_steps episode duration nb_unexpected_ev

EWM_WINDOW = 100
EVALUATED_VALUE = 'nb_unexpected_ev'
DATA = './logs/dqn_AV_Radar-v1_log_19.json'
DATA_REF = './logs/switch_AV_Radar_log.json'
x_array = []
y_array = []
y_ref_array = []
y_ewm_array = []
y_ref_ewm_array = []
data_ewm = pandas.read_json(DATA)
data_ref_ewm = pandas.read_json(DATA_REF)
ewm_value = data_ewm[EVALUATED_VALUE].ewm(span=EWM_WINDOW, adjust=False).mean()
ewm_ref_value = data_ref_ewm[EVALUATED_VALUE].ewm(span=EWM_WINDOW, adjust=False).mean()

with open(DATA) as json_file:
    data = json.load(json_file)
    i = 0
    for p in data[EVALUATED_VALUE]:
        x_array.append(i)
        y_array.append(p)
        y_ewm_array.append(ewm_value[i])
        i += 1

with open(DATA_REF) as json_ref_file:
    data = json.load(json_ref_file)
    i = 0
    for p in data[EVALUATED_VALUE]:
        y_ref_ewm_array.append(ewm_ref_value[i])
        i += 1

print('Average {} value: DQN: {}, SWITCH: {}'.format(EVALUATED_VALUE.upper(), np.mean(y_ewm_array[2000:]), np.mean(y_ref_ewm_array[2000:])))
# plt.plot(x_array, y_array, label=EVALUATED_VALUE.upper())
plt.plot(x_array, y_ewm_array, label=EVALUATED_VALUE.upper() + ' DQN with EWM={}'.format(EWM_WINDOW))
plt.plot(x_array, y_ref_ewm_array, label=EVALUATED_VALUE.upper() + ' ALT_SWITCH with EWM={}'.format(EWM_WINDOW))
plt.legend()
plt.show()
