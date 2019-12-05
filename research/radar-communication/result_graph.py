#  Testing environment for an Autonomous Vehicle

import json
import matplotlib.pyplot as plt
import pandas
import numpy as np

#  loss mae mean_q mean_eps episode_reward nb_episode_steps nb_steps episode duration nb_unexpected_ev

EWM_WINDOW = 10
EVALUATED_VALUE = 'episode_reward'
DATA = './logs/dqn_AV_Radar-v1_log_15.json'
x_array = []
y_array = []
y_ewm_array = []
data_ewm = pandas.read_json(DATA)
ewm_value = data_ewm[EVALUATED_VALUE].ewm(span=EWM_WINDOW, adjust=False).mean()

with open(DATA) as json_file:
    data = json.load(json_file)
    i = 0
    for p in data[EVALUATED_VALUE]:
        print('{} + \t {}'.format(i, p))
        x_array.append(i)
        y_array.append(p)
        y_ewm_array.append(ewm_value[i])
        i += 1

print('Average {} value: {}'.format(EVALUATED_VALUE.upper(), np.mean(y_array[1500:])))
plt.plot(x_array, y_array, label=EVALUATED_VALUE.upper())
plt.plot(x_array, y_ewm_array, label=EVALUATED_VALUE.upper() + ' with EWM={}'.format(EWM_WINDOW))
plt.legend()
plt.show()
