#  Testing environment for an Autonomous Vehicle

import json
import matplotlib.pyplot as plt
#  loss mae mean_q mean_eps episode_reward nb_episode_steps nb_steps episode duration

DATA = '/home/hieu/PycharmProjects/keras-rl/research/radar-communication/logs/dqn_AV_Radar-v1_log_4.json'
x_array = []
y_array = []
with open(DATA) as json_file:
    data = json.load(json_file)
    i = 0
    for p in data['episode_reward']:
        print('{} + \t {}'.format(i, p))
        x_array.append(i)
        y_array.append(p)
        i += 1

plt.plot(x_array, y_array)
plt.show()
