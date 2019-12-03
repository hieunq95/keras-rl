from environment import AV_Environment
import matplotlib.pyplot as plt
import numpy as np


env = AV_Environment()

print(env.observation_space, env.action_space)
histogram = []
for e in range(1000):
    plot_target = 0
    cumulative_reward = 0
    for t in range(1000):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        cumulative_reward += reward
        plot_target = next_state[5]
        histogram.append(plot_target)

        # print(t, next_state, reward, done, info)
        if done == True:
            # print('Break at t = {}'.format(t))
            # plot_target = next_state[1]
            # histogram.append(plot_target)
            env.reset()
            break

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