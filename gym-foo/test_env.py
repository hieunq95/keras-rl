import gym
import gym_foo
import random as random
import numpy as np
import matplotlib.pyplot as plt

from mcml_processor import MCMLProcessor

def base10toN(num, base):
    """Change ``num'' to given base
    Upto base 36 is supported."""

    converted_string, modstring = "", ""
    currentnum = num
    if not 1 < base < 37:
        raise ValueError("base must be between 2 and 36")
    if not num:
        return '0'
    while currentnum:
        mod = currentnum % base
        currentnum = currentnum // base
        converted_string = chr(48 + mod + 7*(mod > 10)) + converted_string
    return converted_string

# ENV_NAME = 'OneRoundNondeterministicReward-v0'
ENV_NAME = 'mcml-v0'
env = gym.make(ENV_NAME)
# print(env.observation_space, env.action_space, env.action_space.nvec)
env.reset()

# nb_actions = 1
# for i in env.action_space.nvec:
#     nb_actions = nb_actions * i
# Test step()
# print(np.min([33, 8]))
# print(np.random.randint(0, 5))
print(env.action_space.shape, env.observation_space.shape, env.action_space.sample())


# xPoints, yPoints = [], []
# for x in range(10):
#     state, reward, done, info = env.step(env.action_space.sample())
#     xPoints.append(x)
#     yPoints.append(reward)
#     env.reset()
#     print(state, reward, done, info)
# plt.plot(xPoints, yPoints)
# plt.show()

# metrics_names = ['loss', 'mean_absolute_error', 'mean_q']
# processor_metrics_names = ['']
# metrics_names += MCMLProcessor.metrics_names()
# print(metrics_names)

print(base10toN(4092, 4))




