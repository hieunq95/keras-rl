import gym
import gym_foo
import random as random
import numpy as np
import matplotlib.pyplot as plt
import math

from mcml_processor import MCMLProcessor
from mcml_env import MCML
from collections import Counter

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

# ENV_NAME = 'MountainCar-v0'
# ENV_NAME = 'mcml-v0'
# env = gym.make(ENV_NAME)
env = MCML()
# print(env.observation_space, env.action_space, env.action_space.nvec)
env.reset()

# nb_actions = 1
# for i in env.action_space.nvec:
#     nb_actions = nb_actions * i
# Test step()
# print(np.min([33, 8]))
# print(np.random.randint(0, 5))
print(env.action_space.shape, env.observation_space.shape, env.action_space.sample())

# print(math.sqrt(10**18) / (0.6 * (10**9)))
# print((3/1.66)**2)

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

# print(base10toN(4092, 4))

num_games = 100
num_steps = []
rewards = []
for num_game in range(num_games):
    num_step = 0
    done = False
    env.reset()
    while not done:
        num_step += 1
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        rewards.append(reward)

    num_steps.append(num_step)
print("Number of games played: {}".format(num_games))
print("Average number of steps: {:0.2f}".format(np.mean(num_steps)))
print(
    "Number of steps distribution \n\t10%: {:0.2f} \t25%: {:0.2f} \t50%: {:0.2f} \t75%: {:0.2f} \t99%: {:0.2f}".format(
        *np.percentile(num_steps, [10, 25, 50, 75, 99])))
print("Distribution of rewards: {}".format(Counter(rewards)))
xAxis = np.arange(len(rewards))
plt.plot(xAxis, rewards)
plt.show()



