import matplotlib.pyplot as plt
import numpy as np
from environment import Environment
from writer_v1 import MCMLWriter
import xlsxwriter
import gym


mempools = []
workbook = xlsxwriter.Workbook()
writer = MCMLWriter(workbook)
env = Environment(mempools, writer)
# Test env cartpole
# env = gym.make('CartPole-v0')
env.seed(123)
state = env.reset()
print(env.observation_space.shape, env.action_space.nvec)
# for ep in range(1000):
actions = []
rewards = []
states = []

# Test convert action

for t in range(3000):
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    state = next_state
    mempool = state[-1]
    mempools.append(mempool)
    rewards.append(reward)

    states.append(next_state[4])
    # states.append(env.observation_space.sample()[3])
    actions.append(action[3])

# plt.plot(np.arange(0, len(rewards)), actions)

# Plot histogram
target1 = states
target2 = actions
target3 = rewards

target = target3

# hist = plt.hist([target1, target2])
hist = plt.hist(target)
print(hist[0], hist[1])
total_apperance = np.sum(hist[0])
hist_percentive = np.copy(hist[0])

for i in range(len(hist_percentive)):
    hist_percentive[i] /= total_apperance
print(hist_percentive, np.sum(hist_percentive))

plt.show()

plt.plot(np.arange(0, len(target)), target)
plt.show()
