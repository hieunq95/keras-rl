import matplotlib.pyplot as plt
import numpy as np
from environment import Environment
from writer_v1 import MCMLWriter
import xlsxwriter


mempools = []
workbook = xlsxwriter.Workbook()
writer = MCMLWriter(workbook)
env = Environment(mempools, writer)
env.seed(1000)
state = env.reset()
print(env.observation_space, env.action_space.nvec)
# for ep in range(1000):
actions = []
rewards = []

# Test convert action

for t in range(5000):
    init_state = state
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    state = next_state
    mempool = state[-1]
    mempools.append(mempool)
    rewards.append(reward)
    actions.append(action[-3])

# plt.plot(np.arange(0, len(rewards)), actions)

# Plot histogram
hist = plt.hist(actions)
print(hist[0])
total_apperance = np.sum(hist[0])
hist_percentive = np.copy(hist[0])

for i in range(len(hist_percentive)):
    hist_percentive[i] /= total_apperance
print(hist_percentive, np.sum(hist_percentive))

plt.show()
