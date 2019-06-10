import matplotlib.pyplot as plt
import numpy as np
from environment import Environment
from writer_v1 import MCMLWriter
import xlsxwriter

rewards = []
mempools = []
workbook = xlsxwriter.Workbook()
writer = MCMLWriter(workbook)
env = Environment(mempools, writer)
env.seed(1000)
state = env.reset()
print(env.observation_space, env.action_space.nvec)
# for ep in range(1000):
init_action = env.action_space.sample()

for t in range(5000):
    init_state = state
    next_state, reward, done, info = env.step(env.action_space.sample())
    state = next_state
    mempool = state[-1]
    mempools.append(mempool)
    rewards.append(reward)


# plt.plot(np.arange(0, len(rewards)), rewards)
hist = plt.hist(rewards)
print(hist[0])
total_apperance = np.sum(hist[0])
hist_percentive = np.copy(hist[0])

for i in range(len(hist_percentive)):
    hist_percentive[i] /= total_apperance

print(1e-3)
print(hist_percentive, np.sum(hist_percentive))
plt.show()
