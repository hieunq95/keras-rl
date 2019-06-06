import matplotlib.pyplot as plt
import numpy as np
from environment import Environment

env = Environment()
env.seed(1000)


print(env.observation_space, env.action_space.nvec)

state = env.reset()
rewards = []
mempools = []
# for ep in range(1000):
init_action = env.action_space.sample()
for t in range(500):
    init_state = state
    next_state, reward, done, info = env.step(init_action)
    state = next_state
    mempool = state[-1]
    mempools.append(mempool)

plt.plot(np.arange(0, len(mempools)), mempools)
plt.show()



# for t in range(2000):
#     #test random action
#     print('state {}'.format(state))
#     action = env.action_space.sample()
#     next_state, reward , done, info = env.step(action)
#     rewards.append(reward)
#     mempool = next_state[-1]
#     mempools.append(mempool)
#     state = next_state
#     print(state, reward, done, info)
#
# plt.plot(np.arange(0, len(mempools)), mempools)
# plt.show()