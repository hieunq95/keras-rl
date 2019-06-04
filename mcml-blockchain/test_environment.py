import matplotlib.pyplot as plt
import numpy as np
from environment import Environment

env = Environment()
env.seed(1000)


print(env.observation_space, env.action_space.nvec)

for i in range(10):
    state_init = env.reset()
    # print(state_init)

rewards = []
state = env.reset()
for t in range(2000):
    #test random action
    print(state)
    next_state, reward , done, info = env.step(state)
    rewards.append(reward)
    state = next_state
    print(state, reward, done, info)

plt.plot(np.arange(0,2000), rewards)
plt.show()