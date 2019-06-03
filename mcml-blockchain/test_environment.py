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
for t in range(2000):
    #test random action

    state, reward , done, info = env.step(env.action_space.sample())
    rewards.append(reward)
    print(state, reward, done, info)

plt.plot(np.arange(0,2000), rewards)
plt.show()