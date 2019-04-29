import gym
import gym_foo
import random as random
import numpy as np
import matplotlib.pyplot as plt

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
print(env.action_space.shape, env.observation_space.shape)

xPoints, yPoints = [], []
for x in range(100):
    state, reward, done, info = env.step(env.action_space.sample())
    xPoints.append(x)
    yPoints.append(reward)
    env.reset()
    print(state, reward, done, info)
plt.plot(xPoints, yPoints)
plt.show()

# print(nb_actions)
# nprandom = np.random.RandomState()
# reset_state = nprandom.randint(low=0, high=10, size=env.observation_space.shape)
# print(reset_state)




