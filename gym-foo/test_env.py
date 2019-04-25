import gym
import gym_foo
import random as random
import numpy as np

# ENV_NAME = 'OneRoundNondeterministicReward-v0'
# ENV_NAME = 'foo-v0'
ENV_NAME = 'mcml-v0'
env = gym.make(ENV_NAME)
print(env.observation_space, env.action_space)
env.reset()

# action = [0, 1]
#
# EPISODE = 5
# for t in range(EPISODE):
#
#     # get reward from taking an random action
#     observation, reward, done, info = env.step(random.randrange(env.action_space.n))
#     print(observation, reward, done, info)
for _ in range(10):
    a = np.random.randint(0, 10 + 1)
    print(a)

