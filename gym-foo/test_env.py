import gym
import gym_foo
import random as random

ENV_NAME = 'OneRoundNondeterministicReward-v0'
# ENV_NAME = 'foo-v0'
env = gym.make(ENV_NAME)
print(env.observation_space.n, env.action_space.n)
env.reset()
action = [0, 1]

EPISODE = 5
for t in range(EPISODE):

    # get reward from taking an random action
    observation, reward, done, info = env.step(random.randrange(env.action_space.n))
    print(observation, reward, done, info)