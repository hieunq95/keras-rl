# From dqn_cartpole.py

import numpy as np
import gym
import gym_foo
import pylab

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory

ENV_NAME = 'mcml-v0'
# ENV_NAME = 'MountainCarContinuous-v0'
# ENV_NAME = 'MountainCar-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
# nb_actions = 1
nb_actions = env.action_space.shape[0] # e.g 4**6 # sulution for action which is not a discrete ?
# for i in env.action_space.nvec:
#     nb_actions *= i

print(nb_actions, env.observation_space.shape)
# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape)) # input #input_shape = (1,) + (4,)
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions)) # output
model.add(Activation('linear'))
# print(model.summary())
print(env.observation_space.sample())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
# policy = BoltzmannQPolicy()
policy = GreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy, enable_double_dqn=False)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
learning_history = dqn.fit(env, nb_steps=50000, visualize=False, verbose=2, nb_max_episode_steps=100)

reward_history = learning_history.history.get('episode_reward')
episode_history = np.arange(0, len(reward_history))
print(reward_history)
# print(reward_history, episode_history)
# plot score and save image
pylab.plot(episode_history, reward_history, 'b')
pylab.savefig("./mcml.png")

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
# dqn.test(env, nb_episodes=5, visualize=True)

