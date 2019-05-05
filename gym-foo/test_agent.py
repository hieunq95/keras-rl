# From dqn_cartpole.py

import numpy as np
import gym
import pylab

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, GreedyQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from mcml_processor import MCMLProcessor
from mcml_env import MCML
from eps_greedy_policy import MyEpsGreedy

env = MCML()
ENV_NAME = 'mcml-test-4'

np.random.seed(123)
env.seed(123)
nb_actions = 4 ** len(env.action_space.nvec)

NB_STEPS = 1000000
# nb_actions = env.action_space # e.g 4**6 # sulution for action which is not a discrete ?
# for i in env.action_space.nvec:
#     nb_actions *= i

print(nb_actions, env.observation_space.shape)
# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape)) # input #input_shape = (1,) + (4,)
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(nb_actions)) # output
model.add(Activation('linear'))
print(model.summary())
# print(env.observation_space.sample())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
# policy = GreedyQPolicy()
#
policy = MyEpsGreedy(eps_max=0.9, eps_min=0, nb_steps=NB_STEPS)

processor = MCMLProcessor()

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy, enable_double_dqn=True, processor=processor)

# dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
#                target_model_update=1e-2, policy=policy, enable_double_dqn=False, processor=processor)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])
# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
# print(dqn.metrics_names[:])

learning_history = dqn.fit(env, nb_steps=NB_STEPS, visualize=False, verbose=2, nb_max_episode_steps=None)

reward_history = learning_history.history.get('episode_reward')
episode_history = np.arange(0, len(reward_history))
# print(reward_history)
# print(reward_history, episode_history)
# plot score and save image
pylab.plot(episode_history, reward_history, 'b')
pylab.savefig("./results/mcml-test-{}.png".format(ENV_NAME))
pylab.show()

# After training is done, we save the final weights.
dqn.save_weights('./results/dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
# dqn.test(env, nb_episodes=5, visualize=True)

