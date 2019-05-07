# From dqn_cartpole.py

import numpy as np
import gym
import pylab
import xlsxwriter

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, GreedyQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from mcml_processor import MCMLProcessor
from mcml_env import MCML
from eps_greedy_policy import MyEpsGreedy
from parameters import Parameters
from results_writer import MCMLWriter


parameters = Parameters()
workbook = xlsxwriter.Workbook(parameters.XLSX_PATH)
writer = MCMLWriter(workbook)
env = MCML(writer)
policy = MyEpsGreedy(environment=env, eps_max=0.9, eps_min=0,
                     eps_training=parameters.EPISODES_TRAINING, writer=writer)
processor = MCMLProcessor()
nb_actions = 4 ** len(env.action_space.nvec)

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

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy, enable_double_dqn=True, processor=processor)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])
# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
# print(dqn.metrics_names[:])
dqn.fit(env, nb_steps=parameters.NB_STEPS, visualize=False, verbose=2, nb_max_episode_steps=None)

workbook.close()

# Finally, evaluate our algorithm for 5 episodes.
# dqn.test(env, nb_episodes=5, visualize=False)

