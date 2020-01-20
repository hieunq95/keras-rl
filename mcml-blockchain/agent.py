# Federated learning - block chain DQN agent
# @Author: Hieu Nguyen

import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from environment import Environment, MyProcessor
from policy_epgreedy import MyEpsGreedy
from writer_v1 import MCMLWriter

TEST_ITERATOR = 6
NB_STEPS = 2000000

NB_TEST_EPISODES = 1000
DECAY_EPSILON_END = 2000
TARGET_MODEL_UPDATE = 1e-3  # hard or soft update
"""
Iteration = 142, charging ~ exponential(1.0), penalty = 2
Iteration = 143, charging ~ exponential(1.0), penalty = 3
Iteration = 144, charging ~ exponential(1.0), penalty = 1   state -> int32
Iteration = 150, charging ~ poisson(1), penalty = 1 * 5
"""

mempool = []
workbook = xlsxwriter.Workbook('./build/results-{}.xlsx'.format(TEST_ITERATOR))
writer = MCMLWriter(workbook)
env = Environment(mempool, writer)

log_filename = './build/dqn_mcml_log.json'
# callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=NB_STEPS/2)]
callbacks = [FileLogger(log_filename, interval=100)]

policy = MyEpsGreedy(env, 0.9, 0.05, DECAY_EPSILON_END, writer)
# policy = GreedyQPolicy()
processor = MyProcessor()

nb_actions = 1
for i in env.action_space.nvec:
    nb_actions *= i

print(env.action_space.nvec, nb_actions, env.observation_space)
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))  #input
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

print(model.summary())

print("****************************************"
      " Start of training {}-th " 
      "****************************************".format(TEST_ITERATOR))

memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=TARGET_MODEL_UPDATE, policy=policy,
               enable_double_dqn=True, processor=processor)
# TODO: what learning rate is efficient enough
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

history = dqn.fit(env, nb_steps=NB_STEPS, visualize=False, verbose=2, nb_max_episode_steps=None, callbacks=callbacks)

print("****************************************"
      " End of training {}-th - switch to test" 
      "****************************************".format(TEST_ITERATOR))
# dqn.test(env, nb_episodes=NB_TEST_EPISODES, visualize=False, nb_max_episode_steps=None)

workbook.close()
# plt.plot(np.arange(0, len(mempool)), mempool)
# plt.savefig('./build/mempool-test-{}.png'.format(TEST_ITERATOR))
# plt.show()
