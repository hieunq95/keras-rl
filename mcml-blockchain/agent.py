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

from environment import Environment, MyProcessor
from policy_epgreedy import MyEpsGreedy
from writer_v1 import MCMLWriter

TEST_ITERATOR = 4

mempool = []
workbook = xlsxwriter.Workbook('./build/results-{}.xlsx'.format(TEST_ITERATOR))
writer = MCMLWriter(workbook)

env = Environment(mempool, writer)

policy = MyEpsGreedy(env, 0.9, 0.0, 1000)
processor = MyProcessor()

nb_actions = 1
for i in env.action_space.nvec:
    nb_actions *= i

print(env.action_space.nvec, nb_actions)
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape)) # input
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(nb_actions)) # output
model.add(Activation('linear'))
print(model.summary())

memory = SequentialMemory(limit=5000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy,
               enable_double_dqn=True, processor=processor)
# TODO: what learning rate is efficient enough
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=300000, visualize=False, verbose=2)

workbook.close()
plt.plot(np.arange(0, len(mempool)), mempool)
plt.savefig('./build/mempool-test-{}.png'.format(TEST_ITERATOR))
plt.show()
