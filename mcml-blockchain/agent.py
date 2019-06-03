# Federated learning - block chain DQN agent
# @Author: Hieu Nguyen

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from environment import Environment, MyProcessor

env = Environment()
processor = MyProcessor()
policy = GreedyQPolicy()

nb_actions = 1
for i in env.action_space.nvec:
    nb_actions *= i

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

memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy,
               enable_double_dqn=True, processor=processor)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
