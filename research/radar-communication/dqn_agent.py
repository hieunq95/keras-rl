import numpy as np
import gym
import argparse

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from environment import AV_Environment

TARGET_MODEL_UPDATE = 1e-3
NB_STEPS = 200000
EPSILON_LINEAR_STEPS = NB_STEPS * 2 / 3
TEST_ID = 4

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='AV_Radar-v1')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

env = AV_Environment()
nb_actions = env.action_space.n
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=EPSILON_LINEAR_STEPS)
memory = SequentialMemory(limit=50000, window_length=1)
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.nvec.shape))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

print(model.summary())

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=TARGET_MODEL_UPDATE, policy=policy,
               enable_double_dqn=False)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

print("****************************************"
      " Start of training {}-th " 
      "****************************************".format(TEST_ID))

if args.mode == 'train':
    weights_filename = './logs/dqn_{}_weights_{}.h5f'.format(args.env_name, TEST_ID)
    checkpoint_weights_filename = './logs/dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = './logs/dqn_{}_log_{}.json'.format(args.env_name, TEST_ID)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=NB_STEPS/2)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, nb_steps=NB_STEPS, visualize=False, verbose=2, nb_max_episode_steps=None, callbacks=callbacks)
    dqn.save_weights(weights_filename, overwrite=True)
    dqn.test(env, nb_episodes=100, visualize=False)
elif args.mode == 'test':
    weights_filename = './logs/dqn_{}_weights_{}.h5f'.format(args.env_name, TEST_ID)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=100, visualize=False)
