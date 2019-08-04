#  Greedy selection policy for mcml-block chain
# @author: Hieu Nguyen

import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter
import math

from environment import Environment, MyProcessor
from policy_epgreedy import MyEpsGreedy
from writer_v1 import MCMLWriter
from config import NB_DEVICES, CPU_SHARES, CAPACITY_MAX, ENERGY_MAX, DATA_MAX, FEERATE_MAX, MEMPOOL_MAX

TEST_ITERATOR = 208
NB_STEP = 1500000 * 2

IS_GREEDY = 1

if IS_GREEDY:
    workbook = xlsxwriter.Workbook('./build/results-greedy-{}.xlsx'.format(TEST_ITERATOR))
else:
    workbook = xlsxwriter.Workbook('./build/results-random-{}.xlsx'.format(TEST_ITERATOR))

print('************ Start test {}, IS_GREEDY {}'.format(TEST_ITERATOR, IS_GREEDY))

writer = MCMLWriter(workbook)
mempool = []
env = Environment(mempool, writer)
env.reset()
episode = 0
ep_reward = 0
ep_steps = 0

FIRST_OFFSET = 0
SECOND_OFFSET = NB_DEVICES
THIRD_OFFSET = 2 * NB_DEVICES

E_UNIT = 1
NU = 10 ** 10
TAU = 10 ** (-28)
MU = 0.6 * (10 ** 9)

for e in range(NB_STEP):
    if IS_GREEDY == 0:
        # print(env.action_sample)
        action = env.action_sample
        state = env.state

        energy = np.copy(action[SECOND_OFFSET:THIRD_OFFSET])
        fee_rate = np.copy(action[THIRD_OFFSET:])
        cpu_shares = np.copy(state[FIRST_OFFSET:SECOND_OFFSET])
        data_min = np.zeros(len(energy))
        data = np.zeros(len(energy))
        for i in range(len(energy)):
            if cpu_shares[i] == 0:
                data_min[i] = 0
            else:
                data_min[i] = math.ceil(E_UNIT * energy[i] / (TAU * NU * (MU * cpu_shares[i])**2))

            if data_min[i] < DATA_MAX:
                data[i] = np.random.randint(low=data_min[i], high=DATA_MAX)
            else:
                data[i] = 0
        # TODO: choose random action
        # action = np.array([data, energy, fee_rate]).flatten()
        data = np.random.randint(0, DATA_MAX, size=NB_DEVICES)
        energy = np.random.randint(0, ENERGY_MAX, size=NB_DEVICES)
        action = np.array([data, energy, fee_rate]).flatten()

    else:
        action = env.action_sample
        state = env.state

        data = np.full(NB_DEVICES, DATA_MAX - 1, dtype=int)
        energy = np.zeros(NB_DEVICES)
        cpu_shares = np.copy(state[FIRST_OFFSET:SECOND_OFFSET])
        fee_rate = np.copy(action[THIRD_OFFSET:])

        for i in range(NB_DEVICES):
            energy_max = max(math.ceil(((MU * cpu_shares[i]) ** 2) * TAU * NU * data[i] / E_UNIT) - 1, 0)
            # energy[i] = np.random.randint(low=0, high=ENERGY_MAX)

            energy[i] = min(energy_max, ENERGY_MAX - 1)

        # TODO: choose greedy action

        action = np.array([data, energy, fee_rate]).flatten()
        # print(action)

    observation, reward, done, info = env.step(action)
    ep_reward += reward
    ep_steps += 1
    if episode > 4000:
        break
    if done:
        episode += 1
        print('Episode: {}, Steps: {}, Ep_steps {},total reward {}'.format(episode, e, ep_steps, ep_reward))
        ep_reward = 0
        ep_steps = 0
        env.reset()

workbook.close()