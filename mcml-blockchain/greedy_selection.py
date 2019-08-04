#  Greedy selection policy for mcml-block chain
# @author: Hieu Nguyen

import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter
import math

from environment import Environment, MyProcessor
from policy_epgreedy import MyEpsGreedy
from writer_v1 import MCMLWriter
from config import NB_DEVICES, CPU_SHARES, CAPACITY_MAX, ENERGY_MAX, DATA_MAX, FEERATE_MAX, MEMPOOL_MAX, MINING_RATE


TEST_ITERATOR = 216
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
    action = env.action_sample
    state = env.state

    energy = np.copy(action[SECOND_OFFSET:THIRD_OFFSET])
    cpu_shares = np.copy(state[FIRST_OFFSET:SECOND_OFFSET])
    capacity = np.copy(state[SECOND_OFFSET:THIRD_OFFSET])

    if IS_GREEDY == 0:
        data = np.random.randint(1, DATA_MAX, size=NB_DEVICES)
        mining_para = np.random.randint(0, MINING_RATE, size=NB_DEVICES)
        # TODO: choose random action
        for i in range(len(data)):
            if cpu_shares[i] == 0:
                energy[i] = 0
            else:
                e_threshold = TAU * NU * data[i] * (MU * capacity[i]) ** 2 / E_UNIT
                e_threshold = max(1, math.ceil(min(capacity[i], e_threshold)))
                energy[i] = np.random.randint(0, e_threshold)

        action = np.array([data, energy, mining_para]).flatten()
        action = action[:THIRD_OFFSET + 1]

    else:
        data = np.full(NB_DEVICES, DATA_MAX - 1)
        mining_para = np.full(NB_DEVICES, MINING_RATE - 1)
        # TODO: choose greedy action
        for i in range(len(data)):
            if cpu_shares[i] == 0:
                energy[i] = 0
            else:
                e_threshold = TAU * NU * data[i] * (MU * cpu_shares[i]) ** 2 / E_UNIT
                e_threshold = max(1, math.ceil(min(capacity[i], e_threshold)))
                energy[i] = np.random.randint(0, e_threshold)

        action = np.array([data, energy, mining_para]).flatten()
        action = action[:THIRD_OFFSET + 1]

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