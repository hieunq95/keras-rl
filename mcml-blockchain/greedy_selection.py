#  Greedy selection policy for mcml-block chain
# @author: Hieu Nguyen

import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter

from environment import Environment, MyProcessor
from policy_epgreedy import MyEpsGreedy
from writer_v1 import MCMLWriter
from config import NB_DEVICES, CPU_SHARES, CAPACITY_MAX, ENERGY_MAX, DATA_MAX, FEERATE_MAX, MEMPOOL_MAX

TEST_ITERATOR = 200
NB_STEP = 1500000 * 3

IS_GREEDY = 0

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

for i in range(NB_STEP):
    if IS_GREEDY:
        action = np.full(3*NB_DEVICES, ENERGY_MAX-1)
    else:
        action = np.random.randint(ENERGY_MAX-1, size=3*NB_DEVICES)
    observation, reward, done, info = env.step(action)
    ep_reward += reward
    ep_steps += 1
    if done:
        episode += 1
        print('Episode: {}, Steps: {}, Ep_steps {},total reward {}'.format(episode, i, ep_steps, ep_reward))
        ep_reward = 0
        ep_steps = 0
        env.reset()

workbook.close()