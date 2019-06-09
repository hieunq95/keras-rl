# Mobile crowd machine learning block chain
# @author: Hieunq

import numpy as np
import math
import gym
import xlsxwriter
from gym import spaces
from gym.utils import seeding
from rl.core import Processor
from config import NB_DEVICES, CPU_SHARES, CAPACITY_MAX, ENERGY_MAX, DATA_MAX, FEERATE_MAX, MEMPOOL_MAX, MEMPOOL_INIT

class Environment(gym.Env):

    """
            - state_space : {f1,f2,...,F, c1, c2, ..., C, m1, m2, ..., M}

            - action_space: {d1, d2, ..., D, e1, e2, ..., E, r1, r2, ..., R}
            """
    def __init__(self, mempool, writer):
        self.writer = writer
        self.DATA_OFFSET = 0
        self.ENERGY_OFFSET = NB_DEVICES
        self.FEERATE_OFFSET = 2 * NB_DEVICES

        action_array = np.array([DATA_MAX, ENERGY_MAX, FEERATE_MAX])
        action_array = np.repeat(action_array, NB_DEVICES)

        state_lower_bound = np.array([0, 0, MEMPOOL_INIT]).repeat(NB_DEVICES)
        state_lower_bound = state_lower_bound[:self.FEERATE_OFFSET + 1]

        state_upper_bound = np.array([CPU_SHARES, CAPACITY_MAX, MEMPOOL_MAX]).repeat(NB_DEVICES)
        state_upper_bound = state_upper_bound[:self.FEERATE_OFFSET + 1]

        self.observation_space = spaces.Box(low=state_lower_bound, high=state_upper_bound, dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(action_array)

        self.mempool_state = MEMPOOL_INIT
        self.mempools = mempool

        self.accumulated_data = 0
        self.episode_counter = 0
        self.episode_reward = 0
        self.step_counter = 0
        self.step_total = 0
        self.energy_per_episode = 0.0
        self.latency_per_episode = 0.0

        self.DONE_FLAG = False

        self.seed(10000)
        self.reset()
        self.step_counter = 0

    def state_transition(self, state, action):
        """
        Perform state transition

        :param state: current state

        :param action: a taken action

        :return: next state
        """

        SECOND_OFFSET = NB_DEVICES
        THIRD_OFFSET = 2 * NB_DEVICES
        # TODO: should be random each episode or each step ?
        cpu_shares_array = np.random.randint(0, CPU_SHARES, size=NB_DEVICES)
        capacity_array = np.copy(state[SECOND_OFFSET:THIRD_OFFSET])
        mempool_array = np.copy(state[THIRD_OFFSET:])

        energy_array = np.copy(action[SECOND_OFFSET:THIRD_OFFSET])
        charging_array = np.random.poisson(1, size=len(energy_array))

        next_capacity_array = np.zeros(NB_DEVICES)
        next_mempool_array = np.copy(mempool_array)
        # TODO: mempool's transition
        for i in range(len(next_mempool_array)):
            if self.step_counter < self.TERMINATION:
                next_mempool_array[i] = mempool_array[i] + 0.95 * np.random.exponential(1.0/self.TERMINATION)
            else:
                next_mempool_array[i] = max(mempool_array[i] - np.random.exponential(1), 0)

        for i in range(len(capacity_array)):
            if energy_array[i] > ENERGY_MAX:
                next_capacity_array[i] = capacity_array[i]
            else:
                next_capacity_array[i] = min(capacity_array[i] - energy_array[i] + charging_array[i], CAPACITY_MAX)
        # Broadcast the size of mempool to the size of capacity
        next_mempool_array = next_mempool_array.repeat(NB_DEVICES)
        next_state = np.array([cpu_shares_array, next_capacity_array, next_mempool_array], dtype=np.float32).flatten()
        # Get first seven elements
        next_state = next_state[:THIRD_OFFSET+1]

        return next_state

    def calculate_latency(self, action):
        """
        Calculate traning latency
        :param action: Taken action

        :return: Latency of the training step
        """
        # print('calculate_latency: action {}'.format(action))
        data = np.copy(action[self.DATA_OFFSET:self.ENERGY_OFFSET])
        energy = np.copy(action[self.ENERGY_OFFSET:self.FEERATE_OFFSET])

        cpu_cycles = self._calculate_cpu_cycles(energy, data)
        latency_array = np.zeros(len(cpu_cycles))
        for k in range(len(cpu_cycles)):
            if cpu_cycles[k] != 0:
                latency_array[k] = 10 ** 10 * data[k] / cpu_cycles[k]

        latency = max(latency_array)

        return latency

    def _get_confirm_prob(self, x, n, r):
        """
        Calculate the probability of a transaction with fee rate r are confirmed in at least n blocks
        :param x: mempool state with dtype = int

        :param n: number of blocks that users should wait for confirmation

        :param r: Fee rate of the transaction

        :return: Probability of confirmation within n blocks, should be greater than 95%
        """
        # Calculate the slope of mempool as a function of r, i.e. arrival rate
        # c(r) = 1 - 0.25 * (r + 1)
        c = 1 - 0.25 * (r + 1)
        y = np.max([(n - x) / c, 0])  # max((n - x) / c, 0)
        sigma = np.array([y ** k * np.exp(-y) / math.factorial(k) for k in range(n)]).sum()
        prob = 1 - sigma
        return prob

    def get_reward(self, action):
        tau = 10 ** (-28)
        nu = 10 ** 10
        delta = 1
        alpha_D = 3
        alpha_L = 1
        alpha_E = 1
        REWARD_BASE = 2
        REWARD_PENATY = 0

        data = np.copy(action[self.DATA_OFFSET:self.ENERGY_OFFSET])
        energy = np.copy(action[self.ENERGY_OFFSET:self.FEERATE_OFFSET])
        feerate = np.copy(action[self.FEERATE_OFFSET:])

        ENERGY_THRESOLD = ENERGY_MAX * NB_DEVICES
        DATA_THRESOLD = DATA_MAX * NB_DEVICES
        LATENCY_THRESOLD = (tau ** 0.5) * (nu ** 1.5) * (delta ** (-0.5)) * DATA_MAX ** 1.5

        accumulated_data = np.sum(data)
        total_energy = np.sum(energy)
        latency = self.calculate_latency(action)

        reward = alpha_D * accumulated_data / DATA_THRESOLD - alpha_E * total_energy / ENERGY_THRESOLD\
                 - alpha_L * latency / LATENCY_THRESOLD
        reward *= 10
        reward += REWARD_BASE

        current_block = self.state[-1]
        current_block = np.int(current_block)
        feerate_min = np.min(feerate)
        fastest_confirm_prob = self._get_confirm_prob(current_block, current_block + 1, feerate_min)

        if fastest_confirm_prob < 0.95:
            reward -= REWARD_PENATY
        # else:
            # print(fastest_confirm_prob)

        return reward

    def _correct_action(self, cpu_cycles, cpu_shares, energy):
        corrected_energy = np.copy(energy)
        for i in range(len(cpu_shares)):
            if cpu_cycles[i] > 0.6 * 10 ** 9 * cpu_shares[i]:
                corrected_energy[i] = ENERGY_MAX + 1

        return corrected_energy

    def _calculate_cpu_cycles(self, energy, data):
        # print('data {}'.format(data))
        cpu_cycles = np.zeros(len(energy))
        for i in range(len(data)):
            if data[i] != 0:
                cpu_cycles[i] = np.sqrt(1 * energy[i]) / np.sqrt((10 ** -18) * data[i])

        return cpu_cycles

    def check_action(self, action):
        """
        Check action constrain and correct action as follows:

        - if energy required > current capacity, then let energy required = current capacity

        - if energy or data = 0, then both should be 0

        - if cpu_cycle_required > u * cpu_shares, then let energy_action = MAX_ACTION + 1 to notice agent should not  make any state transition for this device

        :param data_array:
        :param energy_array:
        :return: corrected action
        """
        state = self.state

        cpushares_array = np.copy(state[self.DATA_OFFSET:self.ENERGY_OFFSET])
        capacity_array = np.copy(state[self.ENERGY_OFFSET:self.FEERATE_OFFSET])
        data_array = np.copy(action[self.DATA_OFFSET:self.ENERGY_OFFSET])
        energy_array = np.copy(action[self.ENERGY_OFFSET:self.FEERATE_OFFSET])
        feerate_array = np.copy(action[self.FEERATE_OFFSET:])


        for i in range(len(energy_array)):
            if data_array[i] == 0 or energy_array[i] == 0:
                energy_array[i] = 0
                data_array[i] = 0

            if energy_array[i] > capacity_array[i]:
                energy_array[i] = capacity_array[i]

        cpu_cyles_array = self._calculate_cpu_cycles(energy_array, data_array)
        new_energy_array = self._correct_action(cpu_cyles_array, cpushares_array, energy_array)
        corrected_action = np.array([data_array, new_energy_array, feerate_array]).flatten()

        return corrected_action

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        # print('action {}'.format(action))
        self.step_counter += 1
        corrected_action = self.check_action(action)
        # print('corrected_action {}'.format(corrected_action))
        # TODO : termination as block generation time follows exponential distribution with mean = 600
        TERMINATION = self.TERMINATION

        state = self.state
        # State transition, action is taken from Processor.process_action()
        next_state = self.state_transition(state, corrected_action)

        self.mempool_state = next_state[-1]
        self.mempools.append(self.mempool_state)

        reward = self.get_reward(corrected_action)

        # For statistic only
        self.step_total += 1
        self.episode_reward += reward
        energy = np.copy(corrected_action[self.ENERGY_OFFSET:self.FEERATE_OFFSET])
        data = np.copy(corrected_action[self.DATA_OFFSET:self.ENERGY_OFFSET])
        latency = self.calculate_latency(corrected_action)
        self.energy_per_episode += np.sum(energy)
        self.latency_per_episode += latency
        self.accumulated_data += data
        # End of statistic

        if self.step_counter == TERMINATION:
            done = True
            # For statistic only
            self.episode_counter += 1
            logs = {
                       'episode': self.episode_counter,
                       'episode_reward': self.episode_reward,
                       'energy': 100.0 * self.energy_per_episode / self.step_counter,
                       'latency': 100.0 * self.latency_per_episode / self.step_counter,
                       'step_total': self.step_total,
                       'episode_steps': self.step_counter,
                       'reward_mean': self.episode_reward / self.step_counter,
                       # 'training_data_mean': self.accumulated_data / self.step_counter,
                   },
            print(self.mempool_state)
            # export results to excel file
            self.writer.general_write(logs, self.episode_counter)
            # End of statistic
        else:
            done = False

        self.DONE_FLAG = done
        self.state = next_state
        return np.array(self.state), reward, done, {}

    def reset(self):
        cpu_shares_init = self.nprandom.randint(CPU_SHARES, size=NB_DEVICES)
        capacity_init = self.nprandom.randint(CAPACITY_MAX, size=NB_DEVICES)
        mempool_init = np.full((NB_DEVICES, ), self.mempool_state)
        state = np.array([cpu_shares_init, capacity_init, mempool_init]).flatten()
        state = state[:self.FEERATE_OFFSET + 1]

        self.mempool_state = state[-1]
        self.mempools.append(self.mempool_state)

        self.state = state
        self.TERMINATION = np.int(np.random.poisson(200))
        self.DONE_FLAG = False

        # For statistic only
        self.step_counter = 0
        self.episode_reward = 0
        self.energy_per_episode = 0
        self.latency_per_episode = 0
        # End of statistic

        return self.state

    def seed(self, seed=None):
        self.nprandom, seed = seeding.np_random(seed)
        return [seed]

    def is_terminated(self):
        """
        Get variables from current environment

        :return: Config info
        """
        return self.DONE_FLAG

class MyProcessor(Processor):

    def __init__(self):
        self.metrics_names = []

        self.DATA_ORDER = DATA_MAX + 1
        self.ENERGY_ORDER = ENERGY_MAX + 1
        self.FEERATE_ORDER = FEERATE_MAX + 1

        self.ACTION_SIZE = 3 * NB_DEVICES
        self.DATA_OFFSET = 0
        self.ENERGY_OFFSET = NB_DEVICES
        self.FEERATE_OFFSET = 2 * NB_DEVICES


    def _convert_action(self, action):
        """
            Convert action from decima to array
            :param action: input action in decima

            :return: action in array
            """

        data_array, energy_array, feerate_array = [], [], []
        action_clone = np.copy(action)

        for i in range(self.ACTION_SIZE):
            if i < self.ENERGY_OFFSET:
                divisor = self.DATA_ORDER ** (self.ACTION_SIZE - (i + 1))
                data_i = action_clone // divisor
                action_clone -= data_i * divisor
                data_array.append(data_i)

            elif (i < self.FEERATE_OFFSET):
                divisor = self.ENERGY_ORDER ** (self.ACTION_SIZE - (i + 1))
                energy_i = action_clone // divisor
                action_clone -= energy_i * divisor
                energy_array.append(energy_i)

            else:
                divisor = self.FEERATE_ORDER ** (self.ACTION_SIZE - (i + 1))
                feerate_i = action_clone // divisor
                action_clone -= feerate_i * divisor
                feerate_array.append(feerate_i)


        processed_action = np.array([data_array, energy_array, feerate_array]).flatten()

        return processed_action


    def process_action(self, action):
        processed_action = self._convert_action(action)
        return processed_action

    def metrics_names(self):
        return metrics_names

    def process_observation(self, observation):
        return observation


