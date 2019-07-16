# Mobile crowd machine learning block chain
# @author: Hieunq

import numpy as np
import math
import gym
import xlsxwriter
from gym import spaces
from gym.utils import seeding
from rl.core import Processor
from config import NB_DEVICES, CPU_SHARES, CAPACITY_MAX, ENERGY_MAX, DATA_MAX, FEERATE_MAX, MEMPOOL_MAX, MEMPOOL_INIT, MEMPOOL_SLOPE, TERMINATION_STEPS, INTERARRIVAL_RATE

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

        action_array = np.array([DATA_MAX, ENERGY_MAX, FEERATE_MAX]).repeat(NB_DEVICES)
        self.action_space = spaces.MultiDiscrete(action_array)

        self.TERMINATION = np.int(np.random.poisson(TERMINATION_STEPS))
        # self.TERMINATION = 1000
        self.ACTION_PENALTY = 0

        self.mempool_state = MEMPOOL_INIT
        self.mempools = mempool

        self.accumulated_data = 0
        self.episode_counter = 0
        self.episode_reward = 0
        self.step_counter = 0
        self.step_total = 0
        self.energy_per_episode = 0.0
        self.latency_per_episode = 0.0
        self.confirm_probability = 0.0
        self.nb_waiting_blocks = 0
        self.DONE_FLAG = False
        self.payment_per_episode = 0.0

        self.ACTION_PENALTY_EP = 0

        self.estimated_feerate = 0

        self.seed(123)
        self.reset()

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
        capacity_array = np.copy(state[SECOND_OFFSET:THIRD_OFFSET])
        energy_array = np.copy(action[SECOND_OFFSET:THIRD_OFFSET])
        charging_array = self.nprandom.poisson(1, size=len(energy_array))
        cpu_shares_array = self.nprandom.randint(CPU_SHARES, size=NB_DEVICES)
        next_capacity_array = np.zeros(len(capacity_array), dtype=np.int32)
        # TODO: mempool's transition
        mempool_state = self.nprandom.poisson(INTERARRIVAL_RATE, size=NB_DEVICES)
        # TODO: state trasition - should do transition when mismatch the constrain ?
        for i in range(len(next_capacity_array)):
            next_capacity_array[i] = min(capacity_array[i] - energy_array[i] + charging_array[i], CAPACITY_MAX-1)

        next_state = np.array([cpu_shares_array, next_capacity_array, mempool_state], dtype=np.int32).flatten()
        next_state = next_state[:THIRD_OFFSET + 1]

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
        # energy = np.copy(action[self.ENERGY_OFFSET:])
        cpu_cycles = self._calculate_cpu_cycles(energy, data)
        latency_array = np.zeros(len(cpu_cycles))
        for k in range(len(cpu_cycles)):
            if cpu_cycles[k] != 0:
                latency_array[k] = 10 ** 10 * data[k] / cpu_cycles[k]

        latency = np.amax(latency_array)
        # latency = np.average(latency_array)
        return latency

    def _get_confirm_prob(self, x, n, r):
        """
        Calculate the probability of a transaction with fee rate r are confirmed in at least n blocks
        :param x: mempool state with dtype = int

        :param n: number of blocks that users should wait for confirmation

        :param r: Fee rate of the transaction

        :return: Probability of confirmation within n blocks, should be greater than 95%
        """
        # #TODO: Calculate the slope of mempool as a function of r, i.e. arrival rate
        # c(r) = 1 - 0.25 * (r + 1)
        c = 0.25
        y = np.max([(n - x) / c, 0])  # max((n - x) / c, 0)
        sigma = np.array([y ** k * np.exp(-y) / math.factorial(k) for k in range(n)]).sum()
        prob = 1 - sigma
        return prob

    def get_reward(self, action):
        tau = 10 ** (-28)
        nu = 10 ** 10
        delta = 1
        training_price = 0.2

        alpha_D = 5
        alpha_L = 2
        alpha_E = 1
        alpha_I = 2
        REWARD_PENATY = 0.5

        data = np.copy(action[self.DATA_OFFSET:self.ENERGY_OFFSET])
        energy = np.copy(action[self.ENERGY_OFFSET:self.FEERATE_OFFSET])
        confirm_fee = np.copy(action[self.FEERATE_OFFSET:])

        mempool_state = int(np.copy(self.state[-1]))
        # payment = [data[k] * (training_price + np.log(1 + mempool_state)) for k in range(len(data))]
        payment = [training_price * data[k] + confirm_fee[k] * np.log(1 + mempool_state) for k in range(len(data))]

        ENERGY_THRESOLD = (ENERGY_MAX-1) * NB_DEVICES
        DATA_THRESOLD = (DATA_MAX-1) * NB_DEVICES
        LATENCY_THRESOLD = (tau ** 0.5) * (nu ** 1.5) * (delta ** (-0.5)) * (DATA_MAX-1) ** 1.5
        # PAYMENT_THRESOLD = (DATA_MAX - 1) * NB_DEVICES * (training_price + np.log(1 + MEMPOOL_MAX))
        PAYMENT_THRESOLD = NB_DEVICES * ((DATA_MAX - 1) * training_price + (FEERATE_MAX - 1) * np.log(1 + MEMPOOL_MAX))

        accumulated_data = np.sum(data)
        total_energy = np.sum(energy)
        latency = self.calculate_latency(action)
        payment = np.sum(payment)
        # TODO : shaping reward function
        reward = alpha_D * accumulated_data / DATA_THRESOLD - alpha_E * total_energy / ENERGY_THRESOLD \
                 - alpha_L * latency / LATENCY_THRESOLD - alpha_I * payment / PAYMENT_THRESOLD
        # d = accumulated_data / DATA_THRESOLD
        # e = total_energy / ENERGY_THRESOLD
        # l = alpha_L * latency / LATENCY_THRESOLD
        # if d > 1.0 or e > 1.0 or l > 1.0:
        #     print('d: {}, e: {}, l: {}'.format(d, e, l))
        # if payment / PAYMENT_THRESOLD > 1:
        #     print('p: {}, p/P: {}'.format(payment, payment / PAYMENT_THRESOLD))
        if self.ACTION_PENALTY > 0:
            reward -= REWARD_PENATY * self.ACTION_PENALTY

        self.ACTION_PENALTY_EP += self.ACTION_PENALTY
        self.payment_per_episode += payment
        # TODO: should be max or min feerate ?
        # TODO: derive the reward for delta_feerate

        return reward

    def _correct_action(self, cpu_cycles, cpu_shares, energy):
        corrected_energy = np.copy(energy)
        for i in range(len(cpu_shares)):
            if cpu_cycles[i] > 0.6 * (10 ** 9) * cpu_shares[i]:
                # corrected_energy[i] = ENERGY_MAX + 1
                # corrected_energy[i] = 0
                self.ACTION_PENALTY += 1

        return corrected_energy

    def _calculate_cpu_cycles(self, energy, data):
        # print('data {}'.format(data))
        cpu_cycles = np.zeros(len(energy))
        for i in range(len(data)):
            if data[i] != 0 and energy[i] != 0:
                cpu_cycles[i] = np.sqrt(1 * energy[i]) / np.sqrt((10 ** -18) * data[i])
                # cpu_cycles[i] = 1 * energy[i] / ((10 ** (-18)) * data[i])
                # cpu_cycles[i] = cpu_cycles[i] ** 0.5
            else:
                cpu_cycles[i] = 0

        return cpu_cycles

    def check_action(self, action):
        """
        Check action constrain and correct action as follows:

        - if energy required > current capacity, then let energy required = current capacity

        - if energy or data = 0, then both should be 0

        - if cpu_cycle_required > u * cpu_shares, then let self.ACTION_PENALTY = number_of_mismatch_actions to notice agent should not make any state transition for this device and get a penalty

        :param data_array:
        :param energy_array:
        :return: corrected action
        """
        state = self.state

        cpushares_array = np.copy(state[self.DATA_OFFSET:self.ENERGY_OFFSET])
        capacity_array = np.copy(state[self.ENERGY_OFFSET:self.FEERATE_OFFSET])
        data_array = np.copy(action[self.DATA_OFFSET:self.ENERGY_OFFSET])
        energy_array = np.copy(action[self.ENERGY_OFFSET:self.FEERATE_OFFSET])
        confirm_fee_array = np.copy(action[self.FEERATE_OFFSET:])
        # TODO: correct the constrain, do not change the action, just count for a penalty of the action miss the constrain
        for i in range(len(energy_array)):
            # TODO: how to choose an action when energy required exceed the capacity
            if energy_array[i] > capacity_array[i]:
                # energy_array[i] = capacity_array[i]
                energy_array[i] = 0
                # energy_array[i] = np.random.randint(0, capacity_array[i]+1)
                self.ACTION_PENALTY += 0
            if data_array[i] == 0 or energy_array[i] == 0:
                # energy_array[i] = 0
                # data_array[i] = 0
                self.ACTION_PENALTY += 2

        cpu_cyles_array = self._calculate_cpu_cycles(energy_array, data_array)
        new_energy_array = self._correct_action(cpu_cyles_array, cpushares_array, energy_array)

        corrected_action = np.array([data_array, new_energy_array, confirm_fee_array]).flatten()
        return corrected_action

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        # print('action {}'.format(action))
        self.step_counter += 1
        self.ACTION_PENALTY = 0
        # print('before-correction {}'.format(self.ACTION_PENALTY))
        corrected_action = self.check_action(action)
        # print('after-corrections {}'.format(self.ACTION_PENALTY))
        state = np.copy(self.state)
        # State transition, action is taken from Processor.process_action()
        next_state = self.state_transition(state, corrected_action)

        self.mempool_state = next_state[-1]
        self.mempools.append(self.mempool_state)
        # TODO: corrected_action is somehow incorrect
        reward = self.get_reward(corrected_action)

        # For statistic only
        self.step_total += 1
        self.episode_reward += reward

        energy = np.copy(corrected_action[self.ENERGY_OFFSET:])
        data = np.copy(corrected_action[self.DATA_OFFSET:self.ENERGY_OFFSET])
        latency = self.calculate_latency(corrected_action)
        self.energy_per_episode += np.sum(energy)
        self.latency_per_episode += latency
        self.accumulated_data += np.sum(data)

        # End of statistic

        # TODO: terminated condition ?
        # if self.step_counter == self.TERMINATION:
        if self.accumulated_data >= 1000:
            # print('accumulated_data {}'.format(self.accumulated_data))
            done = True
            # For statistic only
            self.episode_counter += 1
            logs = {
                    'episode': self.episode_counter,
                    'episode_reward': self.episode_reward,
                    'energy': 1.0 * self.energy_per_episode / self.step_counter,
                    'latency': 1.0 * self.latency_per_episode / self.step_counter,
                    'step_total': self.step_total,
                    'episode_steps': self.step_counter,
                    'reward_mean': self.episode_reward / self.step_counter,
                    'training_data_mean': self.accumulated_data / self.step_counter,
                    'action_penalty': self.ACTION_PENALTY_EP / self.step_counter,
                    'mempool_state': self.mempool_state,
                    'waiting_blocks': self.nb_waiting_blocks,
                    'confirm_prob': self.confirm_probability,
                    'feerate_from_cdf': self.estimated_feerate,
                    'payment': self.payment_per_episode / self.step_counter,
                   },
            # export results to excel file
            self.writer.general_write(logs, self.episode_counter)
            # print('************** d:  {}, e:  {}, l:  {} ***************'
            #       .format(self.accumulated_data / self.step_counter,
            #                 self.energy_per_episode / self.step_counter,
            #                 self.latency_per_episode / self.step_counter))
            # End of statistic
        else:
            done = False
        self.DONE_FLAG = done
        self.state = next_state

        return np.array(self.state), reward, done, {}

    def reset(self):
        cpu_shares_init = self.nprandom.randint(CPU_SHARES, size=NB_DEVICES)
        capacity_init = self.nprandom.randint(CAPACITY_MAX, size=NB_DEVICES)
        mempool_init = np.full((NB_DEVICES, ), min(self.nprandom.poisson(INTERARRIVAL_RATE), MEMPOOL_MAX))
        state = np.array([cpu_shares_init, capacity_init, mempool_init]).flatten()
        state = state[:self.FEERATE_OFFSET + 1]

        self.mempool_state = state[-1]
        self.mempools.append(self.mempool_state)

        self.state = state
        self.TERMINATION = np.int(self.nprandom.poisson(TERMINATION_STEPS))
        # self.TERMINATION = 1000
        self.ACTION_PENALTY = 0
        self.DONE_FLAG = False

        self.estimated_feerate = 0
        self.delta_feerate = 0

        self.nb_waiting_blocks = 0
        # For statistic only
        self.step_counter = 0
        self.episode_reward = 0
        self.energy_per_episode = 0
        self.latency_per_episode = 0
        self.confirm_probability = 0.0
        self.accumulated_data = 0
        self.ACTION_PENALTY_EP = 0
        self.payment_per_episode = 0.0
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
        self.metrics_names = ''
        # TODO: check order of converter
        self.DATA_ORDER = DATA_MAX
        self.ENERGY_ORDER = ENERGY_MAX
        self.FEERATE_ORDER = FEERATE_MAX

        # self.ACTION_SIZE = 3 * NB_DEVICES
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
        # print('pre-process action {}'.format(action))
        data_array, energy_array, feerate_array = [], [], []
        action_clone = np.copy(action)

        for i in range(self.ACTION_SIZE):
            if i < self.ENERGY_OFFSET:
                divisor = self.DATA_ORDER ** (self.ACTION_SIZE - (i + 1))
                data_i = action_clone // divisor
                action_clone -= data_i * divisor
                data_array.append(data_i)
            # else:
            #     divisor = self.ENERGY_ORDER ** (self.ACTION_SIZE - (i + 1))
            #     energy_i = action_clone // divisor
            #     action_clone -= energy_i * divisor
            #     energy_array.append(energy_i)

            elif i >= self.ENERGY_OFFSET and i < self.FEERATE_OFFSET:
                divisor = self.ENERGY_ORDER ** (self.ACTION_SIZE - (i + 1))
                energy_i = action_clone // divisor
                action_clone -= energy_i * divisor
                energy_array.append(energy_i)

            elif i >= self.FEERATE_OFFSET:
                divisor = self.FEERATE_ORDER ** (self.ACTION_SIZE - (i + 1))
                feerate_i = action_clone // divisor
                action_clone -= feerate_i * divisor
                feerate_array.append(feerate_i)
        #TODO: is the q_values is sorted in the corrected order

        processed_action = np.array([data_array, energy_array, feerate_array]).flatten()
        return processed_action


    def process_action(self, action):
        processed_action = self._convert_action(action)
        return processed_action

    def metrics_names(self):
        return metrics_names

    def process_observation(self, observation):
        return observation

    def process_reward(self, reward):
        return reward


