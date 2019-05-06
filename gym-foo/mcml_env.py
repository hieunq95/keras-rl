# Mobile cloud machine learning (MCML) environment
# Author: Hieunq

import gym
import numpy as np
import xlsxwriter
from xlrd import open_workbook
from xlutils.copy import copy
from gym import spaces
from gym.utils import seeding
from parameters import Parameters

# see parameters.py
RESULT_PATH = 'results-2.xlsx'

parameters = Parameters()
workbook = xlsxwriter.Workbook(RESULT_PATH)
worksheet = workbook.add_worksheet()
# worksheet.set_column('A:A', 20)
bold = workbook.add_format({'bold': True})
worksheet.write('A1', 'Episode', bold)
worksheet.write('B1', 'Episode_steps', bold)
worksheet.write('C1', 'Total_reward', bold)
worksheet.write('D1', 'Mean_reward', bold)
worksheet.write('E1', 'Energy', bold)
worksheet.write('F1', 'Latency', bold)
worksheet.write('G1', 'Training_data', bold)

def _to_excel(info, episode):
    """
    Write simulation results into excel file
    :param info: input dictionary
    :param episode: Current episode
    :return: output file
    """
    """
    logs = {
                'episode':  self.episode_counter,
                'episode_reward': self.episode_reward,
                'energy': 100. * total_energy / parameters.ENERGY_THRESOLD, # propotion of energy required
                'latency': latency_max,
                'step_total': self.step_total,
                'episode_steps': self.step_counter,
                'reward_mean': self.episode_reward / self.step_counter
               },
    """

    info = dict(*info)
    worksheet.write(episode, 0, episode)
    worksheet.write(episode, 1, info.get('episode_steps'))
    worksheet.write(episode, 2, info.get('episode_reward'))
    worksheet.write(episode, 3, info.get('reward_mean'))
    worksheet.write(episode, 4, info.get('energy'))
    worksheet.write(episode, 5, info.get('latency'))
    worksheet.write(episode, 6, info.get('training_data'))

    # print("step_total {}, episode {}, episode_steps {}, episode_reward {}, mean_reward {}, "
    #       "energy {}, latency {}".
    #     format(info.get('step_total'), episode, info.get('episode_steps'), info.get('episode_reward'),
    #       info.get('reward_mean'), info.get('energy'), info.get('latency')))

class MCML(gym.Env):
    """
    Mobile cloud machine learning environment
    Simple environfment: 1 mobile agent act as a server
    - State space: S = {(f , c ) ; f ∈ {0, 1, . . . , Fn } , c ∈ {0, 1, . . . , Cn } ,
    - Action space: A = {(d , e);d ≤ Dn , e ≤ cn , f ̄≤ μf, n = (1, . . . , N )}
    - Transition: c' = c - e + A
    """
    def __init__(self):

        high_action = np.array([parameters.DATA_MAX+1, parameters.ENERGY_MAX+1])
        high_action = np.repeat(high_action, parameters.NB_DEVICES)

        # For simplicity, assume F_n = C_n
        self.observation_space = spaces.Box(low=0, high=parameters.CPUSHARE_MAX,
                                            shape=(2 * parameters.NB_DEVICES,), dtype=int)
        # MultiDiscrete samples all actions at one time and return random value for each action !
        self.action_space = spaces.MultiDiscrete(high_action) # A = {(d_1, e_1, ..., d_N, e_N)}
        self.accumulated_data = 0
        self.episode_counter = 0
        self.episode_reward = 0
        self.step_counter = 0
        self.step_total = 0
        self.seed()
        self.reset()

    def step(self, action):
        # print("debug: action {}".format(action))
        # print("debug: action.sample() {}".format(self.action_space.sample()))
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        self.step_counter += 1
        self.step_total += 1
        # extract state information
        f_cpu = np.random.randint(0, parameters.CPUSHARE_MAX+1, size=3) # fn get an random action
        c_unit = state.reshape([2, parameters.NB_DEVICES])[1].copy() # copy cn from current state

        # extract action information, add fn penalty later
        d_unit = np.asarray(action).reshape([2, parameters.NB_DEVICES])[0].copy()
        e_unit = np.asarray(action).reshape([2, parameters.NB_DEVICES])[1].copy() # copy last 3 elements

        #state transition
        # TODO: add An
        A_n = np.random.poisson(parameters.LAMBDA, size=parameters.NB_DEVICES)
        c_unit_next = np.subtract(c_unit, e_unit)
        c_unit_next = np.asarray([c_unit_next[i] + A_n[i] for i in range(len(c_unit_next))])
        c_MAX = np.full((parameters.NB_DEVICES,), parameters.CAPACITY_MAX, dtype=int)

        c_unit_next = np.asarray([np.min([c_unit_next[i], c_MAX[i]]) for i in range(len(c_unit_next))])

        total_data = np.sum(d_unit)
        total_energy = np.sum(e_unit)

        reward = 0
        state_changed = True
        self.accumulated_data += total_data

        latency = np.full((parameters.NB_DEVICES,), 0.0)

        # check en < cn and en/dn constrain
        for i in range(e_unit.shape[0]):
            if e_unit[i] != 0 and d_unit[i] != 0:
                cpu_require = np.full((parameters.NB_DEVICES,), 0)
                cpu_require[i] = (parameters.DELTA * e_unit[i]) / (parameters.TAU * parameters.NU * d_unit[i])
                cpu_require[i] = cpu_require[i] ** 0.5
                if cpu_require[i] <= parameters.MICRO * f_cpu[i] and e_unit[i] <= c_unit[i]:
                    latency[i] = parameters.NU * d_unit[i] / cpu_require[i]
                else:
                    latency = latency
                    reward -= 5
                    state_changed = False
            else:
                latency = latency

        if self.accumulated_data > parameters.MAX_ACCUMULATED_DATA:
            done = True
            self.episode_counter += 1
        else:
            done = False

        done = bool(done)
        latency_max = np.amax(latency)

        reward += 10 * (parameters.SCALE_FACTOR * total_data / parameters.DATA_THRESLOD \
                  - total_energy / parameters.ENERGY_THRESOLD - latency_max / parameters.LATENCY_MAX)
        # add an reward_base to make sure reward > 0 -> encourage maximize positive long term value,
        # rather than negative
        reward += parameters.REWARD_BASE
        self.episode_reward += reward

        if state_changed:
            next_state = np.array([f_cpu, c_unit_next]).flatten()
            self.state = next_state
        else:
            self.state = state

        logs = {
                'episode':  self.episode_counter,
                'episode_reward': self.episode_reward,
                'energy': 100. * total_energy / parameters.ENERGY_THRESOLD, # propotion of energy required
                'latency': latency_max,
                'step_total': self.step_total,
                'episode_steps': self.step_counter,
                'reward_mean': self.episode_reward / self.step_counter,
                'training_data': self.accumulated_data,
               },
        # export results to excel file
        if done:
            _to_excel(logs, self.episode_counter)

        if self.step_total == parameters.NB_STEPS:
            workbook.close()

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.nprandom.randint(low=0, high=parameters.CPUSHARE_MAX + 1, size=self.observation_space.shape) # not so sure
        self.accumulated_data = 0
        self.episode_reward = 0
        self.step_counter = 0
        return self.state

    def seed(self, seed=None):
        self.nprandom, seed = seeding.np_random(seed)
        return [seed]


