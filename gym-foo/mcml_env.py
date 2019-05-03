# Mobile cloud machine learning (MCML) environment
# Author: Hieunq

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from parameters import Parameters

# see parameters.py
parameters = Parameters()

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

        self.seed()
        self.reset()

    def step(self, action):
        # print("debug: action {}".format(action))
        # print("debug: action.sample() {}".format(self.action_space.sample()))
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
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
        else:
            done = False

        done = bool(done)
        latency_max = np.amax(latency)

        reward += 10 * (parameters.SCALE_FACTOR * total_data / parameters.DATA_THRESLOD \
                  - total_energy / parameters.ENERGY_THRESOLD - latency_max / parameters.LATENCY_MAX)
        # add an reward_base to make sure reward > 0 -> encourage maximize positive long term value,
        # rather than negative
        reward += parameters.REWARD_BASE

        if state_changed:
            next_state = np.array([f_cpu, c_unit_next]).flatten()
            self.state = next_state
        else:
            self.state = state

        info = {"reward": reward,
                "total_data": total_data,
                "total_energy": total_energy}
        # self.state = self.observation_space.sample()
        # print(action, e_unit, state, f_cpu, c_unit,c_unit_next, next_state) # [2 2 2 2 0 1]
        return np.array(self.state), reward, done, info

    def reset(self):
        self.state = self.nprandom.randint(low=0, high=parameters.CPUSHARE_MAX + 1, size=self.observation_space.shape) # not so sure
        self.accumulated_data = 0
        return self.state

    def seed(self, seed=None):
        self.nprandom, seed = seeding.np_random(seed)
        return [seed]


