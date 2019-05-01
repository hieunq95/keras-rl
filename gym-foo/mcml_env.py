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
        c_unit_next = np.subtract(c_unit, e_unit)

        total_data = np.sum(d_unit)
        total_energy = np.sum(e_unit)

        reward = 0
        state_changed = True
        self.accumulated_data += total_data
        # check en < cn and en/dn constrain
        for i in range(e_unit.shape[0]):
            if d_unit[i] == 0:
                if e_unit[i] <= c_unit[i]:
                    # done = False
                    reward = reward
                else:
                    # done = True
                    state_changed = False
                    reward -= 5
            else:
                if e_unit[i] <= c_unit[i] and \
                    e_unit[i] / d_unit[i] <= ((f_cpu[i] / parameters.CPU_REQUIRED_CONSTANT) ** 2):
                    # done = False
                    reward = reward
                else:
                    # done = True
                    state_changed = False
                    reward -= 5

        if self.accumulated_data > parameters.MAX_ACCUMULATED_DATA:
            done = True
        else:
            done = False

        done = bool(done)
        reward += 10 * (parameters.SCALE_FACTOR * total_data / parameters.DATA_THRESLOD \
                  - total_energy / parameters.ENERGY_THRESOLD)

        if state_changed:
            next_state = np.array([f_cpu, c_unit_next]).flatten()
            self.state = next_state
        else:
            self.state = state

        # self.state = self.observation_space.sample()
        # print(action, e_unit, state, f_cpu, c_unit,c_unit_next, next_state) # [2 2 2 2 0 1]
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.nprandom.randint(low=0, high=parameters.CPUSHARE_MAX, size=self.observation_space.shape) # not so sure
        self.accumulated_data = 0
        return self.state

    def seed(self, seed=None):
        self.nprandom, seed = seeding.np_random(seed)
        return [seed]


