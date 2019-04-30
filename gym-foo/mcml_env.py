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

        high_action = np.array([parameters.data_max+1, parameters.energy_max+1])
        high_action = np.repeat(high_action, parameters.nb_devices)

        # For simplicity, assume F_n = C_n
        self.observation_space = spaces.Box(low=0, high=parameters.cpushare_max,
                                            shape=(2 * parameters.nb_devices,), dtype=int)
        # MultiDiscrete samples all actions at one time and return random value for each action !
        self.action_space = spaces.MultiDiscrete(high_action) # A = {(d_1, e_1, ..., d_N, e_N)}
        # self.action_space = spaces.Discrete(2)
        # self.action_space = spaces.Box(low=0, high=E_n, shape=(2,), dtype=int)

        self.seed()
        self.reset()

    def step(self, action):
        # print("debug: action {}".format(action))
        # print("debug: action.sample() {}".format(self.action_space.sample()))
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        state = self.state

        f_cpu = np.random.randint(0, parameters.cpushare_max+1, size=3) # fn get an random action
        c_unit = state.reshape([2, parameters.nb_devices])[1].copy() # copy cn from current state

        # add fn penalty later
        d_unit = np.asarray(action).reshape([2, parameters.nb_devices])[0].copy()
        e_unit = np.asarray(action).reshape([2, parameters.nb_devices])[1].copy() # copy last 3 elements

        #state transition
        c_unit_next = np.subtract(c_unit, e_unit)

        total_data = np.sum(d_unit)
        total_energy = np.sum(e_unit)

        if np.amin(c_unit_next) < 0:
            reward = - 10
        else:
            reward = parameters.scale_factor * total_data - total_energy

        next_state = np.array([f_cpu, c_unit_next]).flatten()

        done = False
        # self.state = self.observation_space.sample()
        self.state = next_state
        # print(action, e_unit, state, f_cpu, c_unit,c_unit_next, next_state) # [2 2 2 2 0 1]

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.nprandom.randint(low=0, high=parameters.cpushare_max, size=self.observation_space.shape) # not so sure
        return self.state

    def seed(self, seed=None):
        self.nprandom, seed = seeding.np_random(seed)
        return [seed]


