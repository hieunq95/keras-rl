# Mobile crowd machine learning block chain
# @author: Hieunq

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from config import NB_DEVICES, CPU_SHARES, CAPACITY_MAX, ENERGY_MAX, DATA_MAX, FEERATE_MAX, MEMPOOL_MAX

def calculate_latency():
    ...
def state_transition(state, action):
    """
    Perform state transition

    :param state: current state

    :param action: a taken action

    :return: next state
    """
    def check_constrains(state, action):
        return False
    return state, action
def get_reward():
    ...

class Environment(gym.Env):
    def __init__(self):
        action_array = np.array([DATA_MAX, ENERGY_MAX, FEERATE_MAX])
        action_array = np.repeat(action_array, NB_DEVICES)

        state_lower_bound = np.array([0, 0, 0]).repeat(NB_DEVICES)
        state_upper_bound = np.array([CPU_SHARES, CAPACITY_MAX, MEMPOOL_MAX]).repeat(NB_DEVICES)
        """
        state_space : {f1,f2,...,F, c1, c2, ..., C, m1, m2, ..., M}
        action_space: {d1, d2, ..., D, e1, e2, ..., E, r1, r2, ..., R}
        """
        self.observation_space = spaces.Box(low=state_lower_bound, high=state_upper_bound, dtype=int)
        self.action_space = spaces.MultiDiscrete(action_array)

    def step(self, action):
        state = self.observation_space
        action = self.action_space
        next_state = state_transition(state, action)
        reward = get_reward()
        done = False

        self.state = next_state
        return np.array(self.state), reward, done, {}

    def reset(self):
        cpu_shares_init = self.nprandom.randint(CPU_SHARES, size=NB_DEVICES)
        capacity_init = self.nprandom.randint(CAPACITY_MAX, size=NB_DEVICES)
        mempool_init = self.nprandom.randint(MEMPOOL_MAX, size=NB_DEVICES)
        state = np.array([cpu_shares_init, capacity_init, mempool_init]).flatten()
        self.state = state
        
        return self.state

    def seed(self, seed=None):
        self.nprandom, seed = seeding.np_random(seed)
        return [seed]

class MyPolicy():
    ...