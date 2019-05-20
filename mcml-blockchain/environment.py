# Mobile crowd machine learning block chain
# @author: Hieunq

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from config import NB_DEVICES, CPU_SHARES, CAPACITY_MAX, ENERGY_MAX, DATA_MAX

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
        action_array = np.array([DATA_MAX, ENERGY_MAX])
        action_array = np.repeat(action_array, NB_DEVICES)

        self.observation_space = spaces.Box(low=0, high=CPU_SHARES, shape=(2 * NB_DEVICES,), dtype=int)
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
        self.state = self.nprandom.randint(low=0, high=CPU_SHARES, size=self.observation_space.shape)
        return self.state

    def seed(self, seed=None):
        self.nprandom, seed = seeding.np_random(seed)
        return [seed]

class MyPolicy():
    ...