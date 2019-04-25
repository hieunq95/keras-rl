# Mobile cloud machine learning (MCML) environment
# Author: Hieunq

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

N = 3 # number of mobile devices
E_n = 3 # number of energy units that server requires each training iteration
D_n = 3 # maximum nunber of data units that mobile device n can use to train the model each iteration
nu = 10**10 # ν: number of CPU duty cycles required to train one data unit
tau = 10**(-28) # τ: effective switched capacitance
micro = 0.6 # μ(GHz): number of CPU cycles corresponding to f_n CPU shares
delta = 1 # δ(Joule): energy unit

# Hieunq: newly defined variables
F_n = 10 # maximum number of CPU shares
C_n = 10 # capacity of energy storage

class MCML(gym.Env):
    """
    Mobile cloud machine learning environment
    """
    def __init__(self):

        high_action = np.array([D_n, E_n])
        high_action = np.repeat(high_action, N)

        high_space = np.array([[F_n],
                               [C_n]]) # shape (2,1)
        high_space = np.reshape(high_space, [1, 2]) # shape (1,2)
        high_space = np.repeat(high_space, N, axis=0) # shape (3,2)

        self.observation_space = spaces.Box(np.zeros(high_space.shape),
                                            high_space, dtype=np.int) # Sn = {(fn , cn )}
        self.action_space = spaces.MultiDiscrete(high_action) # A = {(d_1, e_1, ..., d_N, e_N)}

        self.seed()
        self.reset()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        f_n, c_n = state
        # state transition
        # Sn = {(fn , cn ) ; fn ∈ {0, 1, . . . , Fn } , cn ∈ {0, 1, . . . , Cn } ,
        # A = {(d1 , e1 , . . . , dN , eN );dn ≤ Dn , en ≤ cn , fn ̄≤ μfn, n = (1, . . . , N )}
        f_n = np.random.randint(F_n + 1)

        # let's assume action is e_n
        e_n = action
        # c_n = np.amin(c_n - e_n + A_n, C_n)
        self.state = (f_n, c_n)
        next_state = state
        done = False
        return state, reward, done, {}


    def get_observation(self):
        ...

    def reset(self):
        ...

    def seed(self, seed=None):
        self.nprandom, seed = seeding.np_random(seed)
        return [seed]


