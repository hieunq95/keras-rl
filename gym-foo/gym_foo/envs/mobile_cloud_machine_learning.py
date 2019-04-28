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
# Dmax = N * D_n - 1 # offset = 1
# Emax = N * E_n - 1
Dmax = D_n - 1
Emax = E_n - 1
scale_factor = 1000


class MCML(gym.Env):
    """
    Mobile cloud machine learning environment

    - State space: Sn = {(fn , cn ) ; fn ∈ {0, 1, . . . , Fn } , cn ∈ {0, 1, . . . , Cn } ,
    - Action space: A = {(d1 , e1 , . . . , dN , eN );dn ≤ Dn , en ≤ cn , fn ̄≤ μfn, n = (1, . . . , N )}
    - Transition: c_n' = c_n - e_n + A_n
    """
    def __init__(self):

        # high_action = np.array([D_n, E_n])
        # high_action = np.repeat(high_action, N)

        # For simplicity, assume F_n = C_n
        self.observation_space = spaces.Box(low=0, high=F_n, shape=(2*N,), dtype=int)
        # MultiDiscrete samples all actions at one time and return random value for each action !
        # self.action_space = spaces.MultiDiscrete(high_action) # A = {(d_1, e_1, ..., d_N, e_N)}
        self.action_space = spaces.Discrete(2 * N)

        self.seed()
        self.state = None
        self.reset()

    def step(self, action):
        # print("debug: action {}".format(action))
        # print("debug: action.sample() {}".format(self.action_space.sample()))
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # choose 2 values of d_n and e_n, make sure they meet the constrains
        e_n, d_n = np.random.randint(0, E_n), np.random.randint(0, D_n)
        # save the position of action
        e_n_offset = action % N

        state = self.state
        # accumulated_data, energy_consumption, training_latency = 0, 0, 0

        f_array, c_array = [], []
        radom_state = self.observation_space.sample() # random transition following uniform distribution

        for i in range(state.shape[0]): # 0 to 5
            if i < N:
                f_array.append(radom_state[i]) # shape (3, ), append first 3 elements
            else:
                c_array.append(state[i]) # shape (3, )

        A_n = 0
        c_array[e_n_offset] = c_array[e_n_offset] - e_n + A_n

        self.state = np.array([f_array, c_array], dtype=int).flatten() # state update

        done = d_n >= Dmax or e_n >= Emax or c_array[e_n_offset] < 0

        if not done:
            reward =  scale_factor * (d_n / Dmax - e_n / Emax)
        else:
            reward = 0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.nprandom.randint(low=0, high=F_n, size=self.observation_space.shape) # not so sure
        return self.state

    def seed(self, seed=None):
        self.nprandom, seed = seeding.np_random(seed)
        return [seed]


