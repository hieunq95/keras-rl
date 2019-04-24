# Implementation of Categorical distribution (Bernoulli distribution)
# https://github.com/openai/gym/blob/522c2c532293399920743265d9bc761ed18eadb3/gym/envs/toy_text/discrete.py

import numpy as np

from gym import Env, spaces
from gym.utils import seeding

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution

    Each row specifies class probabilities

    :param prob_n: (probability, nextstate, reward, done)
    :param np_random: a radom state
    :return:
    """
    prob_n = np.asarray(prob_n) # to array
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

class DescreteEnv(Env):
    """
    Has the following memebers

    - nS: number of states

    - nA: number of actions

    - P: transitions (*)

    - ids: initial state distribution (**)

    (*) dictionary dict of dicts of lists, where
     P[s][a] == [(probability, nextstate, reward, done),...]

    (**) list of array of length nS
    """

    def __init__(self, nS, nA, P, isd):
        self.nS = nS
        self.nA = nA
        self.P = P
        self.isd = isd
        self.lastaction = None

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.reset()

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        return self.s

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (s, r, d, {"prob" : p})


