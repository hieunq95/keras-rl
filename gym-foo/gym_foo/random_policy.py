# Implement random policy for MCML agent
# Author: Hieunq

import numpy as np
import xlsxwriter
from rl.policy import Policy
from mcml_env import MCML

class RandomPolicy(Policy):
    def __init__(self, environment, writer):
        super(MyEpsGreedy, self).__init__()
        self.environment = environment
        self.writer = writer

    def select_action(self, q_values):
        """
        Seclect an random action from the environment

        :param #

        :return: A random action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        action = np.random.randint(0, nb_actions)
        
        return action



