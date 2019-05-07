# Implementation of Epsilon Greedy Policy
# Author: Hieunq

from rl.policy import Policy
import numpy as np
import xlsxwriter
from mcml_env import MCML

class MyEpsGreedy(Policy):
    """
    Implement the epsilon greedy policy
    """
    def __init__(self, environment, eps_max, eps_min, eps_training, writer):
        super(MyEpsGreedy, self).__init__()
        self.writer = writer
        self.environment = environment
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps = self.eps_max
        # decay each episode
        self.eps_decay = (eps_max - eps_min) / eps_training
        self.episode_counter = 0

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if self.environment.get_env_config():
            self.episode_counter += 1
            if self.eps >= self.eps_min + self.eps_decay:
                self.eps = self.eps - self.eps_decay
            else:
                self.eps = 0

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)

        if self.environment.get_env_config():
            # print("epsilon {}".format(self.eps))
            self.writer.epsilon_write(self.eps, self.episode_counter)

        return action





