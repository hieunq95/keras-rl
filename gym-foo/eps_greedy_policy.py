# Implementation of Epsilon Greedy Policy
# Author: Hieunq

from rl.policy import Policy
import numpy as np

class MyEpsGreedy(Policy):
    """
    Implement the epsilon greedy policy
    """
    def __init__(self, eps_max, eps_min, nb_steps):
        super(MyEpsGreedy, self).__init__()
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.step_counter = 0
        self.eps = self.eps_max

        self.steps_per_episode = 200 # need to correct according to number of steps each episode
        # decay each episode
        self.eps_decay = (eps_max - eps_min) / (nb_steps / self.steps_per_episode)

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        self.step_counter += 1
        nb_actions = q_values.shape[0]

        if (self.eps >= self.eps_min + self.eps_decay)\
                and (self.step_counter % self.steps_per_episode == 0):
            self.eps = self.eps - self.eps_decay

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)

        # if self.step_counter % self.steps_per_episode == 0:
        #     print("epsilon {}".format(self.eps))

        return action





