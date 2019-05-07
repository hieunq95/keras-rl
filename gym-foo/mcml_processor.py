import numpy as np
from rl.core import Processor
from parameters import Parameters

parameters = Parameters()

def _convert_action(action):
    """
    Convert action from decima to array (dn, cn)
    :param action: input action in decima

    :return: action in array
    """
    data_base = parameters.DATA_MAX + 1
    energy_base = parameters.ENERGY_MAX + 1
    action_size = 2 * parameters.NB_DEVICES
    data_array, energy_array = [], []
    action_clone = action

    # e.g. 3469 to 312031
    for i in range( 2 * parameters.NB_DEVICES):
        if i < parameters.NB_DEVICES:
            divisor = data_base ** (action_size - (i + 1))
            data_i = action_clone // divisor
            action_clone -= data_i * divisor
            data_array.append(data_i)
        else:
            divisor = energy_base ** (action_size - (i + 1))
            energy_i = action_clone // divisor
            action_clone -= energy_i * divisor
            energy_array.append(energy_i)

    processed_action = np.array([data_array, energy_array]).flatten()

    return processed_action

class MCMLProcessor(Processor):
    def __init__(self):
        self.metrics_names = []

    def process_action(self, action):
        # processed_action = np.random.randint(0, parameters.ENERGY_MAX + 1, size=6)
        processed_action = _convert_action(action)

        return processed_action

    def metrics_names(self):
        return metrics_names

    def process_observation(self, observation):
        return observation



