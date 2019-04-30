import numpy as np
from rl.core import Processor
from parameters import Parameters

parameters = Parameters()

def base10toN(num, base):
    """Change ``num'' to given base
    Upto base 36 is supported."""

    converted_string, modstring = "", ""
    currentnum = num
    if not 1 < base < 37:
        raise ValueError("base must be between 2 and 36")
    if not num:
        return '0'
    while currentnum:
        mod = currentnum % base
        currentnum = currentnum // base
        converted_string = chr(48 + mod + 7*(mod > 10)) + converted_string
    return converted_string

def _convert_action(action):
    """
    Convert action from decima to array (dn, cn)
    :param action: input action in decima

    :return: action in array
    """
    data_base = parameters.data_max + 1
    energy_base = parameters.energy_max + 1
    action_size = 2 * parameters.nb_devices
    data_array, energy_array = [], []
    action_clone = action

    # e.g. 3469 to 312031
    for i in range( 2 * parameters.nb_devices):
        if i < parameters.nb_devices:
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
    # test
    to_base_4_test = base10toN(action, 4)
    # print("action {}, processed_action {}, to_base_4 {}".format(action, processed_action, to_base_4_test))

    return processed_action

class MCMLProcessor(Processor):
    def __init__(self):
        self.metrics_names = []

    def process_action(self, action):
        # processed_action = np.random.randint(0, parameters.energy_max + 1, size=6)
        processed_action = _convert_action(action)

        return processed_action

    def metrics_names(self):
        return metrics_names

    def process_observation(self, observation):
        return observation

