"""
Config paremeters for Machine crowd machine learning block chain
"""
NB_DEVICES = 2

CPU_SHARES = 4
CAPACITY_MAX = 4

ENERGY_MAX = 4
DATA_MAX = 4
FEERATE_MAX = 4

MEMPOOL_MAX = 10
MEMPOOL_INIT = 1
MEMPOOL_SLOPE = 0.75
#TODO: derive a function for feerate and the slope of mempool


INTERARRIVAL_RATE = 5  # good enough for convergence

TERMINATION_STEPS = 400

"""
Results note:
170:
    NB_DEVICES = 2                              
    MEMPOOL_MAX = 5
    MEMPOOL_INIT = 1
    INTERARRIVAL_RATE = 3
    DECAY_EPSILON_END = 1000 / 0.9 - 0.1
    
172, 173, 174, 175: quick test with various INTERARRIVAL_RATE
    NB_DEVICES = 1
    MEMPOOL_MAX = 10
    MEMPOOL_INIT = 1
    INTERARRIVAL_RATE = 1, 3, 5, 7
    DECAY_EPSILON_END = 1000 / 0.9 - 0.05
    
176, 177, 178, 179
    NB_DEVICES = 2
    ...
    INTERARRIVAL_RATE = 1, 3, 5, 7
    
200:
    Random    
"""