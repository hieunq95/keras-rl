"""
Config paremeters for Machine crowd machine learning block chain
"""
NB_DEVICES = 2

CPU_SHARES = 4
CAPACITY_MAX = 4

ENERGY_MAX = 4
DATA_MAX = 4
FEERATE_MAX = 3

MEMPOOL_MAX = 10
MEMPOOL_INIT = 2
MEMPOOL_SLOPE = 0.75
#TODO: derive a function for feerate and the slope of mempool


INTERARRIVAL_RATE = 0.25

TERMINATION_STEPS = 400