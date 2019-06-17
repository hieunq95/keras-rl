"""
Config paremeters for Machine crowd machine learning block chain
"""
NB_DEVICES = 3

CPU_SHARES = 2
CAPACITY_MAX = 2

ENERGY_MAX = 3
DATA_MAX = 3

MEMPOOL_MAX = 10
MEMPOOL_INIT = 2
MEMPOOL_SLOPE = 0.75
#TODO: derive a function for feerate and the slope of mempool
FEERATE_MAX = 3

INTERARRIVAL_RATE = 0.25