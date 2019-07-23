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


INTERARRIVAL_RATE = 7 # good enough for convergence

TERMINATION_STEPS = 400

# TEST M/M/1 queue model
LAMBDA = 4
MIU = 6  # can serve 6 requests in 1 hour

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
	
180, 181, 182, 183:
...
DECAY_EPSILON_END = 2000 / 0.9 - 0.05

184, 185, 186, 187, 188:
    Test on M/M/1 queue, mempool state follows the geometric distribution
    NB_DEVICES = 1
    p = 1/6, 2/6, 3/6, 4/6, 5/6
    
189, 190, 191, 192, 193:
    NB_DEVICES = 2
    Add queue latency
    p = 1/6, 2/6, 3/6, 4/6, 5/6
    CONDLUDE: FAIL (didnt inclue queue latency in reward), all with LAMBDA = 5, didn't include queue latency

194, 195, 196, 197, 198
    NB_DEVICES = 2
    p = 1/6, 5/6, 2/6, 3/6, 4/6
    
199:
    ...
    policy = GreedyQPolicy()    

200:
    Random    
"""
