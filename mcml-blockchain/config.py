"""
Config paremeters for Machine crowd machine learning block chain
"""
NB_DEVICES = 2

CPU_SHARES = 4
CAPACITY_MAX = 4

ENERGY_MAX = 4
DATA_MAX = 4
FEERATE_MAX = 4
MINING_RATE = 4

MEMPOOL_MAX = 10
MEMPOOL_INIT = 1
MEMPOOL_SLOPE = 0.75

# constant parameters for uplink, downlink latency
SNR = 10  # 10dB
W = 300000  # 300KHz
d_fr = 10  # federated data = 10Kbits
d_train = 10  # local updated data
d_blk = 100
L_wait = 1  # 100ms
BLK_TIME_SCALE = 1  # 1s

# constant parameters for blockchain network
LAMBDA = 3
MIU = 5  # can serve 6 requests in 1 hour

INTERARRIVAL_RATE = 7  # good enough for convergence

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

207:
    Random, Greedy,
    LAMBDA = 4
208, 209, 210, 211:
    Random, Greedy,
    LAMBDA = 1, 2, 3, 5
    
300:
    Test: increase action and state space up to 5
    NB_DEVICES
    policy = GreedyQPolicy() 
301:
    policy = MyEpsGreedy(env, 0.9, 0.05, DECAY_EPSILON_END, writer)
302, 303, 304:
    LAMBDA = 3, 1, 2
305, 306, 307,:
    LAMBDA = 1,2,3
    alpha: D, L, E, I = 5, 3, 1, 1
308:
    LAMBDA = 1
    alpha: D, L, E, I = 0, 3, 0, 0
309:
    LAMBDA = 1
    alpha: D, L, E, I = 5, 3, 2, 1
    mining_rate = LAMBDA + 1 + action[-1]
    FAILED: wrong mining_rate formula
310:
    alpha: D, L, E, I = 5, 3, 1, 2
    FAILED: reset(): m ~ Poisson
311:
    NB_DEVICE = 1
    reset(): m ~ Geometric
312, 313:
    NB_DEVICE = 2
    LAMBDA = 1, 3
314, 315:
    BLK_TIME_SCALE = 50
    L_wait = 10
    LAMBDA = 1, 3
    
316, 317:
    I ~ blk_price * G(m)
212, 213:
    Greedy, Random test
    LAMBDA = 3
318:
    blk_price = 2, training_price = 0.2
    L_wait = 1
    BLK_TIME_SCALE = 1
    LAMBDA = 3
214, 215:
    Greedy, Random,
    ...
319:
    blk_price = 1, training_price = 1
    LAMBDA = 3
    mining_rate = MIU + action[-1]
216:
    Greedy, Random
320:
   blk_price = 0.2, training_price = 1
321:
   blk_price = 0.2, training_price = 2
322:
    blk_price = 0.8, training_price = 0.2
    I ~ 1 / log(1+m)
    NB_DEVICES = 1
323: 
    NB_DEVICES = 2
324:
    NB_DEVICES = 1
    blk_price = 1, training_price = 0
325, 326, 327, 328:
    NB_DEVICES = 2
    blk_price = 0.8, training_price = 0.2
    data_qualities = [1, 1]
    LAMBDA = 1, 2, 3, 4
329, 330, 331:
    NB_DEVICES = 2
    data_qualities = [2, 1] [3, 1] [4, 1]
    LAMBDA = 3
217:
    NB_DEVICES = 2
    LAMBDA = 1
    data_qualities = [1, 1]
    greedy, random
332:
    NB_DEVICES = 2
    data_qualities = [1, 1]
    LAMBDA = 3
    I = psi1 * d_i + psi2 * m
"""
