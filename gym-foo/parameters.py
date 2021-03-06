import math
N = 3 # number of mobile devices
E_n = 3 # number of energy units that server requires each training iteration
D_n = 3 # maximum nunber of data units that mobile device n can use to train the model each iteration
nu = 10**10 # ν: number of CPU duty cycles required to train one data unit
tau = 10**(-28) # τ: effective switched capacitance
micro = 0.6 * 10 ** 9 # μ(GHz): number of CPU cycles corresponding to f_n CPU shares
delta = 1 # δ(Joule): energy unit

# Hieunq: newly defined variables
F_cpu = 3 # maximum number of CPU shares
C_capacity = 3 # capacity of energy storage
Dmax = N * D_n
Emax = N * E_n
scale_factor = 2
F_THRESOLD = 3
max_accumulated_data = 1500
reward_base = 5

class Parameters():
    def __init__(self):
        self.NB_DEVICES = N
        self.ENERGY_MAX = E_n
        self.DATA_MAX = D_n
        self.CPUSHARE_MAX = F_cpu
        self.CAPACITY_MAX = C_capacity
        self.SCALE_FACTOR = scale_factor
        self.CPU_REQUIRED_CONSTANT = math.sqrt(10**18) / (0.6 * (10**9))
        self.ENERGY_THRESOLD = Emax
        self.DATA_THRESLOD = Dmax
        self.MAX_ACCUMULATED_DATA = max_accumulated_data
        self.DELTA = delta
        self.NU = nu
        self.MICRO = micro
        self.TAU = tau
        latency_constant = (tau ** 0.5) * (nu ** 1.5) * (delta ** (-0.5))
        self.LATENCY_MAX = latency_constant * (D_n ** 1.5) * 1
        self.REWARD_BASE = reward_base
        self.LAMBDA = 1

        self.NB_STEPS = 1000000 # 1347 episode
        self.EPISODES_TRAINING = 600

        self.XLSX_PATH = 'results-13.xlsx'
