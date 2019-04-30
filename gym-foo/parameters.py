N = 3 # number of mobile devices
E_n = 3 # number of energy units that server requires each training iteration
D_n = 3 # maximum nunber of data units that mobile device n can use to train the model each iteration
nu = 10**10 # ν: number of CPU duty cycles required to train one data unit
tau = 10**(-28) # τ: effective switched capacitance
micro = 0.6 # μ(GHz): number of CPU cycles corresponding to f_n CPU shares
delta = 1 # δ(Joule): energy unit

# Hieunq: newly defined variables
F_cpu = 10 # maximum number of CPU shares
C_capacity = 10 # capacity of energy storage
# Dmax = N * D_n - 1 # offset = 1
# Emax = N * E_n - 1
Dmax = D_n - 1
Emax = E_n - 1
scale_factor = 10
F_THRESOLD = 3

class Parameters():
    def __init__(self):
        self.nb_devices = N
        self.energy_max = E_n
        self.data_max = D_n
        self.cpushare_max = F_cpu
        self.capacity_max = C_capacity
        self.scale_factor = scale_factor