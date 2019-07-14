import numpy as np
import matplotlib.pyplot as plt
import math

sample_line = 1
T = 5000
decay = 40

c = [0.5, 0.95]
X = np.zeros((sample_line,T), dtype=int)
print(np.random.poisson(0.95, 100))

for i in range(sample_line):
    x = 1
    for t in range(1, T):
        if t % np.random.poisson(decay) != 0:
            # x = x + np.random.poisson(c[i])
            x = x + np.random.exponential(c[i])
            X[i][t] = x
        else:
            x= max(0, x - np.random.poisson(decay))
            X[i][t] = x

# plt.plot(np.arange(T), X[0][:], 'b')
# plt.plot(np.arange(T), X[1][:], 'r')
# plt.plot(np.arange(T), X[2][:], 'g')
# plt.show()

# Probability mass function of confirmation time (in blocks)
fee_rate = [0.25, 0.5, 0.95]
init_mem = [0, 1, 2, 4]
n_max = 10
prob_n = np.zeros((len(fee_rate), n_max), dtype=float)
# print(prob_n)
results = []

# Calculate probability of confirmation time
def get_prob(m, n, fee):
    # c = 1 - 0.25 * (fee + 1)
    c = 0.25
    y = np.max([(n - m) / c, 0]) #max((n - x) / c, 0)
    sigma = np.array([y ** k * np.exp(-y) / math.factorial(k) for k in range(n)]).sum()
    prob = 1 - sigma
    return prob

# Find delta_n = n - x0
for n in range(20):
    for k in range(n):
        # m = np.random.poisson(2)
        m = 2
        prob = get_prob(m, n, 0) # n should equal m + 6 or m + 7 with average traffic c = 0.5, n = m + 2 with c = 0.25
        if prob > 0.9:
            print(n, prob, m, k)
        else:
            print(n, prob, m)
