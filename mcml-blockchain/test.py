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

for i in range(len(fee_rate)):
    for n in range(n_max):
        y_n = max(0,(n - init_mem[1])/fee_rate[i])
        sigma_sum = np.array([y_n ** k / math.factorial(k) * np.exp(-y_n) for k in range(n)]).sum()
        prob_n[i][n] = 1 - sigma_sum
        # print(np.array([(y_n ** k / math.factorial(k)) * np.exp(-y_n)] for k in range(n)).sum())
        # prob_n[i][n] = 1.0 - np.array([(y_n ** k / math.factorial(k)) * np.exp(-y_n)] for k in range(n)).sum()

    # for j in range(n_max):
    #     prob_n[i][j] = 1 - sum(prob_n[i][:j])
# print(prob_n[0], prob_n[1], prob_n[2])
# plt.plot(np.arange(1, n_max), prob_n[0][1:], '-bo', label ='0.25')
# plt.plot(np.arange(1, n_max), prob_n[1][1:], '-r*', label ='0.5')
# plt.plot(np.arange(1, n_max), prob_n[2][1:], '-g^', label ='0.95')
# plt.legend()
# plt.show()

# Factorial test
# result = np.array([f(t) for t in range(1,m+1)]).sum()
# results = []
# for i in range(5):
#     value = np.array([math.factorial(k) for k in range(i+1)]).sum()
#     results.append(value)
# print(results)
# print(math.factorial(0))

# Array test
# arr = np.arange(9)
# print(arr)
# print(arr[3:6])
#
# energy = np.random.randint(low=1, high=4, size=3)
# data = np.random.randint(low=1, high=4, size=3)
# cpu_cycles = np.zeros(3)
# cpu_shares = np.random.randint(low=0, high=4, size=3)
# print(cpu_shares)
# for i in range(len(energy)):
#     cpu_cycles[i] =  np.sqrt(1 * energy[i]) / np.sqrt((10 ** -18) * data[i])
#     if cpu_cycles[i] > 0.6 * 10 ** 9 * cpu_shares[i]:
#         cpu_cycles[i] = 0.6 * 10 ** 9 * cpu_shares[i]
#
# print(cpu_cycles)

# Exponential distribution
# print(np.random.exponential(10, 100))
# exp_dis = np.random.exponential(10, 100)
# int_array = [np.int(k) for k in exp_dis]
# print(int_array)
# plt.plot(np.arange(0, 200), np.random.exponential(10, 200))
# plt.show()

# Calculate probability of confirmation time
def get_prob(x, n, c):
    y = np.max([(n - x) / c, 0]) #max((n - x) / c, 0)
    sigma = np.array([y ** k * np.exp(-y) / math.factorial(k) for k in range(n)]).sum()
    prob = 1 - sigma
    return prob

# Find delta_n = n - x0
x0 = 2
c0 = 0.25
probs = []
delta_n = []
for n in range(x0, 10):
    prob = get_prob(x0, n, c0)
    probs.append(prob)
    delta_n.append(np.max([n - x0, 0]))
print(probs)
print(np.max([5, 6]))
plt.plot(delta_n, probs, '-b*')
plt.show()

# # Find optima c
# x0 = 2
# n_min = x0 + 2
# probs = []
#
# for c in range(3):
#     c_fee = 0.25 * (c + 1)
#     prob = get_prob(x0, n_min, c_fee)
#     probs.append(prob)
#
# print(probs)