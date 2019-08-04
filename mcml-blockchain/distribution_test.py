# @author: Hieu Nguyen
import numpy as np
import matplotlib.pyplot as plt

LAMBDA = 3
MU = [2, 3, 4, 5]
prob = LAMBDA / MU[2]

# Exponential distribution
exp = np.random.exponential(1 / (MU[3] - LAMBDA), size=10000)
print(exp)
exp_array = np.zeros(6)
for i in range(6):
    exp_array[i] = (exp <= (i + 1)).sum() / 10000

plt.plot(np.arange(0, 6), exp_array, 'r-o')
plt.show()

# Geometric distribution
geo = np.random.geometric(1 - prob, size=10000)
print(max(geo))
geo_array = np.zeros(10)

for i in range(10):
    geo_array[i] = (geo == i+1).sum() / 10000

# print(geo_array)
# plt.plot(np.arange(1, 11), geo_array, 'g-o')
# plt.ylim((0, 1.0))
#
# plt.show()
