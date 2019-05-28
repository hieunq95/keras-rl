import numpy as np
import matplotlib.pyplot as plt

sample_line = 1
T = 5000
decay = 100
x = 0
c = [0.95, 0.95]
X = np.zeros((sample_line,T), dtype=int)
print(np.random.poisson(0.95, 100))

for i in range(sample_line):
    x = 0
    for t in range(1, T):
        if t % np.random.poisson(100) != 0:
            x = x + np.random.poisson(c[i])
            X[i][t] = x
        else:
            x= max(0, x - np.random.poisson(decay))
            X[i][t] = x

plt.plot(np.arange(T), X[0][:], 'b')
# plt.plot(np.arange(T), X[1][:], 'r')
# plt.plot(np.arange(T), X[2][:], 'g')
plt.show()
