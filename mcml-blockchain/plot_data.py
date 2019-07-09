import pandas
import matplotlib.pyplot as plt
import numpy as np

TEST_TH = 101
MEMPOOL_TH = 71

data_path = './build/results-{}.xlsx'.format(TEST_TH)
mempool_path = './build/results-{}.xlsx'.format(MEMPOOL_TH)
df = pandas.read_excel(data_path)

#print the column names
print(df.columns[4])

#get the values for a given column
energy = df['Energy'].values
latency = df['Latency'].values
data = df['Training_data_mean'].values

energy_plot = []
latency_plot = []
data_plot = []

print(energy)


for i in range(len(energy)):
    if i % 2 == 0:
        energy_plot.append(energy[i])

for i in range(len(latency)):
    if i % 2 == 0:
        latency_plot.append(latency[i])

for i in range(len(data)):
    if i % 2 == 0:
        data_plot.append(data[i])

episodes = np.arange(0, len(energy_plot))

for k in range(len(episodes)):
    episodes[k] = episodes[k] * 2

plt.plot(episodes, energy_plot)
plt.ylabel('Energy consumption (units)')
plt.xlabel('Episode')
plt.savefig('./results/energy-{}.png'.format(TEST_TH))
plt.show()

plt.plot(episodes, latency_plot)
plt.ylabel('Training latency (units)')
plt.xlabel('Episode')
plt.savefig('./results/latency-{}.png'.format(TEST_TH))
plt.show()

plt.plot(episodes, data_plot)
plt.ylabel('Training data (units)')
plt.xlabel('Episode')
plt.savefig('./results/data-{}.png'.format(TEST_TH))
plt.show()

