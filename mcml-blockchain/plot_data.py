import pandas
import matplotlib.pyplot as plt
import numpy as np

TEST_TH = 12
MEMPOOL_TH = 71

data_path = './build/results-{}.xlsx'.format(TEST_TH)
mempool_path = './build/results-{}.xlsx'.format(MEMPOOL_TH)
df = pandas.read_excel(data_path)

#get the values for a given column
energy = df['Energy'].values
latency = df['Latency'].values
data = df['Training_data_mean'].values
reward = df['Total_reward'].values
payment = df['Payment'].values

energy_ewm = df['Energy'].ewm(span=100, adjust=True).mean()
latency_ewm = df['Latency'].ewm(span=100, adjust=True).mean()
payment_ewm = df['Payment'].ewm(span=100, adjust=True).mean()

energy_plot = []
energy_ewm_plot = []
latency_plot = []
latency_ewm_plot = []
data_plot = []
reward_plot = []
payment_plot = []
payment_ewm_plot = []


for i in range(len(energy)):
    if i % 2 == 0:
        energy_plot.append(energy[i])
        energy_ewm_plot.append(energy_ewm[i])

for i in range(len(latency)):
    if i % 2 == 0:
        latency_plot.append(latency[i])
        latency_ewm_plot.append(latency_ewm[i])

for i in range(len(data)):
    if i % 2 == 0:
        data_plot.append(data[i])

for i in range(len(reward)):
    if i % 2 == 0:
        reward_plot.append(reward[i])

for i in range(len(payment)):
    if i % 2 == 0:
        payment_plot.append(payment[i])
        payment_ewm_plot.append(payment_ewm[i])

episodes = np.arange(0, len(energy_plot))

for k in range(len(episodes)):
    episodes[k] = episodes[k] * 2

plt.plot(episodes, energy_plot, label='Energey')
plt.plot(episodes, energy_ewm_plot, label='Energy_EWM')
plt.legend()
plt.ylabel('Energy consumption (units)')
plt.xlabel('Episode')
plt.savefig('./results/{}-energy.png'.format(TEST_TH))
plt.show()

plt.plot(episodes, latency_plot, label='Latency')
plt.plot(episodes, latency_ewm_plot, label='Latency_EWM')
plt.legend()
plt.ylabel('Training latency (units)')
plt.xlabel('Episode')
plt.savefig('./results/{}-latency.png'.format(TEST_TH))
plt.show()

plt.plot(episodes, data_plot)
plt.ylabel('Training data (units)')
plt.xlabel('Episode')
plt.savefig('./results/{}-data.png'.format(TEST_TH))
plt.show()

plt.plot(episodes, reward_plot)
plt.ylabel('Total reward')
plt.xlabel('Episode')
plt.savefig('./results/{}-reward.png'.format(TEST_TH))
plt.show()

plt.plot(episodes, payment_plot, label='Payment')
plt.plot(episodes, payment_ewm_plot, label='Payment_EWM')
plt.legend()
plt.ylabel('Payment')
plt.xlabel('Episode')
plt.savefig('./results/{}-payment.png'.format(TEST_TH))
plt.show()

