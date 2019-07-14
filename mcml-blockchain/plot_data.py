import pandas
import matplotlib.pyplot as plt
import numpy as np

<<<<<<< HEAD
TEST_TH = 101
=======
TEST_TH = 163
>>>>>>> 68b9f5c9c9ace643b45a52c02d66dde70d046084
MEMPOOL_TH = 71

data_path = './build/results-{}.xlsx'.format(TEST_TH)
mempool_path = './build/results-{}.xlsx'.format(MEMPOOL_TH)
df = pandas.read_excel(data_path)

<<<<<<< HEAD
#print the column names
print(df.columns[4])

=======
>>>>>>> 68b9f5c9c9ace643b45a52c02d66dde70d046084
#get the values for a given column
energy = df['Energy'].values
latency = df['Latency'].values
data = df['Training_data_mean'].values
<<<<<<< HEAD

energy_plot = []
latency_plot = []
data_plot = []

print(energy)
=======
reward = df['Total_reward'].values
payment = df['Payment'].values

energy_ewm = df['Energy'].ewm(span=100, adjust=True).mean()
latency_ewm = df['Latency'].ewm(span=100, adjust=True).mean()

energy_plot = []
energy_ewm_plot = []
latency_plot = []
latency_ewm_plot = []
data_plot = []
reward_plot = []
payment_plot = []
>>>>>>> 68b9f5c9c9ace643b45a52c02d66dde70d046084


for i in range(len(energy)):
    if i % 2 == 0:
        energy_plot.append(energy[i])
<<<<<<< HEAD
=======
        energy_ewm_plot.append(energy_ewm[i])
>>>>>>> 68b9f5c9c9ace643b45a52c02d66dde70d046084

for i in range(len(latency)):
    if i % 2 == 0:
        latency_plot.append(latency[i])
<<<<<<< HEAD
=======
        latency_ewm_plot.append(latency_ewm[i])
>>>>>>> 68b9f5c9c9ace643b45a52c02d66dde70d046084

for i in range(len(data)):
    if i % 2 == 0:
        data_plot.append(data[i])

<<<<<<< HEAD
=======
for i in range(len(reward)):
    if i % 2 == 0:
        reward_plot.append(reward[i])

for i in range(len(payment)):
    if i % 2 == 0:
        payment_plot.append(payment[i])

>>>>>>> 68b9f5c9c9ace643b45a52c02d66dde70d046084
episodes = np.arange(0, len(energy_plot))

for k in range(len(episodes)):
    episodes[k] = episodes[k] * 2

<<<<<<< HEAD
plt.plot(episodes, energy_plot)
plt.ylabel('Energy consumption (units)')
plt.xlabel('Episode')
plt.savefig('./results/energy-{}.png'.format(TEST_TH))
plt.show()

plt.plot(episodes, latency_plot)
plt.ylabel('Training latency (units)')
plt.xlabel('Episode')
plt.savefig('./results/latency-{}.png'.format(TEST_TH))
=======
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
>>>>>>> 68b9f5c9c9ace643b45a52c02d66dde70d046084
plt.show()

plt.plot(episodes, data_plot)
plt.ylabel('Training data (units)')
plt.xlabel('Episode')
<<<<<<< HEAD
plt.savefig('./results/data-{}.png'.format(TEST_TH))
=======
plt.savefig('./results/{}-data.png'.format(TEST_TH))
plt.show()

plt.plot(episodes, reward_plot)
plt.ylabel('Total reward')
plt.xlabel('Episode')
plt.savefig('./results/{}-reward.png'.format(TEST_TH))
plt.show()

plt.plot(episodes, payment_plot)
plt.ylabel('Payment')
plt.xlabel('Episode')
plt.savefig('./results/{}-payment.png'.format(TEST_TH))
>>>>>>> 68b9f5c9c9ace643b45a52c02d66dde70d046084
plt.show()

