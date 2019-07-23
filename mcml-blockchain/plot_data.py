import pandas
import matplotlib.pyplot as plt
import numpy as np


TEST_TH = 181
MEMPOOL_TH = 71
EWM_WINDOW = 50

IS_GREEDY = 1

if IS_GREEDY:
    data_path = './build/results-greedy-{}.xlsx'.format(TEST_TH)
else:
    data_path = './build/results-{}.xlsx'.format(TEST_TH)
mempool_path = './build/results-{}.xlsx'.format(MEMPOOL_TH)
df = pandas.read_excel(data_path)

#get the values for a given column
energy = df['Energy'].values
latency = df['Latency'].values
data = df['Training_data_mean'].values
reward = df['Total_reward'].values
payment = df['Payment'].values

energy_ewm = df['Energy'].ewm(span=EWM_WINDOW, adjust=True).mean()
latency_ewm = df['Latency'].ewm(span=EWM_WINDOW, adjust=True).mean()
payment_ewm = df['Payment'].ewm(span=EWM_WINDOW, adjust=True).mean()
reward_ewm = df['Total_reward'].ewm(span=EWM_WINDOW, adjust=True).mean()

energy_plot = []
energy_ewm_plot = []
latency_plot = []
latency_ewm_plot = []
data_plot = []
reward_plot = []
reward_ewm_plot = []
payment_plot = []
payment_ewm_plot = []


for i in range(len(energy)):
    # if i % 2 == 0:
    energy_plot.append(energy[i])
    energy_ewm_plot.append(energy_ewm[i])

for i in range(len(latency)):
    # if i % 2 == 0:
    latency_plot.append(latency[i])
    latency_ewm_plot.append(latency_ewm[i])

for i in range(len(data)):
    # if i % 2 == 0:
    data_plot.append(data[i])

for i in range(len(reward)):
    # if i % 2 == 0:
    reward_plot.append(reward[i])
    reward_ewm_plot.append(reward_ewm[i])

for i in range(len(payment)):
    # if i % 2 == 0:
    payment_plot.append(payment[i])
    payment_ewm_plot.append(payment_ewm[i])


episodes = np.arange(0, len(energy_plot))

# for k in range(len(episodes)):
#     episodes[k] = episodes[k] * 2

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

plt.plot(episodes, reward_plot, label='Total reward')
plt.plot(episodes, reward_ewm_plot, label='Total reward - EWM')
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


"""
Plot various payment 
From 172 to 175
"""
EWM_PAYMENT_WINDOW = 100

result_1 = './build/results-{}.xlsx'.format(172)
result_2 = './build/results-{}.xlsx'.format(173)
result_3 = './build/results-{}.xlsx'.format(174)
result_4 = './build/results-{}.xlsx'.format(175)

df_1 = pandas.read_excel(result_1)
df_2 = pandas.read_excel(result_2)
df_3 = pandas.read_excel(result_3)
df_4 = pandas.read_excel(result_4)

payment_1 = df_1['Payment'].values
payment_2 = df_2['Payment'].values
payment_3 = df_3['Payment'].values
payment_4 = df_4['Payment'].values

payment_1_ewm = df_1['Payment'].ewm(span=EWM_PAYMENT_WINDOW, adjust=True).mean()
payment_2_ewm = df_2['Payment'].ewm(span=EWM_PAYMENT_WINDOW, adjust=True).mean()
payment_3_ewm = df_3['Payment'].ewm(span=EWM_PAYMENT_WINDOW, adjust=True).mean()
payment_4_ewm = df_4['Payment'].ewm(span=EWM_PAYMENT_WINDOW, adjust=True).mean()

payment_1_plot, payment_1_ewm_plot = [], []
payment_2_plot, payment_2_ewm_plot = [], []
payment_3_plot, payment_3_ewm_plot = [], []
payment_4_plot, payment_4_ewm_plot = [], []

sampled_episodes = 3000
episode_axis = np.arange(0, sampled_episodes)

for i in range(sampled_episodes):
    payment_1_plot.append(payment_1[i])
    payment_2_plot.append(payment_2[i])
    payment_3_plot.append(payment_3[i])
    payment_4_plot.append(payment_4[i])

    payment_1_ewm_plot.append(payment_1_ewm[i])
    payment_2_ewm_plot.append(payment_2_ewm[i])
    payment_3_ewm_plot.append(payment_3_ewm[i])
    payment_4_ewm_plot.append(payment_4_ewm[i])


payments = [payment_1_plot, payment_2_plot, payment_3_plot, payment_4_plot]
payments_ewm = [payment_1_ewm_plot, payment_2_ewm_plot, payment_3_ewm_plot, payment_4_ewm_plot]

for k in range(len(payments)):
    # plt.plot(episode_axis, payments[k], label='Payment-{}'.format(k))
    plt.plot(episode_axis, payments_ewm[k], label='Mean payment - {}'.format(k))

plt.legend()
plt.ylabel('Payment')
plt.xlabel('Episode')
# plt.savefig('./results/{}-payment.png'.format(TEST_TH))
plt.show()



















