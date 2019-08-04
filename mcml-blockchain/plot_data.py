import pandas
import matplotlib.pyplot as plt
import numpy as np

"""
Results:
results-random-201.xlsx
results-greedy-201.xlsx
results-195.xlsx
"""

RANDOM_TEST = './build/results-random-207.xlsx'
GREEDY_TEST = './build/results-greedy-208.xlsx'
DRL_TEST = './build/results-195.xlsx'

EWM_WINDOW = 20
TEST_TH = 195
RANGE = 4000

# MEMPOOL_TH = 71
#
# IS_GREEDY = 1
#
# if IS_GREEDY == 0:
#     data_path = './build/results-random-{}.xlsx'.format(TEST_TH)
# elif IS_GREEDY == 1:
#     data_path = './build/results-greedy-{}.xlsx'.format(TEST_TH)
# else:
#     data_path = './build/results-{}.xlsx'.format(TEST_TH)
# mempool_path = './build/results-{}.xlsx'.format(MEMPOOL_TH)

df = pandas.read_excel(DRL_TEST)
df_random = pandas.read_excel(RANDOM_TEST)
df_greedy = pandas.read_excel(GREEDY_TEST)

#get the values for a given column
energy = df['Energy'].values
latency = df['Latency'].values
data = df['Training_data_mean'].values
reward = df['Total_reward'].values
payment = df['Payment'].values

energy_greedy = df_greedy['Energy'].ewm(EWM_WINDOW, adjust=False).mean()
latency_greedy = df_greedy['Latency'].ewm(EWM_WINDOW, adjust=False).mean()
data_greedy = df_greedy['Training_data_mean'].ewm(EWM_WINDOW, adjust=False).mean()
reward_greedy = df_greedy['Total_reward'].ewm(EWM_WINDOW, adjust=False).mean()
payment_greedy = df_greedy['Payment'].ewm(EWM_WINDOW, adjust=False).mean()

energy_random = df_random['Energy'].ewm(EWM_WINDOW, adjust=False).mean()
latency_random = df_random['Latency'].ewm(EWM_WINDOW, adjust=False).mean()
data_random = df_random['Training_data_mean'].ewm(EWM_WINDOW, adjust=False).mean()
reward_random = df_random['Total_reward'].ewm(EWM_WINDOW, adjust=False).mean()
payment_random = df_random['Payment'].ewm(EWM_WINDOW, adjust=False).mean()

energy_ewm = df['Energy'].ewm(span=EWM_WINDOW, adjust=False).mean()
latency_ewm = df['Latency'].ewm(span=EWM_WINDOW, adjust=False).mean()
payment_ewm = df['Payment'].ewm(span=EWM_WINDOW, adjust=False).mean()
data_ewm = df['Training_data_mean'].ewm(span=EWM_WINDOW, adjust=False).mean()
reward_ewm = df['Total_reward'].ewm(span=EWM_WINDOW, adjust=False).mean()

energy_plot = []
energy_ewm_plot = []
latency_plot = []
latency_ewm_plot = []
data_plot = []
data_ewm_plot = []
reward_plot = []
reward_ewm_plot = []
payment_plot = []
payment_ewm_plot = []

energy_random_plot = []
latency_random_plot = []
payment_random_plot = []

energy_greedy_plot = []
latency_greedy_plot = []
payment_greedy_plot = []


for i in range(RANGE):
    energy_ewm_plot.append(energy_ewm[i])
    latency_ewm_plot.append(latency_ewm[i])
    payment_ewm_plot.append(payment_ewm[i])

    energy_random_plot.append(energy_random[i])
    latency_random_plot.append(latency_random[i])
    payment_random_plot.append(payment_random[i])

    energy_greedy_plot.append(energy_greedy[i])
    latency_greedy_plot.append(latency_greedy[i])
    payment_greedy_plot.append(payment_greedy[i])


episodes = np.arange(0, RANGE)

plt.plot(episodes, energy_ewm_plot, label='Double-DQN')
plt.plot(episodes, energy_random_plot, label='Random')
plt.plot(episodes, energy_greedy_plot, label='Greedy')
plt.legend()
plt.ylabel('Energy consumption (units)')
plt.xlabel('Episode')
plt.savefig('./results/energy-final.png')
plt.show()

plt.plot(episodes, latency_ewm_plot, label='Double-DQN')
plt.plot(episodes, latency_random_plot, label='Random')
plt.plot(episodes, latency_greedy_plot, label='Greedy')
plt.legend()
plt.ylabel('Total latency (units)')
plt.xlabel('Episode')
plt.savefig('./results/latency-final.png')
plt.show()

plt.plot(episodes, payment_ewm_plot, label='Double-DQN')
plt.plot(episodes, payment_random_plot, label='Random')
plt.plot(episodes, payment_greedy_plot, label='Greedy')
plt.legend()
plt.ylabel('Payment cost')
plt.xlabel('Episode')
plt.savefig('./results/payment-final.png')
plt.show()

# plt.plot(episodes, energy_plot, label='Energy')
# plt.plot(episodes, energy_ewm_plot, label='Mean value')
# plt.legend()
# plt.ylabel('Energy consumption (units)')
# plt.xlabel('Episode')
# plt.savefig('./results/{}-energy.png'.format(TEST_TH))
# plt.show()
#
# # plt.plot(episodes, latency_plot, label='Latency')
# plt.plot(episodes, latency_ewm_plot, label='Mean value')
# plt.legend()
# plt.ylabel('Total latency (units)')
# plt.xlabel('Episode')
# plt.savefig('./results/{}-latency.png'.format(TEST_TH))
# plt.show()
#
# plt.plot(episodes, data_plot, label='Training data')
# plt.plot(episodes, data_ewm_plot, label='Mean value')
# plt.legend()
# plt.ylabel('Training data (units)')
# plt.xlabel('Episode')
# plt.savefig('./results/{}-data.png'.format(TEST_TH))
# plt.show()
#
# plt.plot(episodes, reward_plot, label='Total reward')
# plt.plot(episodes, reward_ewm_plot, label='Mean value')
# plt.legend()
# plt.ylabel('Total reward')
# plt.xlabel('Episode')
# plt.savefig('./results/{}-reward.png'.format(TEST_TH))
# plt.show()
#
# plt.plot(episodes, payment_plot, label='Payment cost')
# plt.plot(episodes, payment_ewm_plot, label='Mean value')
# plt.legend()
# plt.ylabel('Payment cost')
# plt.xlabel('Episode')
# plt.savefig('./results/{}-payment.png'.format(TEST_TH))
# plt.show()


"""
Plot various payment 
From 172 to 175
"""
EWM_PAYMENT_WINDOW = 100

result_1 = './build/results-{}.xlsx'.format(196)
result_2 = './build/results-{}.xlsx'.format(197)
result_3 = './build/results-{}.xlsx'.format(198)
result_4 = './build/results-{}.xlsx'.format(195)

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

sampled_episodes = np.min([len(payment_1), len(payment_2), len(payment_3), len(payment_4)])
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

# for k in range(0, len(payments)):
#     # plt.plot(episode_axis, payments[k], label='Payment-{}'.format(k))
#     plt.plot(episode_axis, payments_ewm[k], label='$\lambda = {}$'.format(k + 2))

# plt.legend()
# plt.ylabel('Payment cost')
# plt.xlabel('Episode')
# plt.savefig('./results/{}-payment-comparision.png'.format(TEST_TH))
# plt.show()















