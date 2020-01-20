import pandas
import matplotlib.pyplot as plt
import numpy as np

fig_size = plt.rcParams["figure.figsize"]
print("Current size: {}".format(fig_size))
fig_size[0] = 6.4
fig_size[1] = 4.8
plt.rcParams["figure.figsize"] = fig_size

arrow_properties = dict(
    facecolor="black", width=0.5,
    headwidth=6, shrink=0.05)

# RANDOM_TEST = './build/results-random-1.xlsx'
RANDOM_TEST = './build/q_learning-result-44.xlsx'
# GREEDY_TEST = './build/results-greedy-1.xlsx'
GREEDY_TEST = './build/q_learning-result-43.xlsx'
# DRL_TEST = './build/results-6.xlsx'
DRL_TEST = './build/q_learning-result-41.xlsx'
# Q_LEARNING_TEST = './build/q_learning-result-2.xlsx'
Q_LEARNING_TEST = './build/q_learning-result-42.xlsx'
DATA_Q1 = './build/q_learning-result-15.xlsx'
DATA_Q2 = './build/q_learning-result-16.xlsx'
DATA_Q3 = './build/q_learning-result-17.xlsx'
DATA_Q4 = './build/q_learning-result-18.xlsx'

EWM_WINDOW = 200
TEST_TH = 314
RANGE = 4000
LATENCY_UNIT = 60  # seconds
ENERGY_UNIT = 100  # Joul
"""
******************** Data quality ***************************
"""

df = pandas.read_excel(DRL_TEST)
df_qlearning = pandas.read_excel(Q_LEARNING_TEST)
df_random = pandas.read_excel(RANDOM_TEST)
df_greedy = pandas.read_excel(GREEDY_TEST)

df_data1 = pandas.read_excel(DATA_Q1)
df_data2 = pandas.read_excel(DATA_Q2)
df_data3 = pandas.read_excel(DATA_Q3)
df_data4 = pandas.read_excel(DATA_Q4)

data_q1_mean = [np.mean(df_data1['Training_data_mean_1'].values[3000:4000]),
                np.mean(df_data1['Training_data_mean_2'].values[3000:4000]),
                np.mean(df_data1['Training_data_mean_3'].values[3000:4000])]
data_q2_mean = [np.mean(df_data2['Training_data_mean_1'].values[3000:4000]),
                np.mean(df_data2['Training_data_mean_2'].values[3000:4000]),
                np.mean(df_data2['Training_data_mean_3'].values[3000:4000])]
data_q3_mean = [np.mean(df_data3['Training_data_mean_1'].values[3000:4000]),
                np.mean(df_data3['Training_data_mean_2'].values[3000:4000]),
                np.mean(df_data3['Training_data_mean_3'].values[3000:4000])]
data_q4_mean = [np.mean(df_data4['Training_data_mean_1'].values[3000:4000]),
                np.mean(df_data4['Training_data_mean_2'].values[3000:4000]),
                np.mean(df_data4['Training_data_mean_3'].values[3000:4000])]

# print(data_q1_mean, data_q2_mean, data_q3_mean, data_q4_mean)
# data_quality_legend = ('1:1:1', '2:2:1', '3:2:1', '4:2:1')
# plt.plot(data_quality_legend, np.array([data_q1_mean[0], data_q2_mean[0], data_q3_mean[0], data_q4_mean[0]]),
#          'b-*', label='Device-1')
# plt.plot(data_quality_legend, np.array([data_q1_mean[1], data_q2_mean[1], data_q3_mean[1], data_q4_mean[1]]),
#          'r-^', label='Device-2')
# plt.plot(data_quality_legend, np.array([data_q1_mean[2], data_q2_mean[2], data_q3_mean[2], data_q4_mean[2]]),
#          'g-o', label='Device-3')
#
# plt.xlabel('Data quality ratio $\eta_1 : \eta_2 : \eta_3 $ ')
# plt.ylabel('Number of data units taken')
# plt.ylim([600, 1400])
# plt.legend()
# plt.savefig('./results/data-quality-final.png')
# plt.show()

# get the values for a given column
energy = df['Energy'].values
latency = df['Latency'].values
data = df['Training_data_mean'].values
reward = df['Total_reward'].values
payment = df['Payment'].values

energy_q = df_qlearning['Energy'].ewm(EWM_WINDOW, adjust=False).mean()
latency_q = df_qlearning['Latency'].ewm(EWM_WINDOW, adjust=False).mean()
data_q = df_qlearning['Training_data_mean'].ewm(EWM_WINDOW, adjust=False).mean()
reward_q = df_qlearning['Total_reward'].ewm(EWM_WINDOW, adjust=False).mean()
payment_q = df_qlearning['Payment'].ewm(EWM_WINDOW, adjust=False).mean()

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

energy_ewm = df['Energy'].ewm(span=EWM_WINDOW * 10, adjust=False).mean()
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

energy_q_plot = []
latency_q_plot = []
payment_q_plot = []
reward_q_plot = []
data_q_plot = []

energy_random_plot = []
latency_random_plot = []
payment_random_plot = []
reward_random_plot = []
data_random_plot = []

energy_greedy_plot = []
latency_greedy_plot = []
payment_greedy_plot = []
reward_greedy_plot = []
data_greedy_plot = []

for i in range(RANGE):
    reward_ewm_plot.append(reward_ewm[i])
    energy_ewm_plot.append(energy_ewm[i] * ENERGY_UNIT)
    latency_ewm_plot.append(latency_ewm[i] * LATENCY_UNIT)
    payment_ewm_plot.append(payment_ewm[i])
    data_ewm_plot.append(data_ewm[i])

    reward_q_plot.append(reward_q[i])
    energy_q_plot.append(energy_q[i] * ENERGY_UNIT)
    latency_q_plot.append(latency_q[i] * LATENCY_UNIT)
    payment_q_plot.append(payment_q[i])
    data_q_plot.append(data_q[i])

    energy_random_plot.append(energy_random[i] * ENERGY_UNIT)
    latency_random_plot.append(latency_random[i] * LATENCY_UNIT)
    payment_random_plot.append(payment_random[i])
    reward_random_plot.append(reward_random[i])
    data_random_plot.append(data_random[i])

    energy_greedy_plot.append(energy_greedy[i] * ENERGY_UNIT)
    latency_greedy_plot.append(latency_greedy[i] * LATENCY_UNIT)
    payment_greedy_plot.append(payment_greedy[i])
    reward_greedy_plot.append(reward_greedy[i])
    data_greedy_plot.append(data_greedy[i])

episodes = np.arange(0, RANGE)
"""
**************** Reward *************************
"""
mean_rewards = [np.mean(df['Total_reward'].values[3000:4000]),
                np.mean(df_qlearning['Total_reward'].values[3000:4000]),
                np.mean(df_random['Total_reward'].values[3000:4000]),
                np.mean(df_greedy['Total_reward'].values[3000:4000])
                ]
# plt.annotate(
#     "DQN", xy=(1500, 160),
#     xytext=(1000, 300),
#     arrowprops=arrow_properties)
# plt.annotate(
#     "Q-leaning", xy=(3000, 120),
#     xytext=(3300, 30),
#     arrowprops=arrow_properties)
# plt.annotate(
#     "Greedy", xy=(3000, -30),
#     xytext=(3300, -170),
#     arrowprops=arrow_properties)
# plt.annotate(
#     "Random", xy=(3200, -460),
#     xytext=(3500, -560),
#     arrowprops=arrow_properties)

plt.plot(episodes, reward_ewm_plot, label='DQN')
plt.plot(episodes, reward_q_plot, label='Q-learning')
plt.plot(episodes, reward_greedy_plot, label='Greedy')
plt.plot(episodes, reward_random_plot, label='Random')
plt.xlabel('Episode')
plt.ylabel('Cumulative reward')
plt.legend()
# plt.savefig('./results/reward-final.png')
plt.show()
"""
****************** Energy ***********************
"""
mean_energies = [np.mean(df['Energy'].values[3000:4000]),
                 np.mean(df_qlearning['Energy'].values[3000:4000]),
                 np.mean(df_random['Energy'].values[3000:4000]),
                 np.mean(df_greedy['Energy'].values[3000:4000])
                 ]
plt.annotate(
    "DQN", xy=(2500, 6.3 * ENERGY_UNIT),
    xytext=(2100, 5.9 * ENERGY_UNIT),
    arrowprops=arrow_properties)
plt.annotate(
    "Q-leaning", xy=(3000, 7.16 * ENERGY_UNIT),
    xytext=(3200, 7.4 * ENERGY_UNIT),
    arrowprops=arrow_properties)
plt.annotate(
    "Greedy", xy=(3000, 8.9 * ENERGY_UNIT),
    xytext=(3300, 8.7 * ENERGY_UNIT),
    arrowprops=arrow_properties)
plt.annotate(
    "Random", xy=(3000, 7.7 * ENERGY_UNIT),
    xytext=(3300, 7.9 * ENERGY_UNIT),
    arrowprops=arrow_properties)

# plt.plot(episodes, energy_ewm_plot, label='DQN')
# plt.plot(episodes, energy_q_plot, label='Q-learning')
# plt.plot(episodes, energy_random_plot, label='Random')
# plt.plot(episodes, energy_greedy_plot, label='Greedy')
# plt.arrow(1, -0.00010, 0, -0.00005, length_includes_head=True,
#           head_width=0.08, head_length=0.00002)
# plt.xlabel('Episode')
# plt.ylabel('Energy consumption (Joule)')
# plt.legend()
# plt.savefig('./results/energy-final.png')
# plt.show()

"""
****************** Latency ***********************
"""
mean_latencies = [np.mean(df['Latency'].values[3000:4000]),
                  np.mean(df_qlearning['Latency'].values[3000:4000]),
                  np.mean(df_random['Latency'].values[3000:4000]),
                  np.mean(df_greedy['Latency'].values[3000:4000])
                  ]
plt.annotate(
    "DQN", xy=(1600, 6 * LATENCY_UNIT),
    xytext=(1100, 2 * LATENCY_UNIT),
    arrowprops=arrow_properties)
plt.annotate(
    "Q-leaning", xy=(3000, 7.2 * LATENCY_UNIT),
    xytext=(3200, 10 * LATENCY_UNIT),
    arrowprops=arrow_properties)
plt.annotate(
    "Greedy", xy=(3000, 31 * LATENCY_UNIT),
    xytext=(3300, 27 * LATENCY_UNIT),
    arrowprops=arrow_properties)
plt.annotate(
    "Random", xy=(3000, 19.8 * LATENCY_UNIT),
    xytext=(3300, 17 * LATENCY_UNIT),
    arrowprops=arrow_properties)

# plt.plot(episodes, latency_ewm_plot, label='DQN')
# plt.plot(episodes, latency_q_plot, label='Q-learning')
# plt.plot(episodes, latency_random_plot, label='Random')
# plt.plot(episodes, latency_greedy_plot, label='Greedy')
# plt.legend()
# plt.ylabel('Total latency (seconds)')
# plt.xlabel('Episode')
# plt.savefig('./results/latency-final.png')
# plt.show()

"""
***************** Payment ************************
"""
mean_payments = [np.mean(df['Payment'].values[3000:4000]),
                 np.mean(df_qlearning['Payment'].values[3000:4000]),
                 np.mean(df_random['Payment'].values[3000:4000]),
                 np.mean(df_greedy['Payment'].values[3000:4000])
                 ]
plt.annotate(
    "DQN", xy=(3030, 5.63),
    xytext=(3320, 5.4),
    arrowprops=arrow_properties)
plt.annotate(
    "Q-leaning", xy=(3030, 5.12),
    xytext=(3320, 4.92),
    arrowprops=arrow_properties)
plt.annotate(
    "Greedy", xy=(600, 5.95),
    xytext=(890, 5.75),
    arrowprops=arrow_properties)
plt.annotate(
    "Random", xy=(3030, 3.9),
    xytext=(3320, 3.7),
    arrowprops=arrow_properties)

# plt.plot(episodes, data_ewm_plot, label='DQN')
# plt.plot(episodes, data_q_plot, label='Q-learning')
# plt.plot(episodes, data_random_plot, label='Random')
# plt.plot(episodes, data_greedy_plot, label='Greedy')
# plt.legend()
# plt.ylabel('Average data units/iteration')
# plt.xlabel('Episode')
# plt.savefig('./results/data-units.pdf')
# plt.show()

"""
************************ Mining rate *************************
"""
df_ql_4 = pandas.read_excel('./build/q_learning-result-4.xlsx')
# df_ql_5 = pandas.read_excel('./build/q_learning-result-5.xlsx')
df_ql_5 = pandas.read_excel('./build/results-6.xlsx')
df_ql_6 = pandas.read_excel('./build/q_learning-result-27.xlsx')
df_ql_7 = pandas.read_excel('./build/q_learning-result-28.xlsx')
df_ql_8 = pandas.read_excel('./build/q_learning-result-29.xlsx')
df_ql_9 = pandas.read_excel('./build/q_learning-result-30.xlsx')

mining_rates = [np.mean(df_ql_4['Mining rate'].values[3000:4000]),
                np.mean(df_ql_5['Mining rate'].values[3000:4000]),
                np.mean(df_ql_6['Mining rate'].values[3000:4000]),
                np.mean(df_ql_7['Mining rate'].values[3000:4000]),
                np.mean(df_ql_8['Mining rate'].values[3000:4000]),
                np.mean(df_ql_9['Mining rate'].values[3000:4000])
                ]
mining_rate_1 = df_ql_6['Mining rate'].ewm(EWM_WINDOW * 10, adjust=False).mean()
mining_rate_2 = df_ql_7['Mining rate'].ewm(EWM_WINDOW * 10, adjust=False).mean()
mining_rate_3 = df_ql_5['Mining rate'].ewm(EWM_WINDOW * 10, adjust=False).mean()
mining_rate_4 = df_ql_9['Mining rate'].ewm(EWM_WINDOW * 10, adjust=False).mean()

avg_mining_rates_1 = []
avg_mining_rates_2 = []
avg_mining_rates_3 = []
avg_mining_rates_4 = []

for i in range(RANGE):
    avg_mining_rates_1.append(mining_rate_1[i])
    avg_mining_rates_2.append(mining_rate_2[i])
    avg_mining_rates_3.append(mining_rate_3[i])
    avg_mining_rates_4.append(mining_rate_4[i])

# plt.plot(episodes, avg_mining_rates_1, label='$ \lambda = 1$')
# plt.plot(episodes, avg_mining_rates_2, label='$ \lambda = 2$')
# plt.plot(episodes, avg_mining_rates_3, label='$ \lambda = 3$')
# plt.plot(episodes, avg_mining_rates_4, label='$ \lambda = 4$')
# plt.xlabel('Episode')
# plt.ylabel('Average mining rate')
# plt.legend()
# plt.savefig('./results/mining-final.png')
# plt.show()
# print('Mean values:\n Reward: {}\n Energy: {},\n Latency: {},\n Payment: {},\n Mining rate: {}'
#       .format(mean_rewards, mean_energies, mean_latencies, mean_payments, mining_rates))

"""
Payment function visualization
"""
x_asis = np.arange(1, 11)
payment_costs = [0.8 / np.log(1+m) for m in range(1, 11)]
plt.plot(x_asis, payment_costs, '-^')
plt.xlabel('Number of pending transactions $m$')
plt.ylabel('Cost paid to the mining winner')
plt.xticks(x_asis)
plt.grid(b=True, which='both', axis='both', linestyle='-.', linewidth=0.2)
plt.legend()
plt.savefig('./results/payment-to-winner.pdf')
plt.show()






