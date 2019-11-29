from environment import AV_Environment
import matplotlib.pyplot as plt


env = AV_Environment()

print(env.observation_space, env.action_space)
histogram = []
for e in range(10000):
    plot_target = 0
    cumulative_reward = 0
    for t in range(1000):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        cumulative_reward += reward
        plot_target = reward
        histogram.append(plot_target)
        # print(t, next_state, reward, done, info)
        if done == True:
            # print('Break at t = {}'.format(t))
            plot_target = reward
            histogram.append(plot_target)
            env.reset()
            break

plt.hist(histogram)
plt.show()