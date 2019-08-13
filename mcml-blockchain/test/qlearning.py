# Solving frozen-lake using Q-learning
# @author: Hieu Nguyen
#  https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

import numpy as np
import gym
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(0.5)

env = gym.make('Taxi-v2')
print(env.action_space, env.observation_space)
# env.env.s = 328
env.reset()
env.render()

# Q-learning
q_table = np.zeros([env.observation_space.n, env.action_space.n])

"""
Training the agent
"""
# Training parameters
NB_EPISODES = 100000
# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1
# For plotting metrics
all_epochs = []
all_penalties = []
rewards_plot = []

for i in range(NB_EPISODES):
    state = env.reset()
    episode_reward = 0

    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        # for plotting
        episode_reward += reward

    rewards_plot.append(episode_reward)
    if i % 1000 == 0:
        clear_output(wait=True)
        print('Episode: {}, Episode_reward: {}, Episode_epochs: {}, Penalties: {}'.
              format(i, episode_reward, epochs, penalties))

print('Training finished./n')
print(q_table[328])
plt.plot(np.arange(NB_EPISODES), rewards_plot)

plt.show()
# End of Q-learning

"""
Test learnt agent
"""
total_epochs, total_penalties = 0, 0
episodes = 100
frames = []

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == - 10:
            penalties += 1

        frames.append({
                    'frame': env.render(mode='ansi'),
                    'state': state,
                    'action': action,
                    'reward': reward
                })
        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print_frames(frames)
print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penyoualties / episodes}")

# Random policy:
# epochs = 0
# penalties, reward = 0, 0
# frames = []
# done = False
#
# for e in range(10000):
#     action = env.action_space.sample()
#     state, reward, done, info = env.step(action)
#
#     if reward == -10:
#         penalties += 1
#
#     frames.append({
#         'frame': env.render(mode='ansi'),
#         'state': state,
#         'action': action,
#         'reward': reward
#     })
#     epochs += 1
#     if reward == 20:
#         break
#
# print('Timesteps taken: {}'.format(epochs))
# print('Penalties incurred: {}'.format(penalties))
#
# print_frames(frames)
# print_frames('End training')
# End of Random-policy
