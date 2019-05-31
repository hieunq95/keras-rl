from environment import Environment

env = Environment()
env.seed(1000)


print(env.observation_space, env.action_space)

for i in range(10):
    state_init = env.reset()
    # print(state_init)

for t in range(2000):
    #test random action

    state, reward , done, info = env.step(env.action_space.sample())
    print(state, reward, done, info)
