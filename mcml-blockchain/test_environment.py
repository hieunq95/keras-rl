from environment import Environment

env = Environment()
env.seed(1000)
print(env.observation_space, env.action_space)

for i in range(10):
    state_init = env.reset()
    print(state_init)

