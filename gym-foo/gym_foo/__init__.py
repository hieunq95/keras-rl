from gym.envs.registration import register

register(
    id='foo-v0',
    entry_point='gym_foo.envs:FooEnv',
)

register(
    id='OneRoundNondeterministicReward-v0',
    entry_point='gym_foo.envs:OneRoundNondeterministicRewardEnv',
)