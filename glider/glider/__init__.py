from gym.envs.registration import register

register(
    id='glider-v0',
    entry_point='glider.envs:gliderEnv',
)
