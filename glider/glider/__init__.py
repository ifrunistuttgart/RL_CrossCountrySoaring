from gym.envs.registration import register

register(
    id='glider2D-v1',
    entry_point='glider.envs:gliderEnv2D',
)
register(
    id='glider3D-v0',
    entry_point='glider.envs:GliderEnv3D',
)
