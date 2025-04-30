import gymnasium as gym

def make_env(env_name: str):
    return gym.make(env_name)
