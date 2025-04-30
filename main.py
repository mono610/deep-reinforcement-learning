from envs.cartpole_env import make_env
from utils.util import load_config

def main():
    config = load_config('configs/config.yaml')
    env = make_env(config['env']['name'])
    print('Environment name: ' + config['env']['name'])

    obs, _ = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            print('episode done')
            break
    env.close()

if __name__ == "__main__":
    main()
