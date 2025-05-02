import matplotlib.pyplot as plt

def plot_rewards(rewards: list[float], path: str) -> None:
    plt.figure()
    plt.plot(rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig(path)
    plt.close()