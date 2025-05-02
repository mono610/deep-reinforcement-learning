import os
from typing import Dict, Any

import gymnasium as gym

from agents.base_agent import BaseAgent
from utils.replay_buffer import ReplayBuffer
from utils.plotting import plot_rewards

class Trainer:
    def __init__(
        self,
        env: gym.Env,
        agent: BaseAgent,
        buffer: ReplayBuffer,
        config: Dict[str, Any]
    ) -> None:
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.config = config

        self.rewards: list[float] = []
        self.output_dir = "output/img"


    def train(self) -> None:
        for episode in range(self.config["num_episodes"]):
            state, _ = self.env.reset()
            total_reward = 0

            for step in range(self.config["max_steps"]):
                action = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.buffer.store_transition(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if len(self.buffer) >= self.config["batch_size"]:
                    self.agent.update(self.buffer)

                if done:
                    break

            self.rewards.append(total_reward)

            print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

        plot_path = os.path.join("output/img", "rewards_plot.png")
        plot_rewards(self.rewards, plot_path)
        model_path = os.path.join("output/model", "dqn_model.pth")
        self.agent.save(model_path)