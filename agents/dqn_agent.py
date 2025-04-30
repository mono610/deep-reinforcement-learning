from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent
from networks.q_network import QNetwork

class DQNAgent(BaseAgent):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            config: Dict[str, Any],
            device: torch.device
        ) -> None:
        self.device = device
        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.target_update_freq = config['target_update_freq']

        self.q_network = QNetwork(state_dim, action_dim, config['hidden_dim']).to(device)
        self.target_network = QNetwork(state_dim, action_dim, config['hidden_dim']).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['lr'])
        self.criterion = nn.MSELoss()

        self.step_count = 0


    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.q_network.output_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()


    def update(self, batch: Dict[str, np.ndarray]) -> float:
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).unsqueeze(1).to(self.device)

        q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()
