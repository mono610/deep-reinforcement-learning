from collections import deque
import random
from typing import Deque, Dict

import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = Deque[Dict[str, np.ndarray]] = deque(maxlen=capacity)


    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        self.buffer.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        })


    def sampling(self, batch_size: int) -> Dict[str, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        return {
            "states": np.array([b["states"] for b in batch]),
            "actions": np.array([b["actions"] for b in batch]),
            "rewards": np.array([b["rewards"] for b in batch]),
            "next_states": np.array([b["next_states"] for b in batch]),
            "dones": np.array([b["dones"] for b in batch])
        }


    def __len__(self) -> int:
        return len(self.buffer)