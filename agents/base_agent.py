from abc import ABC, abstractmethod
import numpy as np

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        """観測から行動を選択する"""
        pass

    @abstractmethod
    def update(self) -> None:
        """エージェントの学習処理"""
        pass
