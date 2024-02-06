from abc import ABC, abstractmethod
from typing import Optional
import torch


class BasePayoff(ABC):

    @abstractmethod
    def payoff(self, x: torch.Tensor, K: float, call: Optional[bool] = True) -> torch.Tensor:
        raise NotImplementedError
