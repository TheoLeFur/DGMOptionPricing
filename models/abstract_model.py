from abc import ABC, abstractmethod
from typing import Dict
import torch


class AbstractModel(ABC):

    @abstractmethod
    def loss_fn(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def step(self, *args, **kwargs) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def log(self, data: Dict) -> None:
        raise NotImplementedError
