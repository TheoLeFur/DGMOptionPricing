from abc import ABC, abstractmethod
from typing import Optional, Callable
import torch.nn as nn


class BasePDE(ABC):

    def __init__(
            self,
            device,
            time_dim: Optional[bool],
            space_dim: Optional[int] = 1,
    ):
        self.time_dim = time_dim
        self.space_dim = space_dim
        self.device = device

    @abstractmethod
    def differential_operator(self, model: nn.Module, **kwargs) -> Callable:
        """
        Spatial differential operator that acts on u(t,x)
        :return: Callable value on u
        """
        raise NotImplementedError

    @abstractmethod
    def time_boundary(self, **kwargs) -> Callable:
        """
        Boundary condition for t = 0
        :return: Callable on x
        """
        raise NotImplementedError

    @abstractmethod
    def space_boundary(self) -> Callable:
        """

        :return: Callable on x, t, where x is on the boundary of the domain
        """
        raise NotImplementedError
