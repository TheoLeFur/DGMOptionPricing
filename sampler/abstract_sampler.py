from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple, Any
from dataclasses import dataclass
import torch


@dataclass
class Sample:
    domain_sample: torch.Tensor = None
    initial_sample: torch.Tensor = None
    boundary_sample: Tuple[torch.Tensor, Any] = None


class AbstractSampler(ABC):

    def __init__(
            self,
            n_points: int,
            device,
            time_dim: Optional[bool] = True,
            space_dim: Optional[int] = 1,
    ):
        """
        Basic sampler for Deep Galerkin Method
        :param n_points: Number of points we want to sample
        :param n_dim: Number of dimensions
        """

        self.n_points = n_points
        self.time_dim = time_dim
        self.space_dim = space_dim

        self.device = device

    @abstractmethod
    def sample_domain(self):
        """
        Sample points inside the domain
        :return: Array with sampled points
        """
        raise NotImplementedError

    @abstractmethod
    def sample_boundary(self):
        """
        Sample points on the boundary
        :return: Array with sampled points
        """
        raise NotImplementedError

    @abstractmethod
    def sample_initial(self):
        """
        Sample points on initial condition set.
        :return: Array with sampled points
        """
        raise NotImplementedError

    @abstractmethod
    def sample_all(self) -> Sample:
        raise NotImplementedError
