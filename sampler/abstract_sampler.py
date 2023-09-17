from abc import ABC, abstractmethod
import numpy as np


class AbstractSampler(ABC):

    def __init__(
            self,
            n_points: int,
            n_dim: int):
        """
        Basic sampler for Deep Garlekin Method
        :param n_points: Number of points we want to sample
        :param n_dim: Number of dimensions
        """
        self.n_points = n_points
        self.n_dim = n_dim

    @abstractmethod
    def sample_domain(self) -> np.ndarray:
        """
        Sample points inside the domain
        :return: Array with sampled points
        """
        raise NotImplementedError

    @abstractmethod
    def sample_boundary(self) -> np.ndarray:
        """
        Sample points on the boundary
        :return: Array with sampled points
        """
        raise NotImplementedError

    @abstractmethod
    def sample_initial(self) -> np.ndarray:
        """
        Sample points on initial condition set.
        :return: Array with sampled poijts
        """
        raise NotImplementedError
