import numpy as np
from typing import List

from sampler.abstract_sampler import AbstractSampler


class AmericanCallOptionSampler(AbstractSampler):

    def __init__(
            self,
            n_points: int,
            n_dim: int,
            t_start: float,
            t_end: float,
            domain: List[List]
    ):
        """
        This sampler will be used for solving the Black-Scholes PDE for american Call Option
        :param n_points: Number of points
        :param n_dim: Number of dimensions
        :param t_start: Start time
        :param t_end: End time
        :param domain: Stock price domain, that is assumed to be of the form  [a1, b1] x ... x [an, bn]
        TODO : Implement a class domain for more complex domains
        """
        super().__init__(
            n_points=n_points,
            n_dim=n_dim
        )

        self.t_start = t_start
        self.t_end = t_end

        if len(domain) != self.n_dim:
            raise ValueError("Dimension of the domain and n_dim do not match")
        for _, interval in domain:
            if interval[0] >= interval[1]:
                raise ValueError("Lower bound of interval should be strictly smaller than upper bound")

        self.domain = domain

    def sample_domain(self, probability_law=None) -> np.ndarray:
        if probability_law is None:
            probability_law = np.random.uniform
            sampled_points: np.ndarray = np.concatenate(
                [probability_law(self.t_start, self.t_end, (self.n_points, 1)),
                 *[probability_law(
                     self.domain[i][0], self.domain[i][1], (self.n_points, 1)) for i, _ in enumerate(self.domain)]
                 ]
            )
            return sampled_points
        else:
            raise NotImplementedError

    def sample_boundary(self, probability_law=None) -> np.ndarray:
        pass

    def sample_initial(self, probability_law=None) -> np.ndarray:
        pass
