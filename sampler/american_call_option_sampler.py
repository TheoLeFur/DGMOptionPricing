import numpy as np
from typing import List, Optional, Callable, Tuple

from sampler.abstract_sampler import AbstractSampler


class AmericanCallOptionSampler(AbstractSampler):

    def __init__(
            self,
            n_points: int,
            n_dim: int,
            t_start: float,
            t_end: float,
            domain: List[int]
    ):
        """
        This sampler will be used for solving the Black-Scholes PDE for american Call Option
        :param n_points: Number of points
        :param n_dim: Number of dimensions
        :param t_start: Start time
        :param t_end: End time
        :param domain: Stock price domain, that is assumed to be of the form  [a, b] x ... x [a, b]
        # TODO implement a more general sampler where one allows different strike prices.

        """
        super().__init__(
            n_points=n_points,
            n_dim=n_dim
        )

        self.t_start = t_start
        self.t_end = t_end
        self.domain = domain

    def sample_domain(self, probability_law: Optional[Callable] = None) -> np.ndarray:
        """
        Sample points from the interior of the domain
        :param probability_law: Probability Law we wish to sample from
        :return: Output array
        """
        if probability_law is None:
            probability_law: Callable = np.random.uniform
            sampled_points: np.ndarray = np.concatenate(
                [probability_law(self.t_start, self.t_end, (self.n_points, 1)),
                 probability_law(
                     self.domain[0], self.domain[1], (self.n_points, self.n_dim))
                 ],
                axis=1
            )

            return sampled_points
        else:
            raise NotImplementedError

    def sample_boundary(self, probability_law: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points from the interior of the domain
        :param probability_law: Probability Law we wish to sample from
        :return: Output array
        """

        if probability_law is None:
            probability_law = np.random.uniform
            if self.n_dim > 1:
                sampled_points_top: np.ndarray = np.concatenate(
                    [probability_law(self.t_start, self.t_end, size=(self.n_points, 1)),
                     np.ones((self.n_points, 1)) * self.domain[1],
                     probability_law(self.domain[0], self.domain[1], (self.n_points, self.n_dim - 1))],
                    axis=1
                )
                sampled_points_bottom: np.ndarray = np.concatenate(
                    [probability_law(self.t_start, self.t_end, size=(self.n_points, 1)),
                     np.ones((self.n_points, 1)) * self.domain[0],
                     probability_law(self.domain[0], self.domain[1], (self.n_points, self.n_dim - 1))],
                    axis=1
                )
            else:
                sampled_points_top: np.ndarray = np.concatenate(
                    [
                        probability_law(self.t_start, self.t_end, size=(self.n_points, 1)),
                        np.ones((self.n_points, 1)) * self.domain[1]
                    ],
                    axis=1
                )

                sampled_points_bottom: np.ndarray = np.concatenate(
                    [
                        probability_law(self.t_start, self.t_end, size=(self.n_points, 1)),
                        np.ones((self.n_points, 1)) * self.domain[0],

                    ],
                    axis=1
                )
            return sampled_points_top, sampled_points_bottom

        else:
            raise NotImplementedError

    def sample_initial(self, probability_law: Optional[Callable] = None) -> np.ndarray:

        """
        Sample points for the set given by the initial condition
        :param probability_law: probability distribution we sample from
        :return: Sampled points
        """

        if probability_law is None:
            probability_law = np.random.uniform

            sampled_points = np.concatenate([
                np.ones((self.n_points, 1)) * self.t_end,
                probability_law(self.domain[0], self.domain[1], (self.n_points, self.n_dim))
            ],
                axis=1
            )
            return sampled_points
        else:
            raise NotImplementedError
