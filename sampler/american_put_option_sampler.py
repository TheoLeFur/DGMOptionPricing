from typing import List, Tuple

from sampler.abstract_sampler import AbstractSampler, Sample
from utils.autodiff.ptu import *


class AmericanPutOptionSampler(AbstractSampler):

    def __init__(
            self,
            n_points: int,
            time_range: List,
            space_range: List,
            device,
            time_dim: Optional[bool] = True,
            space_dim: Optional[int] = 1,
    ):
        super().__init__(
            n_points=n_points,
            time_dim=time_dim,
            space_dim=space_dim,
            device=device
        )

        self.total_dim = self.space_dim + 1 if self.time_dim else 0

        self.time_range: List[float] = time_range
        self.space_range: List[float] = space_range

        assert len(self.time_range) == 2
        assert len(self.space_range) == 2

    def sample_domain(self) -> torch.Tensor:
        sampled = np.concatenate(
            [
                np.random.uniform(self.time_range[0], self.time_range[1], size=(self.n_points, 1)),
                np.random.uniform(self.space_range[0], self.space_range[1], size=(self.n_points, self.space_dim))
            ],
            axis=1
        )

        return from_numpy(sampled, self.device)

    def sample_boundary(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = np.concatenate([
            np.random.uniform(
                self.time_range[0],
                self.time_range[1],
                size=(self.n_points, 1)
            ),
            self.space_range[1] * np.ones((self.n_points, self.space_dim))
        ], axis=1)

        x2 = np.concatenate([
            np.random.uniform(
                self.time_range[0],
                self.time_range[1],
                size=(self.n_points, 1)
            ),
            self.space_range[0] * np.ones((self.n_points, self.space_dim))
        ], axis=1)

        return from_numpy(x1, self.device), from_numpy(x2, self.device)

    def sample_initial(self) -> torch.Tensor:
        sampled = np.concatenate([
            np.ones((self.n_points, 1)) * self.time_range[1],
            np.random.uniform(self.space_range[0], self.space_range[1], size=(self.n_points, self.space_dim))
        ],
            axis=1)

        return from_numpy(sampled, self.device)

    def sample_all(self) -> Sample:

        return Sample(
            domain_sample=self.sample_domain(),
            initial_sample=self.sample_initial(),
            boundary_sample=self.sample_boundary()
        )
