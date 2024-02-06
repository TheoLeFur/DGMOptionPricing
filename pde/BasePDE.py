from abc import ABC
from typing import Optional


class BasePDE(ABC):

    def __init__(
            self,
            n_dimensions: int,
            time_dim: bool,
            start_time: Optional[float] = None,
            end_time: Optional[float] = None):

        self.time_dim = time_dim
        if time_dim:
            self.spatial_dim = n_dimensions - 1
            if start_time is None:
                self.start_time = 0
            else:
                self.start_time = start_time
            if end_time is None:
                self.end_time = 1
            else:
                self.end_time = end_time

            if self.start_time > self.end_time:
                raise ValueError("Starting time cannot be smaller than ending time")
        else:
            self.spatial_dim = n_dimensions
