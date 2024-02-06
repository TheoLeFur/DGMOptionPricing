from pde.BasePDE import BasePDE
import numpy as np
from typing import Callable
from typing import Optional


class BlackScholesPDE(BasePDE):

    def __init__(
            self,
            n_dimensions: int,
            time_dim: bool,
            drift: float,
            volatility: float,
            correlation: np.ndarray,
            interest_rate: float,
            payoff: Callable,
            start_time: Optional[float] = None,
            end_time: Optional[float] = None
    ):
        super().__init__(
            n_dimensions=n_dimensions,
            time_dim=time_dim,
            start_time=start_time,
            end_time=end_time
        )

        if correlation.shape != (self.spatial_dim, self.spatial_dim):
            raise ValueError("Correlation matrix has incorrect shape")

        self.drift = drift
        self.volatility = volatility
        self.interest_rate = interest_rate
        self.terminal_payoff = payoff
