import torch
from typing import Optional

from pde.utils.payoffs.base_payoff import BasePayoff


class GeometricPayoff(BasePayoff):
    """

    This class implements a geometric payoff for derivative securities. In a market composed of S1, ..., SN
    underlying stocks,the payoff is given by:
        geometric_mean(S1(T), ... SN(T)),
    where T is the exercise date of the option.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(BasePayoff, cls).__new__(cls)
        else:
            return cls._instance

    def payoff(self, x: torch.Tensor, K: float, call: Optional[bool] = True) -> torch.Tensor:

        # TODO: correct this
        dim: int = x.shape[0]
        mean = torch.pow(torch.cumprod(x, dim=0), 1 / dim)
        return max(mean - K, 0) if call else max(K - mean, 0)
