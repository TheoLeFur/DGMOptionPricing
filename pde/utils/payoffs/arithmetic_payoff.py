import torch
from typing import Optional
from utils.ptu import *

from pde.utils.payoffs.base_payoff import BasePayoff


class ArithmeticPayoff(BasePayoff):
    """
    This class gies an arithmetic payoff. In a market composed of S1, ..., SN underlying stocks, the
    payoff is given by arithmetic_mean(S1(T), ... SN(T)), where T is the exercise date of the option.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(BasePayoff, cls).__new__(cls)
        else:
            return cls._instance

    def payoff(
            self,
            x: torch.Tensor,
            K: float,
            call: Optional[bool] = True,
            device: Optional = "cpu"
    ) -> torch.Tensor:

        mean = torch.mean(x, dim=1).reshape(-1, 1)
        strikes = torch.ones_like(mean, device=device) * K
        null_tensor = torch.zeros_like(strikes, device=device)

        print("mean", mean.device)
        print("strikes", strikes.device)
        print("null", null_tensor.device)

        return torch.fmax(mean - strikes, null_tensor) if call else torch.fmax(strikes - mean, null_tensor)
