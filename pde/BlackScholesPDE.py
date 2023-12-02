from pde.BasePDE import BasePDE
import numpy as np
from typing import Callable
from typing import Optional

import torch
import torch.nn as nn


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
            end_time: Optional[float] = None,
            device=None
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

        if device is not None:
            self.device = device
        else:
            self.device = "cpu"

    def zero_order_op(
            self
    ):
        raise NotImplementedError

    def first_order_op(
            self,
            y_hat: torch.Tensor,
            x: torch.Tensor
    ):
        grads = torch.autograd.grad(
            y_hat,
            x,
            grad_outputs=torch.ones(y_hat.shape).to(self.device),
            retain_graph=True,
            create_graph=True,
            only_inputs=True)[0]

        time_gradients, space_gradients = grads[:, 0].view(-1, 1), grads[:, 1].view(-1, 1)
        return time_gradients + self.interest_rate * x * space_gradients

    def second_order_op(
            self,
            space_gradients: torch.Tensor,
            x: torch.Tensor,
            monte_carlo: Optional[bool] = False
    ):

        if monte_carlo:
            return torch.zeros(1)
        else:
            second_order_grads = torch.autograd.grad(
                space_gradients,
                x,
                grad_outputs=torch.ones(space_gradients.shape).to(self.device),
                create_graph=True,
                only_inputs=True)[0][:, 1].view(-1, 1)

            return 0.5 * torch.square(self.volatility * x) * second_order_grads



