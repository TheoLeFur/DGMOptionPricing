from dataclasses import dataclass
from typing import Callable
from typing import Optional

import torch
import torch.autograd as grad
import torch.nn as nn

from pde.BasePDE import BasePDE
from pde.utils.payoffs.base_payoff import BasePayoff
from pde.utils.payoffs.arithmetic_payoff import ArithmeticPayoff


@dataclass
class BSData:
    correlation: float = 1
    volatility: float = 0
    interest_rate: float = 0
    dividend_rate: float = 0
    strike_price: float = 0
    payoff: BasePayoff = ArithmeticPayoff()


class FreeBoundaryPDE(BasePDE):

    def __init__(
            self,
            device,
            time_dim: Optional[bool],
            space_dim: Optional[int] = 1,
            bs_data: Optional[BSData] = None,
    ):
        """

        :param time_dim:
        :param space_dim:
        :param bs_data:
        """
        super().__init__(
            time_dim=time_dim,
            space_dim=space_dim,
            device=device
        )

        if bs_data is None:
            self.bs_data = BSData()
        else:
            self.bs_data = bs_data

        correlation_matrix = self.bs_data.correlation * torch.ones((self.space_dim, self.space_dim))
        self.correlation_matrix = correlation_matrix.fill_diagonal_(1).clone().to(self.device)

    @property
    def option_data(self) -> BSData:
        return self.bs_data

    @property
    def payoff(self) -> BasePayoff:
        return self.bs_data.payoff

    def differential_operator(self, model: nn.Module, **kwargs) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        :param model:
        :param kwargs:
        :return:
        """

        def diff_op(x: torch.Tensor) -> torch.Tensor:
            """
            :param x:
            :return:
            """
            model_output: torch.Tensor = model(x)
            first_derivative: torch.Tensor = grad.grad(
                outputs=model_output,
                inputs=x,
                grad_outputs=torch.ones(model_output.shape, device=self.device),
                create_graph=True,
                only_inputs=True
            )[0]

            time_derivative, space_derivative = first_derivative[:, 0].view(-1, 1), first_derivative[:, 1].view(-1, 1)
            second_space_derivative: torch.Tensor = grad.grad(
                outputs=space_derivative,
                inputs=x,
                grad_outputs=torch.ones(space_derivative.shape, device=self.device),
                create_graph=True,
                only_inputs=True
            )[0]

            second_space_derivative: torch.Tensor = second_space_derivative[:, 1].view(-1, 1)
            spatial_var: torch.Tensor = x[:, 1].view(-1, 1)

            return time_derivative + self.bs_data.interest_rate * spatial_var * space_derivative + \
                self.bs_data.volatility ** 2 * torch.matmul(
                    second_space_derivative,
                    self.correlation_matrix
                ) - self.bs_data.interest_rate * model_output

        return diff_op

    def time_boundary(self, model: nn.Module, **kwargs) -> Callable[[torch.Tensor], torch.Tensor]:
        def time_boundary_op(x: torch.Tensor):
            return model(x)

        return time_boundary_op

    def space_boundary(self) -> Callable:
        raise NotImplementedError("We have a free boundary PDE")
