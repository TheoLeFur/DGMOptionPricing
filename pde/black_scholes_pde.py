from typing import Optional
import utils.ptu as ptu
from abc import ABC
import torch
from pde.parabolic_pde import ParabolicPDE
import numpy as np
import torch.autograd as grad


class BlackScholesPDE(ParabolicPDE):

    def __init__(
            self,
            drift: np.ndarray,
            diffusion_matrix: np.ndarray,
            volatility: np.ndarray,
            rate: float,
            name: str,
            time_dimension: bool = False,
            space_dimension: int = 1,
    ):

        super().__init__(
            drift=drift,
            diffusion_matrix=diffusion_matrix,
            name=name,
            time_dimension=time_dimension,
            space_dimension=space_dimension
        )
        self.volatility = volatility

        if rate < 0:
            raise ValueError("Negative rates not allowed")
        else:
            self.rate = rate

    def get_zeroth_order_differential_operator(
            self,
            model_output_domain: torch.Tensor
    ):

        return - self.rate * model_output_domain

    def get_first_order_differential_operator(
            self,
            model_output_domain: torch.Tensor,
            x_domain: torch.Tensor,
            create_graph: Optional[bool] = True,
            retain_graph: Optional[bool] = True,
            only_inputs: Optional[bool] = True,
    ):

        gradients_domain: torch.Tensor = grad.grad(
            outputs=model_output_domain,
            inputs=x_domain,
            grad_outputs=torch.ones(model_output_domain.shape),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        time_derivative: torch.Tensor = gradients_domain[:, 0].view(-1, 1)
        space_gradients: torch.Tensor = gradients_domain[:, 1:].reshape(-1, self.space_dimension)
        drift: torch.Tensor = ptu.from_numpy(self.drift)

        return time_derivative + torch.dot(drift, space_gradients)

    def get_second_order_differential_operator(
            self,
            model_output_domain: torch.Tensor,
            x_domain: torch.Tensor,
            delta: float,
            use_monte_carlo: Optional[bool] = True,
            create_graph: Optional[bool] = True,
            retain_graph: Optional[bool] = True,
            only_inputs: Optional[bool] = True,
    ):

        diffusion = ptu.from_numpy(self.diffusion_matrix)
        random_gaussian_vector = ptu.sample_multi_dimensional_bm(
            delta,
            diffusion
        )

        gradients_domain: torch.Tensor = grad.grad(
            outputs=model_output_domain,
            inputs=x_domain,
            grad_outputs=torch.ones(model_output_domain.shape),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        space_gradients: torch.Tensor = gradients_domain[:, 1:].reshape(-1, self.space_dimension)
