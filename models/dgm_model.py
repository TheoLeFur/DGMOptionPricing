from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.autograd as grad
import torch.nn as nn

import utils.ptu as ptu
from models.abstract_model import AbstractModel
from pde.parabolic_pde import ParabolicPDE
from nn.dgm_net import DGMNet
from sampler.abstract_sampler import AbstractSampler
from sampler.american_call_option_sampler import AmericanCallOptionSampler
from utils.logger import Logger


class DGMModel(AbstractModel):
    def __init__(
            self,
            net_params: Dict,
            sampler: AbstractSampler,
            pde: ParabolicPDE,
            criterion: Callable = None,
            optimizer=None,
            device: torch.device = None
    ) -> None:

        """
        An instance of a DGMModel, that will serve for training the neural network to approximate the solution to
        the PDE
        :param net_params: network parameters
        :param sampler: Sampler
        :param criterion: Loss function criterion, defaulted to mean squared error
        :param optimizer: Optimizer used for gradient descent, defaulted to ADAM
        :param device: Device on which training occurs, defaulted to CPU
        """
        self.net_params = net_params
        self.n_dim: int = self.net_params["input_dim"] - 1

        self.model: DGMNet = DGMNet(
            **self.net_params
        )

        self.sampler: AbstractSampler = sampler
        self.pde: ParabolicPDE = pde

        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion

        if optimizer is None:
            self.optimizer = torch.optim.Adam
        else:
            self.optimizer = optimizer

        if device is None:
            self.device = "cpu"
        else:
            self.device = device

        self.logger = Logger()

    def loss_fn(
            self,
            x_domain: np.ndarray,
            x_boundary: Tuple[np.ndarray, np.ndarray],
            x_initial: np.ndarray
    ) -> torch.Tensor:
        """
        Implements the loss function for solving the PDE.
        :return: Value of the loss function
        """
        x_domain: torch.Tensor = ptu.from_numpy(x_domain, self.device, requires_grad=True)
        model_output_domain: torch.Tensor = self.model(x_domain)
        drift, diffusion = self.pde.get_drift(), self.pde.get_diffusion()
        drift, diffusion = (ptu.from_numpy(drift, self.device, requires_grad=True),
                            ptu.from_numpy(diffusion, self.device, requires_grad=True)
                            )

        gradients_domain: torch.Tensor = grad.grad(
            outputs=model_output_domain,
            inputs=x_domain,
            grad_outputs=torch.ones(model_output_domain.shape).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        time_derivative: torch.Tensor = gradients_domain[:, 0].view(-1, 1)
        space_gradients: torch.Tensor = gradients_domain[:, 1:].reshape(-1, self.n_dim)

        first_order_operator = torch.matmul(drift.reshape(-1,1), space_gradients.T)
        print(first_order_operator.shape)
        domain_loss_term = self.criterion(-time_derivative, first_order_operator)

        # TODO : implement the second derivative monte carlo approximation algorithm

        x_boundary_top, x_boundary_bottom = x_boundary

        x_boundary_top = ptu.from_numpy(x_boundary_top, self.device, requires_grad=True)
        x_boundary_bottom = ptu.from_numpy(x_boundary_bottom, self.device, requires_grad=True)

        model_output_boundary_top: torch.Tensor = self.model(x_boundary_top)
        model_output_boundary_bottom: torch.Tensor = self.model(x_boundary_bottom)

        x_initial: torch.Tensor = ptu.from_numpy(x_initial, self.device, requires_grad=True)
        model_output_initial: torch.Tensor = self.model(x_initial)

    def step(self):

        """
        Take a training step
        :return: Value of the loss function at current iteration
        """
        self.optimizer.zero_grad()
        loss: torch.Tensor = self.loss_fn()
        loss.backward()
        self.optimizer.step()

        self.logger.write_loss(loss.item())

        return loss.item()


if __name__ == '__main__':
    time_dim = 1
    space_dim = 2

    net_params: Dict = dict(
        input_dim=time_dim + space_dim,
        output_dim=1,
        n_layers=2,
        n_units=32,
        activation_fn=nn.Tanh()
    )

    pde = ParabolicPDE(
        drift = np.ones((space_dim, 1)),
        diffusion_matrix=np.ones((space_dim, space_dim)),
        time_dimension=True,
        space_dimension=space_dim,
        name="bs_pde"
    )

    sampler = AmericanCallOptionSampler(
        n_points=5,
        n_dim=space_dim,
        t_start=0,
        t_end=1,
        domain=[0, 2]
    )

    model = DGMModel(
        net_params=net_params,
        sampler=sampler,
        pde=pde
    )

    loss = model.loss_fn(
        x_domain=sampler.sample_domain(),
        x_boundary=sampler.sample_boundary(),
        x_initial=sampler.sample_initial()
    )
