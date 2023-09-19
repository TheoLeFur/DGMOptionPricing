import numpy as np
from typing import Tuple

from models.abstract_model import AbstractModel
from nn.dgm_net import DGMNet
import torch
import torch.nn as nn
from sampler.abstract_sampler import AbstractSampler
from typing import Callable, Dict
from utils.logger import Logger
import torch.autograd as grad
from sampler.american_call_option_sampler import AmericanCallOptionSampler


class DGMModel(AbstractModel):
    def __init__(
            self,
            net_params: Dict,
            sampler: AbstractSampler,
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

        self.model = DGMNet(
            **self.net_params
        )

        self.sampler = sampler
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
    ) -> torch.Tensor:
        """
        Implements the loss function for solving the PDE.
        :return: Value of the loss function
        """
        print(x_domain.dtype)
        x_domain: torch.Tensor = torch.from_numpy(x_domain).to(self.device)
        model_output_domain: torch.Tensor = self.model(x_domain)

        gradients = grad.grad(
            outputs=model_output_domain,
            inputs=x_domain,
            grad_outputs=torch.ones(model_output_domain).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        print(gradients)

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
    net_params: Dict = dict(
        input_dim=2,
        output_dim=1,
        n_layers=2,
        n_units=32,
        activation_fn=nn.Tanh()
    )

    sampler = AmericanCallOptionSampler(
        n_points=10,
        n_dim=2,
        t_start=0,
        t_end=1,
        domain=[0, 2]
    )

    model = DGMModel(
        net_params=net_params,
        sampler=sampler
    )

    loss = model.loss_fn(
        sampler.sample_domain()
    )
