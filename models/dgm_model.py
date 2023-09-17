from models.abstract_model import AbstractModel
from nn.dgm_net import DGMNet
import torch
import torch.nn as nn
from sampler.abstract_sampler import AbstractSampler
from typing import Callable, Dict


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
            self.optimizer = torch.optim.adam
        else:
            self.optimizer = optimizer

        if device is None:
            self.device = "cpu"
        else:
            self.device = device


    def loss_fn(self) -> torch.Tensor:
        raise NotImplementedError

    def step(self):

        """
        Take a training step
        :return: Value of the loss function at current iteration
        """

        self.optimizer.zero_grad()
        loss: torch.Tensor = self.loss_fn()
        loss.backward()
        self.optimizer.step()

        return loss.item()
