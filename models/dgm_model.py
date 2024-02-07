import torch.optim

from models.abstract_model import AbstractModel
from typing import Callable, Optional, Dict, List
import torch.nn as nn

from torch.optim.lr_scheduler import ExponentialLR

from sampler.american_put_option_sampler import AmericanPutOptionSampler
from sampler.abstract_sampler import Sample

from nn.dgm_net import DGMNet
from pde.FreeBoundaryPDE import FreeBoundaryPDE
from pde.utils.payoffs.base_payoff import BasePayoff


class DGMModel(AbstractModel):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            n_layers: int,
            n_units: int,
            sampler_data: Dict,
            pde_data: Dict,
            activation_fn: Callable = None,
            output_activation_fn: Callable = None,
            skip_connection: Optional[bool] = False,
            init_method: Callable = nn.init.xavier_uniform,
            criterion=None,
            optimizer=None,
            learning_rate: Optional[float] = 3e-5,
            learning_rate_scheduler: Optional[str] = None,
            learning_rate_scheduler_params: Optional[Dict] = None,
            device: Optional = None
    ) -> None:

        self.dgm_net = DGMNet(
            input_dim=input_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            n_units=n_units,
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn,
            skip_connection=skip_connection,
            init_method=init_method
        )
        if device is None:
            self.device = "cpu"
        else:
            self.device = device

        self.dgm_net.to(self.device)

        self.sampler = AmericanPutOptionSampler(**sampler_data, device=self.device)
        self.pde = FreeBoundaryPDE(**pde_data, device=self.device)
        self.payoff = self.pde.payoff

        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion

        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.dgm_net.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer

        if learning_rate_scheduler == "ema":
            self.learning_rate_scheduler = ExponentialLR(self.optimizer, learning_rate_scheduler_params["gamma"])
        else:
            raise NotImplementedError

    @property
    def nn_model(self):
        return self.dgm_net

    def loss_fn(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """

        :param args:
        :param kwargs:
        :return:
        """

        sample: Sample = self.sampler.sample_all()
        loss_data: Dict = {"domain_loss": self.domain_loss(sample.domain_sample),
                           "time_boundary_loss": self.time_boundary_loss(sample.initial_sample),
                           "space_boundary_loss": self.space_boundary_loss(sample.boundary_sample)}

        loss_data["total_loss"] = loss_data["domain_loss"] + loss_data["time_boundary_loss"] + loss_data[
            "space_boundary_loss"]
        return loss_data

    def step(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """

        :param args:
        :param kwargs:
        :return:
        """
        self.optimizer.zero_grad()
        loss_data: Dict[str, torch.Tensor] = self.loss_fn(*args, **kwargs)
        loss_data["total_loss"].backward()
        self.optimizer.step()

        return loss_data

    def log(self, data: Dict) -> None:
        """

        :param data:
        :return:
        """
        raise NotImplementedError

    def adjust_learning_rate(self) -> None:
        """
        Call if one wants to update the learning rate schedule
        :return: None
        """
        self.learning_rate_scheduler.step()

    def domain_loss(self, domain_sample: torch.Tensor) -> torch.Tensor:
        """

        :param domain_sample:
        :return:
        """
        op_action: torch.Tensor = self.pde.differential_operator(model=self.dgm_net)(domain_sample)

        domain_loss = self.criterion(op_action, torch.zeros_like(op_action, device=self.device))

        return domain_loss

    def time_boundary_loss(self, time_boundary_sample, **kwargs) -> torch.Tensor:
        """

        :param time_boundary_sample:
        :param kwargs:
        :return:
        """

        model_value = self.pde.time_boundary(self.dgm_net, **kwargs)(time_boundary_sample)
        time_target_fn: Callable[[torch.Tensor, torch.float64, torch.bool, str], torch.Tensor] = self.payoff.payoff
        time_target: torch.Tensor = time_target_fn(time_boundary_sample[:, 1:],
                                                   self.pde.option_data.strike_price,
                                                   device=self.device)

        time_boundary_loss = self.criterion(time_target, model_value)

        return time_boundary_loss

    def space_boundary_loss(self, space_boundary_sample, **kwargs) -> torch.Tensor:
        """

        :param space_boundary_sample:
        :param kwargs:
        :return:
        """

        space_range: List = self.sampler.space_range

        lower, upper = space_boundary_sample
        lower_value, upper_value = self.dgm_net(lower), self.dgm_net(upper)

        lower_target = torch.ones_like(lower_value) * space_range[0]
        upper_target = torch.ones_like(upper_value) * space_range[-1]

        [lower_target, upper_target] = map(lambda t: t.to(self.device), [lower_target, upper_target])

        return self.criterion(lower_value, lower_target) + self.criterion(upper_value, upper_target)
