from typing import Optional, Callable

import torch
import torch.nn as nn
from typing import Optional


class DGMNet(nn.Module):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            n_layers: int,
            n_units: int,
            activation_fn: Callable,
            output_activation_fn: Optional[Callable] = None,
            skip_connection: Optional[bool] = False,
            init_method: Callable = nn.init.xavier_uniform,
            n_linear_layers: int = 9) -> None:
        """
        This module builds the DGM network architecture.

        :param input_dim: Input dimension (Number of stocks + time dimension)
        :param output_dim: Output Dimensions
        :param n_layers: Number of layers
        :param n_units: Number of units
        :param activation_fn: Middle layer activation function
        :param output_activation_fn: Output layer activation function
        :param skip_connection: Skip
        :param init_method:
        :param n_linear_layers:
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.n_layers = n_layers
        self.n_units = n_units
        self.activation_fn = activation_fn
        if output_activation_fn is not None:
            self.output_activation_fn = output_activation_fn
        else:
            self.output_activation_fn = nn.Identity()

        self.skip_connection = skip_connection

        self.n_linear_layers = n_linear_layers
        self.init_method = init_method

        # Now we define the network architecture
        self.Sw = nn.Linear(self.input_dim, self.n_units)

        self.Uz = nn.Linear(self.input_dim, self.n_units)
        self.Wz = nn.Linear(self.n_units, self.n_units)

        self.Ug = nn.Linear(self.input_dim, self.n_units)
        self.Wg = nn.Linear(self.n_units, self.n_units)

        self.Ur = nn.Linear(self.input_dim, self.n_units)
        self.Wr = nn.Linear(self.n_units, self.n_units)

        self.Uh = nn.Linear(self.input_dim, self.n_units)
        self.Wh = nn.Linear(self.n_units, self.n_units)

        self.output_layer = nn.Linear(self.n_units, self.output_dim)

    @property
    def get_state(self):
        return self.state_dict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network
        :param x: Input tensor
        :return: Output
        """
        output: torch.Tensor
        S1: torch.Tensor = self.Sw(x)
        for i in range(self.n_layers):
            S: torch.Tensor = S1 if i == 0 else self.activation_fn(output)

            Z: torch.Tensor = self.activation_fn(self.Uz(x) + self.Wz(S))
            G: torch.Tensor = self.activation_fn(self.Ug(x) + self.Wg(S1))
            R: torch.Tensor = self.activation_fn(self.Ur(x) + self.Wg(S))
            H: torch.Tensor = self.activation_fn(self.Uh(x) + self.Wg(torch.mul(S, R)))

            output: torch.Tensor = torch.mul(1 - G, H) + torch.mul(Z, S)

        out: torch.Tensor = self.output_activation_fn(self.output_layer(output))
        return out
s