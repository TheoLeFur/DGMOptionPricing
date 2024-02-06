from typing import Callable

import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(
            self,
            n_layers: int,
            n_units: int,
            input_dim: int,
            output_dim: int,
            activation_fn: Callable = None,
            output_activation_fn: Callable = None,
            init_method: Callable = None
    ) -> None:
        """
        This represents an MLP module
        :param n_layers: Number of layers
        :param n_units: Number of units in each layer
        :param input_dim: Input dimension
        :param output_dim: Output dimension
        :param activation_fn: Activation function for the hidden layers
        :param output_activation_fn: Activation function for the output layer
        :param init_method: Initialisation method for layers
        """
        super().__init__()

        self.n_layers = n_layers
        self.n_units = n_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_fn = activation_fn
        self.output_activation_fn = output_activation_fn

        self.init_method = init_method

        layers = []

        size: int = self.input_dim
        for _ in range(self.n_layers - 1):
            layer = nn.Linear(self.input_dim, size)
            if self.init_method is not None:
                layer.apply(self.init_method)
            if self.activation_fn is not None:
                layers.append(self.activation_fn)
            size = self.n_units
        output_layer = nn.Linear(size, self.output_dim)
        if self.init_method is not None:
            output_layer.apply(self.init_method)
        layers.append(output_layer)
        if self.output_activation_fn is not None:
            layers.append(self.output_activation_fn)

        self.module = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.module(x)
