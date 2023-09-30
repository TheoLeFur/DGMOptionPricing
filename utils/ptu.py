import numpy as np
import torch
from typing import Optional
import torch


def from_numpy(array: np.ndarray, device: Optional = None, requires_grad: Optional = True) -> torch.Tensor:
    """
    Convert from np.ndarray to torch tensor, converts to standard float32.
    :param requires_grad: If true, we require the gradients from the tensor.
    :param array: Numpy array
    :param device: optional device we load our tensor onto
    :return: Equivalent tensor
    """
    output: torch.Tensor = torch.from_numpy(array)
    output = output.type(torch.float32)
    if device is not None:
        output.to(device)
    if requires_grad:
        output.requires_grad_()
    return output


def sample_multi_dimensional_bm(
        t: float,
        diffusion_matrix: torch.tensor,
):
    """
    Sample from a multivariate norma distribution
    :param t: Time parameter
    :param diffusion_matrix: Diffusion matrix of the parabolic PDE
    :return: Samples from distribution
    """
    mean = torch.zeros(diffusion_matrix.shape[0])
    multivariate_normal: torch.distributions.Distribution = torch.distributions.multivariate_normal.MultivariateNormal(
        mean, t * diffusion_matrix)
    return multivariate_normal.sample()
