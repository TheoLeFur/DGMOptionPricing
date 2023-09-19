import numpy as np
import torch
from typing import Optional


def from_numpy(array: np.ndarray, device: Optional = None, requires_grad: Optional = True) -> torch.Tensor:
    """
    Convert from np.ndarray to torch tensor, converts to standard float32.
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
