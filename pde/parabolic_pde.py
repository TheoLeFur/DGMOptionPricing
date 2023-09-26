import numpy as np

from pde.abstract_pde import AbstractPDE


class ParabolicPDE(AbstractPDE):

    def __init__(
            self,
            drift: np.ndarray,
            diffusion_matrix: np.ndarray,
            name: str,
            time_dimension: bool = False,
            space_dimension: int = 1,
    ):
        super().__init__(
            name=name,
            time_dimension=time_dimension,
            space_dimension=space_dimension,
        )

        if np.shape(drift) == (self.space_dimension, 1):
            self.drift = drift
        else:
            raise ValueError("Drift vector is not consistent with specified dimension")

        if np.shape(diffusion_matrix) == (self.space_dimension, self.space_dimension):
            self.diffusion_matrix = diffusion_matrix
        else:
            raise ValueError("Diffusion matrix is not consistent with specified dimension")

    def get_drift(self) -> np.ndarray:
        return self.drift

    def get_diffusion(self) -> np.ndarray:
        return self.diffusion_matrix
