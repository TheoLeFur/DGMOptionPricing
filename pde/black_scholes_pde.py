from pde.parabolic_pde import ParabolicPDE
import numpy as np


class BlackScholesPDE(ParabolicPDE):

    def __init__(
            self,
            drift: np.ndarray,
            diffusion_matrix: np.ndarray,
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

        if rate < 0:
            raise ValueError("Negative rates not allowed")
        else:
            self.rate = rate




