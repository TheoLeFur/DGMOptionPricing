from trainer.base_trainer import BaseTrainer
from typing import Dict, List, Optional, Callable
import torch
import torch.nn as nn

from tqdm import tqdm

from models.dgm_model import DGMModel


class PDETrainer(BaseTrainer):

    def __init__(
            self,
            n_epochs: int,
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
            saving_period: Optional[int] = 1000,
            save_path: Optional[str] = None,
            learning_rate: Optional[float] = 3e-5,
            learning_rate_scheduler: Optional[str] = None,
            learning_rate_scheduler_params: Optional[Dict] = None,
            device: Optional = None
    ):
        super().__init__(n_epochs=n_epochs)

        self.model = DGMModel(
            input_dim=input_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            n_units=n_units,
            sampler_data=sampler_data,
            pde_data=pde_data,
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn,
            skip_connection=skip_connection,
            init_method=init_method,
            criterion=criterion,
            optimizer=optimizer,
            learning_rate=learning_rate,
            learning_rate_scheduler=learning_rate_scheduler,
            learning_rate_scheduler_params=learning_rate_scheduler_params,
            device=device
        )

        self.saving_period = saving_period
        self.save_path = save_path

    def run(self):
        for i in tqdm(range(self.n_epochs)):

            if i % self.saving_period == 0:
                torch.save(self.model.nn_model.state_dict(), self.save_path)
            data = self.model.step()

            def tensor_to_item(t: torch.Tensor):
                return t.item()

            print({key: tensor_to_item(value) for key, value in data.items()})
            self.model.adjust_learning_rate()
