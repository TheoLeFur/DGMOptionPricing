from abc import ABC, abstractmethod
from typing import Dict


class BaseTrainer(ABC):

    def __init__(
            self,
            n_epochs: int,
    ):
        self.n_epochs = n_epochs

    @abstractmethod
    def run(self):
        raise NotImplementedError
