from abc import ABC, abstractmethod


class AbstractModel(ABC):

    @abstractmethod
    def loss_fn(self, *args, **kwargs):
        raise NotImplementedError

    def step(self, *args, **kwargs):
        raise NotImplementedError
