from abc import ABC, abstractmethod


class AbstractPDE(ABC):

    def __init__(
            self,
            name: str,
            time_dimension: bool = False,
            space_dimension: int = 1
    ) -> None:
        self.name = name
        self.space_dimension = space_dimension
        if time_dimension:
            self.time_dimension = 1
        else:
            self.time_dimension = 0
        self.total_dimension = self.time_dimension + self.space_dimension

    def __repr__(self) -> str:
        return f"PDE of type{self.name}"

    def get_total_dimension(self) -> int:
        """
        Get the total dimension of the PDE
        :return: total dimension
        """
        return self.total_dimension

    def get_time_dimension(self) -> int:
        """
        Get the time dimension of the PDE
        :return: time dimension
        """
        return self.time_dimension

    def get_space_dimension(self) -> int:
        """
        Get the space dimension of the PDE
        :return: space dim
        """
        return self.space_dimension

    @abstractmethod
    def get_zeroth_order_differential_operator(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_first_order_differential_operator(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_second_order_differential_operator(self, *args, **kwargs):
        raise NotImplementedError




