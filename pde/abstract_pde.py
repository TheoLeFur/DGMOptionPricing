from abc import ABC


class AbstractPDE(ABC):

    def __init__(
            self,
            name: str
    ) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"PDE of type{self.name}"
