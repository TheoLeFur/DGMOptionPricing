from typing import List, Dict


class Logger:

    def __init__(self):
        self.history: Dict = {}
        self.history["loss"]: List = []

    def write_loss(self, value):
        self.history["loss"].append(value)
