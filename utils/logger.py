from typing import List, Dict
import pickle


class Logger:

    def __init__(
            self,
            logging_param_keys: Dict):
        self.logging_param_keys = logging_param_keys
        self.history: Dict = {}
        for key in logging_param_keys:
            self.history[key]: List = []

    def write_loss(self, values: Dict):
        """

        :param values:
        :return:
        """
        keys_included = all(key in self.logging_param_keys for key in values)
        if not keys_included:
            raise ValueError("Trying to log keys that are not specified in the logger.")
        for key in values.keys():
            self.history[key].append(values[key])

    @staticmethod
    def save_history(
            history: Dict,
            filename: str
    ) -> None:
        with open(filename, 'wb') as pickle_file:
            pickle.dump(history, pickle_file)

    @staticmethod
    def read_history(
            filename: str
    ) -> Dict:
        with open(filename, 'rb') as pickle_file:
            output = pickle.load(pickle_file)
        return output
