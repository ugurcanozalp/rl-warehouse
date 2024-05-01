
import os
from typing import List, Tuple, Union, Dict, Any
import jsonlines


class Logger:
    """Generic logger class for RL experimenting. 
    """
    def __init__(self, path, only_bypass: bool = True):
        self._path = path
        self._only_bypass = only_bypass
        if not os.path.isdir(path):
            os.makedirs(path)
        
    def open(self):
        self._hparam_file = jsonlines.open(os.path.join(self._path, "hparams.jsonl"), "w", flush=True)
        self._info_file = jsonlines.open(os.path.join(self._path, "info.jsonl"), "w", flush=True)
        self._data_file = jsonlines.open(os.path.join(self._path, "data.jsonl"), "w", flush=True)

    def log(self, log_name: str, log_value: float, step: int = None):
        """Log a parameter during training

        Args:
            log_name (str): Name of the logged parameter
            log_value (float): Value of the logged parameter
            step (int): Step index
        """
        self._data_file.write({"step": step, log_name: log_value})

    def log_hparams(self, hparams: Dict[str, Union[bool, str, float, int]]):
        """Log a hyperparameters of the run

        Args:
            hparams (Dict): Hyperparameter dictionary
        """
        self._hparam_file.write(hparams)

    def log_text(self, description: str, text: str): 
        """Log a textual info

        Args:
            description (str): Text description
            text (str): Text to be logged
        """
        self._info_file.write({description: text})

    def close(self):
        self._hparam_file.close()
        self._info_file.close()
        self._data_file.close()