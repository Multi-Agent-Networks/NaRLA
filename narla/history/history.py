import torch
import random
from typing import Dict, List


class History:
    def __init__(self):
        self._history: Dict[str, List[torch.Tensor | float]] = {}

    def get(self, key: str) -> list:
        """
        Access an element from the History

        :param key: Name of element to Access
        """
        return self._history.get(key)

    def record(self, **kwargs):
        """
        Store all the keyword arguments

        :param kwargs: If the key word doesn't yet exist an internal list will be created and the value appended to it
        """
        for key, value in kwargs.items():
            if key not in self._history:
                self._history[key] = []

            self._history[key].append(value)

    def sample(self, names: List[str], sample_size: int, from_most_recent: int = 10_000) -> List[List[torch.Tensor]]:
        sample = []
        for name in names:
            items = self.get(name)[-from_most_recent:]
            sample.append(
                random.Random(123).sample(items, sample_size)
            )

        return sample

    def __len__(self) -> int:
        minimum_length = 1e10
        for history_record in self._history.values():
            minimum_length = min(minimum_length, len(history_record))

        return minimum_length
