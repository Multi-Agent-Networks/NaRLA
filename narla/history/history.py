import torch
import narla
import random
from typing import Dict, List


class History:
    def __init__(self):
        self._history: Dict[str, List[torch.Tensor | float]] = {}

    def clear(self):
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
                if key == narla.history.saved_data.NEXT_OBSERVATION:
                    self._history[key] = [None]

            elif key == narla.history.saved_data.OBSERVATION:
                self.record(**{
                    narla.history.saved_data.NEXT_OBSERVATION: value
                })

            if key == narla.history.saved_data.NEXT_OBSERVATION:
                self._history[narla.history.saved_data.NEXT_OBSERVATION][-1] = value
                self._history[narla.history.saved_data.NEXT_OBSERVATION].append(None)

            else:
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
