import dataclasses

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import narla
from narla.settings.base_settings import BaseSettings


@dataclasses.dataclass
class TrialSettings(BaseSettings):
    batch_size: int = 128
    """Batch size to use during training"""

    device: Literal["cpu", "cuda"] = "cpu"
    """Device to put the network on"""

    gpu: int = 0
    """GPU ID to run on"""

    maximum_episodes: int = 10_000
    """Total number of episodes to run for"""

    random_seed: int = 0
    """Random seed"""

    results_directory: str = ""
    """Path to save results"""

    save_every: int = 1_000
    """Save results every n steps"""

    trial_id: int = 0
    """Unique ID of the trial being run, corresponds to the path of data saving <results_directory>/<trial_id>/"""

    def __post_init__(self):
        narla.settings.reproducibility(self.random_seed)
