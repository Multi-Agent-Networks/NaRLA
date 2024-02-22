from __future__ import annotations

import os
import enum
import tyro
import narla
import dataclasses
import prettyprinter
try:
    from typing import Literal
except:
    from typing_extensions import Literal
from narla.history import reward_types
from narla.neurons import ALL_NEURONS, AvailableNeurons
from narla.environments import ALL_ENVIRONMENTS, GymEnvironments


@dataclasses.dataclass
class Settings:
    batch_size: int = 128
    """Batch size to use during training"""

    device: Literal["cpu", "cuda"] = "cuda"
    """Device to put the network on"""

    environment: ALL_ENVIRONMENTS = GymEnvironments.CART_POLE
    """Environment to train on"""

    gpu: int = 0
    """GPU ID to run on"""

    learning_rate: float = 1e-4
    """Learning rate for the individual neuron networks"""

    maximum_episodes: int = 10_000
    """Total number of episodes to run for"""

    neuron_type: ALL_NEURONS = AvailableNeurons.DEEP_Q
    """What to of neuron to use in the network"""

    number_of_layers: int = 2
    """Total number of layers to use in the network"""

    number_of_neurons_per_layer: int = 15
    """Number of neurons per layer in the network (the last layer always has only one neuron)"""

    render: bool = False
    """If true will visualize the environment"""

    results_directory: str = ""
    """Path to save results"""

    reward_type: Literal[reward_types.TASK_REWARD] = reward_types.TASK_REWARD
    """Type of reward to use during training"""

    save_every: int = 1_000
    """Save results every n steps"""

    trial_id: int = 0
    """Unique ID of the trial being run, corresponds to the path of data saving <results_directory>/<trial_id>/"""

    def as_dictionary(self) -> dict:
        """
        Convert the Settings to a dictionary
        """
        return dataclasses.asdict(self)

    def to_command_string(self, prefix: str = ""):
        """
        Converts Settings to a command line string
        """
        arguments = []
        for key, value in self.as_dictionary().items():
            if isinstance(value, enum.Enum):
                value = value.name

            if type(value) is bool:
                if value is False:
                    continue
                else:
                    value = ""

            arguments.append(f"--{prefix}{key} {value}")

        return " ".join(arguments)


def parse_args() -> Settings:
    """
    Parse the command line arguments for the Settings
    """
    narla.settings = tyro.cli(Settings)
    prettyprinter.pprint(narla.settings.as_dictionary())
    print(flush=True)

    if narla.settings.results_directory:
        # Creating <results_directory>/<trial_id>/
        trial_path = narla.io.format_trial_path(narla.settings)
        narla.io.make_directories(trial_path)

        # Saving settings to <results_directory>/<trial_id>/settings.json
        settings_file = os.path.join(trial_path, "settings.yaml")
        narla.io.save_settings(
            file=settings_file,
            settings=narla.settings
        )

    if narla.settings.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(narla.settings.gpu)

    return narla.settings
