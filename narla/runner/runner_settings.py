from __future__ import annotations

import os
import enum
import tyro
import narla
import itertools
import dataclasses
import prettyprinter
from typing import List
from narla import Settings
from narla.history import reward_types
from narla.neurons import AvailableNeurons
from narla.environments import GymEnvironments


@dataclasses.dataclass
class RunnerSettings:
    settings: narla.Settings = Settings()

    environments: List[GymEnvironments] = (GymEnvironments.CART_POLE,)
    """Environment to train on"""

    gpus: List[int] = (0,)
    """GPU ID to run on"""

    jobs_per_gpu: int = 2
    """Number of Jobs to put on each GPU"""

    learning_rates: List[float] = (1e-4,)
    """Learning rates for the individual neuron networks"""

    neuron_types: List[AvailableNeurons] = (AvailableNeurons.DEEP_Q, AvailableNeurons.ACTOR_CRITIC)
    """What to of neuron to use in the network"""

    number_of_layers: List[int] = range(1, 10)
    """Total number of layers to use in the network"""

    number_of_neurons_per_layer: List[int] = (15,)
    """Number of neurons per layer in the network (the last layer always has only one neuron)"""

    reward_types: List[str] = (reward_types.TASK_REWARD,)
    """Type of reward to use during training"""

    def as_dictionary(self) -> dict:
        """
        Convert the RunnerSettings to a dictionary
        """
        return dataclasses.asdict(self)

    def product(self) -> List[narla.Settings]:
        """
        Create the product of all Settings based on the RunnerSettings values
        """
        all_settings = []
        meta_settings = itertools.product(
            self.environments,
            self.neuron_types,
            self.number_of_layers,
            self.number_of_neurons_per_layer,
            self.reward_types
        )

        for trial_id, meta_setting in enumerate(meta_settings):
            self.settings.trial_id = trial_id
            settings = narla.Settings(**self.settings.as_dictionary())

            settings.environment = meta_setting[0]
            settings.neuron_type = meta_setting[1]
            settings.number_of_layers = meta_setting[2]
            settings.number_of_neurons_per_layer = meta_setting[3]
            settings.reward_type = meta_setting[4]

            all_settings.append(settings)

        return all_settings

    def to_command_string(self):
        """
        Converts RunnerSettings to a command line string
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

            arguments.append(f"--{key} {value}")

        return " ".join(arguments)


def parse_args() -> RunnerSettings:
    """
    Parse the command line arguments for the RunnerSettings
    """
    runner_settings = tyro.cli(RunnerSettings)
    prettyprinter.pprint(runner_settings.as_dictionary())
    print(flush=True)

    if runner_settings.settings.results_directory:
        narla.io.make_directories(runner_settings.settings.results_directory)

        # Saving settings to <results_directory>/settings.yaml
        settings_file = os.path.join(runner_settings.settings.results_directory, "settings.yaml")
        narla.io.save_settings(
            file=settings_file,
            settings=runner_settings
        )

    return runner_settings
