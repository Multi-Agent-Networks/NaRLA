from __future__ import annotations

import os
import itertools
import dataclasses
from typing import List

import tyro
import prettyprinter

import narla
from narla.settings.settings import Settings
from narla.neurons.neuron_types import NeuronTypes
from narla.rewards.reward_types import RewardTypes
from narla.settings.base_settings import BaseSettings
from narla.environments.available_environments import GymEnvironments


@dataclasses.dataclass
class RunnerSettings(BaseSettings):
    settings: Settings = dataclasses.field(default_factory=lambda: Settings())

    environments: List[GymEnvironments] = (GymEnvironments.CART_POLE,)
    """Environment to train on"""

    gpus: List[int] = (0,)
    """GPU ID to run on"""

    jobs_per_gpu: int = 2
    """Number of Jobs to put on each GPU"""

    learning_rates: List[float] = (1e-4,)
    """Learning rates for the individual neuron networks"""

    local_connectivity: List[bool] = (False, True)
    """If True Network will use local connectivity"""

    neuron_types: List[NeuronTypes] = (NeuronTypes.POLICY_GRADIENT, NeuronTypes.DEEP_Q, NeuronTypes.ACTOR_CRITIC)
    """What to of neuron to use in the network"""

    number_of_layers: List[int] = range(2, 10)
    """Total number of layers to use in the network"""

    number_of_neurons_per_layer: List[int] = (15,)
    """Number of neurons per layer in the network (the last layer always has only one neuron)"""

    reward_types: list = dataclasses.field(
        default_factory=lambda: [
            [RewardTypes.TASK_REWARD],
            [RewardTypes.TASK_REWARD, *RewardTypes.biological_reward_types()],
        ],
    )
    """Type of reward to use during training"""

    def product(self) -> List[narla.settings.Settings]:
        """
        Create the product of all Settings based on the RunnerSettings values
        """
        all_settings = []
        meta_settings = itertools.product(
            self.environments,
            self.learning_rates,
            self.local_connectivity,
            self.neuron_types,
            self.number_of_layers,
            self.number_of_neurons_per_layer,
            self.reward_types,
        )

        for trial_id, meta_setting in enumerate(meta_settings):
            self.settings.trial_settings.trial_id = trial_id
            settings = self.settings.clone()

            settings.environment_settings.environment = meta_setting[0]
            settings.multi_agent_network_settings.layer_settings.neuron_settings.learning_rate = meta_setting[1]
            settings.multi_agent_network_settings.local_connectivity = meta_setting[2]
            settings.multi_agent_network_settings.layer_settings.neuron_settings.neuron_type = meta_setting[3]
            settings.multi_agent_network_settings.number_of_layers = meta_setting[4]
            settings.multi_agent_network_settings.layer_settings.number_of_neurons_per_layer = meta_setting[5]
            settings.multi_agent_network_settings.reward_types = meta_setting[6]

            all_settings.append(settings)

        return all_settings


def parse_args() -> RunnerSettings:
    """
    Parse the command line arguments for the RunnerSettings
    """
    runner_settings = tyro.cli(RunnerSettings)
    prettyprinter.pprint(runner_settings.as_dictionary())
    print(flush=True)

    if runner_settings.settings.trial_settings.results_directory:
        narla.io.make_directories(runner_settings.settings.trial_settings.results_directory)

        # Saving settings to <results_directory>/settings.yaml
        settings_file = os.path.join(runner_settings.settings.trial_settings.results_directory, "settings.yaml")
        narla.io.save_settings(file=settings_file, settings=runner_settings)

    return runner_settings
