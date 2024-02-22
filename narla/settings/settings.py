from __future__ import annotations

import os
import dataclasses

import tyro
import prettyprinter

import narla
from narla.settings.base_settings import BaseSettings
from narla.settings.trial_settings import TrialSettings
from narla.environments.environment_settings import EnvironmentSettings
from narla.multi_agent_network.multi_agent_network_settings import (
    MultiAgentNetworkSettings,
)


@dataclasses.dataclass
class Settings(BaseSettings):
    environment_settings: EnvironmentSettings = dataclasses.field(default_factory=lambda: EnvironmentSettings())

    multi_agent_network_settings: MultiAgentNetworkSettings = dataclasses.field(default_factory=lambda: MultiAgentNetworkSettings())

    trial_settings: TrialSettings = dataclasses.field(default_factory=lambda: TrialSettings())


def parse_args() -> Settings:
    """
    Parse the command line arguments for the Settings
    """
    narla.experiment_settings = tyro.cli(Settings)
    prettyprinter.pprint(narla.experiment_settings.as_dictionary())
    print(flush=True)

    if narla.experiment_settings.trial_settings.results_directory:
        # Creating <results_directory>/<trial_id>/
        trial_path = narla.io.format_trial_path(narla.experiment_settings)
        narla.io.make_directories(trial_path)

        # Saving settings to <results_directory>/<trial_id>/settings.json
        settings_file = os.path.join(trial_path, "settings.yaml")
        narla.io.save_settings(file=settings_file, settings=narla.experiment_settings)

    if narla.experiment_settings.trial_settings.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(narla.experiment_settings.trial_settings.gpu)

    return narla.experiment_settings
