from __future__ import annotations

import os
from typing import Union

import narla


def format_trial_path(settings: narla.settings.Settings) -> str:
    """
    Format the trial path ``settings.results_directory/%05d % settings.trial_id"

    :param settings: Settings object
    """
    path = os.path.join(settings.trial_settings.results_directory, "%05d" % settings.trial_settings.trial_id)
    return path


def load_runner_settings(file: str) -> narla.runner.RunnerSettings:
    """
    Load RunnerSettings from a file

    :param file: Path to the RunnerSettings yaml file
    """
    settings = narla.runner.RunnerSettings()
    dictionary = narla.io.load_yaml(file)
    settings.__dict__.update(dictionary)

    return settings


def load_settings(file: str) -> narla.settings.Settings:
    """
    Load Settings from a file

    :param file: Path to the Settings yaml file
    """
    settings = narla.settings.Settings()
    dictionary = narla.io.load_yaml(file)
    settings.__dict__.update(dictionary)

    return settings


def save_settings(file: str, settings: Union[narla.settings.Settings, narla.runner.RunnerSettings]):
    """
    Save settings to a file

    :param file: Path to the yaml file
    :param settings: Settings to be saved
    """
    narla.io.save_yaml(yaml_file=file, dictionary=settings.as_dictionary())
