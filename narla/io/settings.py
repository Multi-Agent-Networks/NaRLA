from __future__ import annotations

import os
import narla


def format_trial_path(settings: narla.Settings) -> str:
    path = os.path.join(settings.results_directory, "%05d" % settings.trial_id)
    return path


def load_runner_settings(file: str) -> narla.runner.Settings:
    settings = narla.runner.Settings()
    dictionary = narla.io.load_yaml(file)
    settings.__dict__.update(dictionary)

    return settings


def load_settings(file: str) -> narla.Settings:
    settings = narla.Settings()
    dictionary = narla.io.load_yaml(file)
    settings.__dict__.update(dictionary)

    return settings


def save_settings(file: str, settings: narla.Settings):
    narla.io.save_yaml(
        yaml_file=file,
        dictionary=settings.as_dictionary()
    )
