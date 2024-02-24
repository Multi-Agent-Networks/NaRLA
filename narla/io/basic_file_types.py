import os
import json

import yaml
import pandas as pd


def make_directories(path: str):
    """
    Create the directory if it doesn't exist

    :param path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def load_csv(csv_file: str) -> pd.DataFrame:
    """
    Load csv file

    :param csv_file: Path to the csv file
    """
    data_frame = pd.read_csv(csv_file)

    return data_frame


def load_yaml(yaml_file: str) -> dict:
    """
    Load a yaml file into a dictionary

    :param yaml_file: /path/to/yaml_file.yaml
    """
    with open(yaml_file, "r") as stream:
        return yaml.load(stream, Loader=yaml.Loader)


def save_json(json_file: str, dictionary: dict):
    """
    Save a dictionary into a json file
    :param json_file: /path/to/json_file.json
    :param dictionary: Dictionary
    """
    with open(json_file, "w") as stream:
        json.dump(dictionary, stream, indent=4, sort_keys=True)


def save_yaml(yaml_file: str, dictionary: dict):
    """
    Save a dictionary into a yaml file

    :param yaml_file: /path/to/yaml_file.yaml
    :param dictionary: Dictionary
    """
    with open(yaml_file, "w") as stream:
        yaml.dump(dictionary, stream, encoding="utf-8", allow_unicode=True)
