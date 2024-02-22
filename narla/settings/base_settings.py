from __future__ import annotations

import copy
import enum
import dataclasses
from typing import Any, Dict, List, Type, TypeVar

import tyro
import caml_core as core
from dacite import from_dict
from typing_extensions import Self

AnySettings = TypeVar("AnySettings", bound="Settings")


@dataclasses.dataclass
class BaseSettings:
    """
    This is the Settings baseclass.

    This can be used to create dataclasses for command line arguments.
    See `Tyro <https://github.com/brentyi/tyro>`_ for more information
    """

    def as_dictionary(self) -> Dict[str, Any]:
        """Convert the RunnerSettings to a dictionary.

        .. note::
            This function is more backwards compatible than ``dataclasses.asdict(self)``

        :returns: Settings object as a Dictionary
        """
        dictionary: Dict[str, Any] = {}
        for field in dataclasses.fields(self):
            # This check ensures that the dataclasses' fields are also attributes of the object. When an object has been
            # serialized and the original class adds an attribute this would cause issues after deserialization.
            if not hasattr(self, field.name):
                continue

            value = getattr(self, field.name)
            if isinstance(value, BaseSettings):
                value = value.as_dictionary()

            dictionary[field.name] = value

        return dictionary

    def clone(self) -> Self:
        """
        Create a deepcopy of the Settings
        """
        return copy.deepcopy(self)

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> BaseSettings:
        """Create the Settings object from the dictionary.

        :param dictionary: Dictionary to parse into the Settings
        :returns: Settings object
        """
        return from_dict(cls, dictionary)

    @classmethod
    def from_yaml(cls: Type[AnySettings], yaml_file: str = "") -> AnySettings:
        """Load the Settings from a yaml file.

        :param yaml_file: Path to a yaml file containing serialized Settings object (or dict)
        :raises TypeError: If the loaded type does not match the class type
        :returns: Settings
        """
        settings = core.io.load_yaml(yaml_file)
        # Treat dict objects as Settings objects
        if isinstance(settings, dict):
            settings = from_dict(data_class=cls, data=settings)

        if not isinstance(settings, cls):
            raise TypeError(f"Loaded Type {type(settings)} != {cls} for Settings retrieved from {yaml_file}!")

        return settings

    @classmethod
    def parse_args(cls: Type[AnySettings]) -> AnySettings:
        """
        Parse the command line arguments for the Settings.

        :return: Settings object
        """
        settings, _ = tyro.cli(cls, return_unknown_args=True)
        if hasattr(settings, "yaml_file") and getattr(settings, "yaml_file"):
            return cls.from_yaml(getattr(settings, "yaml_file"))

        return settings

    def to_command_string(self, prefix: str = ""):
        """
        Converts Settings to a command line string
        """
        arguments = []
        for field in dataclasses.fields(self):
            if not hasattr(self, field.name):
                continue

            value = getattr(self, field.name)
            if isinstance(value, BaseSettings):
                sub_string = value.to_command_string(prefix=f"{prefix}{field.name}.")
                arguments.append(sub_string)
                continue

            if isinstance(value, list):
                # Handle special cast of List[Enum]
                # NOTE: Won't work for List[BaseSettings]
                value = " ".join([item.name if isinstance(item, enum.Enum) else item for item in value])

            if isinstance(value, enum.Enum):
                value = value.name

            elif type(value) is bool:
                if value is False:
                    continue
                else:
                    value = ""

            arguments.append(f"--{prefix}{field.name} {value}")

        return " ".join(arguments)
