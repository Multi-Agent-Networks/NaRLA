import dataclasses

from narla.environments import ALL_ENVIRONMENTS, GymEnvironments
from narla.settings.base_settings import BaseSettings


@dataclasses.dataclass
class EnvironmentSettings(BaseSettings):
    environment: ALL_ENVIRONMENTS = GymEnvironments.CART_POLE
    """Environment to train on"""

    render: bool = False
    """If true will visualize the environment"""
