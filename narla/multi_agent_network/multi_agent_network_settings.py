import dataclasses
from typing import List

from narla.settings.base_settings import BaseSettings
from narla.rewards.reward_types import RewardTypes
from narla.multi_agent_network.layer_settings import LayerSettings


@dataclasses.dataclass
class MultiAgentNetworkSettings(BaseSettings):
    layer_settings: LayerSettings = dataclasses.field(default_factory=lambda: LayerSettings())
    """Settings for the Layers in the MultiAgentNetwork"""

    reward_types: List[RewardTypes] = dataclasses.field(default_factory=lambda: [RewardTypes.TASK_REWARD])
    """Reward types to be used by the Neurons for learning"""

    number_of_layers: int = 2
    """Total number of layers to use in the network"""
