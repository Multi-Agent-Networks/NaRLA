from __future__ import annotations

import tap
from typing import Literal
from narla.neurons import neuron_types
from narla.history import reward_types


class Settings(tap.Tap):
    batch_size: int = 128
    """Batch size to use during training"""

    device: Literal["cpu", "cuda"] = "cuda"
    """Device to put the network on"""

    environment: Literal["cart-pole"] = "cart-pole"
    """Environment to train on"""

    neuron_type: Literal[neuron_types.DEEP_Q, neuron_types.ACTOR_CRITIC] = neuron_types.DEEP_Q
    """What to of neuron to use in the network"""

    number_of_episodes: int = 10_000
    """Total number of episodes to run for"""

    number_of_layers: int = 2
    """Total number of layers to use in the network"""

    number_of_neurons_per_layer: int = 15
    """Number of neurons per layer in the network (the last layer always has only one neuron)"""

    render: bool = False
    """If true will visualize the environment"""

    reward_type: Literal[reward_types.TASK_REWARD] = reward_types.TASK_REWARD
    """Type of reward to use during training"""
