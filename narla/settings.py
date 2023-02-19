from __future__ import annotations

import tap
import narla
from typing import Literal
from narla.neurons import neuron_types
from narla.history import reward_types


class Settings(tap.Tap):
    batch_size: int = 128
    device: Literal["cpu", "cuda"] = "cuda"
    environment: Literal["cart-pole"] = "cart-pole"
    neuron_type: Literal[neuron_types.DEEP_Q, neuron_types.ACTOR_CRITIC] = neuron_types.DEEP_Q
    number_of_layers: int = 2
    number_of_neurons_per_layer: int = 15
    update_every: int = 1
    render: bool = False
    reward_type: Literal[reward_types.TASK_REWARD] = reward_types.TASK_REWARD
