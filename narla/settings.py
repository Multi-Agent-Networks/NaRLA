import tap
from typing import Literal


class Settings(tap.Tap):
    batch_size: int = 128
    device: Literal["cpu", "cuda"] = "cuda"
    environment: Literal["cart-pole"] = "cart-pole"
    neuron_type: Literal["deep_q", "policy_gradient"] = "deep_q"
    number_of_layers: int = 2
    number_of_neurons: int = 15
    update_every: int = 1
    render: bool = False
    reward_type: Literal["all", "task", "bio", "bio_then_all"] = "all"
