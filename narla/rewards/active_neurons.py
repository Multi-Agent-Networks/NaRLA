from __future__ import annotations

import narla
import torch
from narla.rewards import Reward


class ActiveNeurons(Reward):
    """
    Rewards Neurons for becoming active
    """

    def compute(
        self,
        current_layer: narla.multi_agent_network.Layer,
        next_layer: narla.multi_agent_network.Layer,
    ) -> torch.Tensor:

        return current_layer.layer_output
