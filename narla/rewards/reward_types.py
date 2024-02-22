from __future__ import annotations

import enum
from typing import List

import narla


class RewardTypes(str, enum.Enum):
    ACTIVE_NEURONS = "active_neurons"
    LAYER_SPARSITY = "layer_sparsity"
    TASK_REWARD = "task_reward"

    @property
    def biological_reward_types(self) -> List[RewardTypes]:
        """
        Get the RewardTypes that are biological
        """
        return [RewardTypes.ACTIVE_NEURONS, RewardTypes.LAYER_SPARSITY]

    def to_reward(self) -> narla.rewards.Reward:
        """
        Convert the RewardType to a Reward object
        """
        if self == RewardTypes.ACTIVE_NEURONS:
            return narla.rewards.ActiveNeurons()
        elif self == RewardTypes.LAYER_SPARSITY:
            return narla.rewards.LayerSparsity()

        raise NotImplementedError(f"Reward not implemented for type: {self}")
