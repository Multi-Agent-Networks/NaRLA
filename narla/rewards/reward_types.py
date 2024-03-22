from __future__ import annotations

import enum
from typing import List

import narla


class RewardTypes(str, enum.Enum):
    ACTIVE_NEURONS = "active_neurons"
    ACTIVITY_TRACE = "activity_trace"
    LAYER_SPARSITY = "layer_sparsity"
    PREDICTION = "prediction"
    TASK_REWARD = "task_reward"

    @staticmethod
    def biological_reward_types() -> List[RewardTypes]:
        """
        Get the RewardTypes that are biological
        """
        return [RewardTypes.ACTIVE_NEURONS, RewardTypes.LAYER_SPARSITY, RewardTypes.PREDICTION, RewardTypes.ACTIVITY_TRACE]

    def to_reward(self) -> narla.rewards.Reward:
        """
        Convert the RewardType to a Reward object
        """
        if self == RewardTypes.ACTIVE_NEURONS:
            return narla.rewards.ActiveNeurons()
        elif self == RewardTypes.LAYER_SPARSITY:
            return narla.rewards.LayerSparsity()
        elif self == RewardTypes.PREDICTION:
            return narla.rewards.Prediction()
        elif self == RewardTypes.ACTIVITY_TRACE:
            return narla.rewards.ActivityTrace()

        raise NotImplementedError(f"Reward not implemented for type: {self}")
