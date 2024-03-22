from __future__ import annotations

import torch

import narla
from narla.rewards.biological_reward import BiologicalReward


class ActivityTrace(BiologicalReward):
    """
    Penalize Neurons for becoming too active
    """

    def compute(self, network: narla.multi_agent_network.MultiAgentNetwork, layer_index: int) -> torch.Tensor:
        layer_rewards = []
        for neuron in network.layers[layer_index].neurons:
            actions = neuron.history.get("action", stack=True).squeeze().float()
            x = torch.abs(torch.mean(actions) - 0.5)
            reward = -1746463 + (0.9256812 - -1746463) / (1 + (x / 236.1357) ** 2.160334)

            layer_rewards.append(reward)

        return torch.tensor([layer_rewards])
