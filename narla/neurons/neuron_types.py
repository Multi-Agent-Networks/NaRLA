from __future__ import annotations

import enum
from typing import Type

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import narla


class NeuronTypes(enum.Enum):
    ACTOR_CRITIC = "actor_critic"
    DEEP_Q = "deep_q"
    POLICY_GRADIENT = "policy_gradient"

    def to_neuron_type(self) -> Type[narla.neurons.Neuron]:
        """
        Convert Neuron Enum to class Type
        """
        if self == NeuronTypes.ACTOR_CRITIC:
            return narla.neurons.actor_critic.Neuron
        elif self == NeuronTypes.DEEP_Q:
            return narla.neurons.deep_q.Neuron
        elif self == NeuronTypes.POLICY_GRADIENT:
            return narla.neurons.policy_gradient.Neuron

    def __str__(self):
        return self.value


ALL_NEURONS = Literal[
    NeuronTypes.ACTOR_CRITIC,
    NeuronTypes.DEEP_Q,
    NeuronTypes.POLICY_GRADIENT,
]
