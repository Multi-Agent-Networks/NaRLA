from __future__ import annotations

import dataclasses

import narla
from narla.neurons.neuron_types import AvailableNeurons
from narla.settings.base_settings import BaseSettings


@dataclasses.dataclass
class NeuronSettings(BaseSettings):
    learning_rate: float = 1e-4
    """Learning rate for Neurons"""

    neuron_type: AvailableNeurons = AvailableNeurons.POLICY_GRADIENT
    """Type of Neuron that will be used"""

    def create_neuron(self, observation_size: int, number_of_actions: int) -> narla.neurons.Neuron:
        NeuronType = self.neuron_type.to_neuron_type()

        return NeuronType(
            observation_size=observation_size,
            number_of_actions=number_of_actions,
            learning_rate=self.learning_rate,
        )
