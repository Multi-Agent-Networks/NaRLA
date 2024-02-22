from __future__ import annotations

from typing import List

import torch

import narla


class Layer:
    """
    A Layer contains a list of Neurons

    :param observation_size: Size of the observation which the Layer will receive
    :param number_of_actions: Number of actions available to the Layer
    :param layer_settings: Settings for the Layer
    """

    def __init__(
        self,
        observation_size: int,
        number_of_actions: int,
        layer_settings: narla.multi_agent_network.LayerSettings,
    ):
        self.observation_size = observation_size
        self.number_of_actions = number_of_actions

        self._layer_output: torch.Tensor | None = None
        self._neurons: List[narla.neurons.Neuron] = self._build_layer(
            observation_size=observation_size,
            number_of_actions=number_of_actions,
            layer_settings=layer_settings,
        )

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Take an action based on the observation

        :param observation: Observation from the Layer's local environment
        """
        layer_output = torch.zeros((1, self.number_of_neurons), device=narla.experiment_settings.trial_settings.device)
        for index, neuron in enumerate(self._neurons):
            action = neuron.act(observation)
            layer_output[0, index] = action

        self._layer_output = layer_output

        return layer_output

    @staticmethod
    def _build_layer(
        observation_size: int,
        number_of_actions: int,
        layer_settings: narla.multi_agent_network.LayerSettings,
    ) -> List[narla.neurons.Neuron]:
        neurons: List[narla.neurons.Neuron] = []
        for _ in range(layer_settings.number_of_neurons_per_layer):
            neuron = layer_settings.neuron_settings.create_neuron(
                observation_size=observation_size,
                number_of_actions=number_of_actions,
            )
            neurons.append(neuron)

        return neurons

    def distribute_to_neurons(self, **kwargs):
        """
        Distribute data to the Neurons

        :param kwargs: Key word arguments to be distributed
        """
        for neuron in self._neurons:
            neuron.record(**kwargs)

    @property
    def layer_output(self) -> torch.Tensor:
        """
        Access the output of the Layer
        """
        return self._layer_output

    def learn(self):
        """
        Execute learning phase for Neurons
        """
        for neuron in self._neurons:
            neuron.learn()

    @property
    def neurons(self) -> List[narla.neurons.Neuron]:
        return self._neurons

    @property
    def number_of_neurons(self) -> int:
        """
        Number of Neurons in the Layer
        """
        return len(self._neurons)
