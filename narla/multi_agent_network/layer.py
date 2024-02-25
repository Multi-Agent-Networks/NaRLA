from __future__ import annotations

from typing import List

import numpy as np
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
        # Defines the connectivity of Neurons in this Layer to the previous layer
        self.connectivity: torch.Tensor | None = None

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

        # Map the input observation through the connectivity matrix
        observation = observation.T * self.connectivity

        for index, neuron in enumerate(self._neurons):
            neuron_input = observation[:, index].reshape(1, -1)
            action = neuron.act(neuron_input)
            layer_output[0, index] = action

        self._layer_output = layer_output

        return layer_output

    @staticmethod
    def build_connectivity(observation_size: int, number_of_neurons: int, local_connectivity) -> torch.Tensor:
        """
        Build the connectivity matrix

        :param observation_size: Number of inputs in the observation
        :param number_of_neurons: Number of Neurons in the Layer
        :param local_connectivity: If ``True`` the connectivity matrix will be a diagonal with offsets
        """
        if local_connectivity:
            connectivity = np.zeros(shape=(observation_size, number_of_neurons), dtype=bool)
            for offset in [-1, 0, 1]:
                connectivity |= np.eye(observation_size, number_of_neurons, k=offset, dtype=bool)

            connectivity = torch.tensor(connectivity.astype(int))

        else:
            connectivity = torch.ones(size=(observation_size, number_of_neurons))

        connectivity = connectivity.to(device=narla.experiment_settings.trial_settings.device)

        return connectivity

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
        """
        Access the Neurons from the Layer
        """
        return self._neurons

    @property
    def number_of_neurons(self) -> int:
        """
        Number of Neurons in the Layer
        """
        return len(self._neurons)
