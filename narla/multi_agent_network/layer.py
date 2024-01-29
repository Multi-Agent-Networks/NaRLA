import narla
import torch
from typing import List


class Layer:
    """
    A Layer contains a list of Neurons

    :param observation_size: Size of the observation which the Layer will receive
    :param number_of_actions: Number of actions available to the Layer
    :param number_of_neurons: Number of Neurons in the Layer
    """
    def __init__(self, observation_size: int, learning_rate: float, number_of_actions: int, number_of_neurons: int):
        self.observation_size = observation_size
        self.number_of_actions = number_of_actions

        self._neurons: List[narla.neurons.Neuron] = self._build_layer(
            observation_size=observation_size,
            learning_rate=learning_rate,
            number_of_actions=number_of_actions,
            number_of_neurons=number_of_neurons
        )

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Take an action based on the observation

        :param observation: Observation from the Layer's local environment
        """
        layer_output = torch.zeros((1, self.number_of_neurons), device=narla.settings.device)
        for index, neuron in enumerate(self._neurons):
            action = neuron.act(observation)
            layer_output[0, index] = action

        return layer_output

    @staticmethod
    def _build_layer(
        observation_size: int,
        learning_rate: float,
        number_of_actions: int,
        number_of_neurons: int
    ) -> List[narla.neurons.Neuron]:
        neurons: List[narla.neurons.Neuron] = []
        for _ in range(number_of_neurons):
            NeuronType = narla.neurons.deep_q.Neuron
            if narla.settings.neuron_type == narla.neurons.AvailableNeurons.ACTOR_CRITIC:
                NeuronType = narla.neurons.actor_critic.Neuron
            if narla.settings.neuron_type == narla.neurons.AvailableNeurons.POLICY_GRADIENT:
                NeuronType = narla.neurons.policy_gradient.Neuron

            neuron = NeuronType(
                observation_size=observation_size,
                number_of_actions=number_of_actions,
                learning_rate=learning_rate,
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

    def learn(self):
        """
        Execute learning phase for Neurons
        """
        for neuron in self._neurons:
            neuron.learn()

    @property
    def number_of_neurons(self) -> int:
        """
        Number of Neurons in the Layer
        """
        return len(self._neurons)
