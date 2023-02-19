import narla
import torch
from typing import List


class Layer:
    def __init__(self, observation_size: int, number_of_actions: int, number_of_neurons: int):
        self.observation_size = observation_size
        self.number_of_actions = number_of_actions

        self._neurons: List[narla.neurons.Neuron] = self._build_layer(
            observation_size=observation_size,
            number_of_actions=number_of_actions,
            number_of_neurons=number_of_neurons
        )

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        layer_output = torch.zeros((1, self.number_of_neurons), device=narla.settings.device)
        for index, neuron in enumerate(self._neurons):
            action = neuron.act(observation)
            layer_output[0, index] = action

        return layer_output

    @staticmethod
    def _build_layer(
        observation_size: int,
        number_of_actions: int,
        number_of_neurons: int
    ) -> List[narla.neurons.Neuron]:
        neurons: List[narla.neurons.Neuron] = []
        for _ in range(number_of_neurons):
            NeuronType = narla.neurons.deep_q.Neuron
            if narla.settings.neuron_type == narla.neurons.neuron_types.ACTOR_CRITIC:
                NeuronType = narla.neurons.actor_critic.Neuron

            neuron = NeuronType(
                observation_size=observation_size,
                number_of_actions=number_of_actions
            )
            neurons.append(neuron)

        return neurons

    def learn(self):
        for neuron in self._neurons:
            neuron.learn()

    @property
    def number_of_neurons(self) -> int:
        return len(self._neurons)

    def record(self, **kwargs):
        for neuron in self._neurons:
            neuron.record(**kwargs)
