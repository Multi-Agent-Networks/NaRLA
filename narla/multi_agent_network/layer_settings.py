import dataclasses

from narla.settings.base_settings import BaseSettings
from narla.neurons.neuron_settings import NeuronSettings


@dataclasses.dataclass
class LayerSettings(BaseSettings):
    neuron_settings: NeuronSettings = dataclasses.field(default_factory=lambda: NeuronSettings())

    number_of_neurons_per_layer: int = 15
    """Number of neurons per layer in the network (the last layer always has only one neuron)"""
