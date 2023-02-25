import enum
try:
    from typing import Literal
except:
    from typing_extensions import Literal


class AvailableNeurons(enum.Enum):
    ACTOR_CRITIC = "actor_critic"
    DEEP_Q = "deep_q"

    def __str__(self):
        return self.value


ALL_NEURONS = Literal[
    AvailableNeurons.ACTOR_CRITIC,
    AvailableNeurons.DEEP_Q
]
