import enum
try:
    from typing import Literal
except:
    from typing_extensions import Literal


class AvailableEnvironments(enum.Enum):
    def __str__(self):
        return self.value


class GymEnvironments(AvailableEnvironments):
    CART_POLE = "CartPole-v1"
    MOUNTAIN_CAR = "MountainCar-v0"


ALL_ENVIRONMENTS = Literal[
    GymEnvironments.CART_POLE,
    GymEnvironments.MOUNTAIN_CAR
]
