"""Module for modeling atmospheric effects on free-space optical communication
channels."""

from abc import ABC
import numpy as np
from numpy.typing import NDArray
from typing import Callable

IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]


class Atmosphere(ABC):
    """Abstract base class representing the atmospheric effects on a free-space optical
    communication channel."""

    def __init__(
        self, cn2_profile: Callable, wind_speed: float, visibility: float
    ) -> None:
        """Initialize the atmosphere with the given parameters.

        Parameters
        ----------
        cn2_profile : Callable
            The refractive index structure constant profile as a function of altitude.
        wind_speed : float
            The wind speed in m/s, which can affect the beam propagation.
        visibility : float
            The visibility in kilometers, which affects the atmospheric attenuation.
        """
        self.cn2_profile = cn2_profile
        self.wind_speed = wind_speed
        self.visibility = visibility

    @property
    def cn2_profile(self) -> Callable:
        """Return the refractive index structure constant profile as a function of
        altitude."""
        return self._cn2_profile

    @cn2_profile.setter
    def cn2_profile(self, value: Callable) -> None:
        if not callable(value):
            raise ValueError(
                f"cn2_profile must be a callable function, got {type(value)}"
            )
        self._cn2_profile = value

    @property
    def wind_speed(self) -> float:
        """Return the wind speed in m/s.

        Must be non-negative.
        """
        return self._wind_speed

    @wind_speed.setter
    def wind_speed(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"wind_speed must be non-negative, got {value}")
        self._wind_speed = value

    @property
    def visibility(self) -> float:
        """Return the visibility in kilometers.

        Must be positive.
        """
        return self._visibility

    @visibility.setter
    def visibility(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"visibility must be positive, got {value}")
        self._visibility = value


class SatToGroundAtmosphere(Atmosphere):
    """Class representing the atmospheric effects on a satellite-to-ground free-space
    optical communication channel."""

    pass
