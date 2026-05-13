"""Module for modeling atmospheric effects on free-space optical communication channels."""

from abc import ABC


class Atmosphere(ABC):
    """Abstract base class representing the atmospheric effects on a free-space optical communication channel.

    Parameters
    ----------
    cn2_profile : np.ndarray
        The refractive index structure constant profile as a function of altitude, which quantifies the strength of atmospheric turbulence.
    wind_speed : float
        The wind speed in m/s, which can affect the beam propagation.
    visibility : float
        The visibility in kilometers, which affects the atmospheric attenuation.
    """

    def __init__(self, cn2_profile, wind_speed, visibility):
        self.cn2_profile = cn2_profile
        self.wind_speed = wind_speed
        self.visibility = visibility

    @property
    def wind_speed(self):
        return self._wind_speed

    @wind_speed.setter
    def wind_speed(self, value):
        if value < 0:
            raise ValueError(f"wind_speed must be non-negative, got {value}")
        self._wind_speed = value

    @property
    def visibility(self):
        return self._visibility

    @visibility.setter
    def visibility(self, value):
        if value <= 0:
            raise ValueError(f"visibility must be positive, got {value}")
        self._visibility = value


class SatToGroundAtmosphere(Atmosphere):
    """Class representing the atmospheric effects on a satellite-to-ground free-space optical communication channel."""

    pass
