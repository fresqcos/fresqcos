"""Module for modeling atmospheric effects on free-space optical communication
channels."""

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Union
from numbers import Real

IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]


class Atmosphere:
    """Class representing the atmospheric effects on a free-space optical
    communication channel."""

    def __init__(
        self,
        cn2_profile: Union[Callable, FloatArray, float],
        wind_speed: float,
        visibility: float,
        cn2_altitudes: FloatArray | None = None,
    ) -> None:
        """Initialize the atmosphere with the given parameters.

        Parameters
        ----------
        cn2_profile : Union[Callable, FloatArray, float]
            The refractive index structure constant profile as a function of altitude.
            If a FloatArray, cn2_altitudes must also be provided.
        wind_speed : float
            The wind speed in m/s, which can affect the beam propagation.
        visibility : float
            The visibility in kilometers, which affects the atmospheric attenuation.
        cn2_altitudes : FloatArray | None
            The altitudes in meters corresponding to the cn2_profile values if cn2_profile
            is a FloatArray. Ignored if cn2_profile is callable or scalar.
        """
        self.cn2_altitudes = cn2_altitudes
        self.cn2_profile = cn2_profile
        self.wind_speed = wind_speed
        self.visibility = visibility

    @property
    def cn2_profile(self) -> Callable | float:
        """Return the Cn2 profile as a callable function or constant value."""
        return self._cn2_profile

    @cn2_profile.setter
    def cn2_profile(self, value: Union[Callable, FloatArray, float]) -> None:
        """Set the Cn2 profile.

        If value is callable, it is used directly.
        If value is a numpy array, it is interpolated.
        If value is a single number, it used directly.
        """
        if callable(value):
            self._cn2_profile = value

        elif isinstance(value, Real):
            cn2_value = float(value)

            if cn2_value < 0:
                raise ValueError(f"cn2_profile must be non-negative, got {cn2_value}")

            self._cn2_profile = cn2_value

        elif isinstance(value, np.ndarray):
            if self._cn2_altitudes is None:
                raise ValueError(
                    "cn2_altitudes must be provided when cn2_profile is a FloatArray."
                )

            if value.shape != self._cn2_altitudes.shape:
                raise ValueError(
                    f"cn2_profile and cn2_altitudes must have the same shape, "
                    f"got {value.shape} and {self._cn2_altitudes.shape}."
                )

            self._cn2_profile = lambda h: np.interp(h, self._cn2_altitudes, value)

        else:
            raise ValueError(
                "cn2_profile must be a callable, numpy array, or scalar value, "
                f"got {type(value)}."
            )

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

    @property
    def cn2_altitudes(self) -> FloatArray | None:
        """Return the altitudes corresponding to the cn2_profile values if cn2_profile is a
        FloatArray."""
        return self._cn2_altitudes

    @cn2_altitudes.setter
    def cn2_altitudes(self, value: FloatArray | None) -> None:
        if value is not None and not isinstance(value, np.ndarray):
            raise ValueError(
                f"cn2_altitudes must be a numpy array or None, got {type(value)}"
            )
        self._cn2_altitudes = value
