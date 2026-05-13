"""Module containing classed related to the transmitter of a free-space optical communication system."""

from abc import ABC, abstractmethod


class Transmitter(ABC):
    """Abstract base class representing the transmitter telescope of a free-space optical communication system.

    Parameters
    ----------
    wavelength : float
        The wavelength of the classical optical signal.
    waist_radius : float
        The radius of the beam waist.
    obscuration_ratio : float
        The ratio of the obscured area to the total area of the aperture.
    internal_loss : float
        The internal loss in the transmitter.
    pointing_error : float
        The pointing error of the transmitter.
    """

    def __init__(self, wavelength, waist_radius, obscuration_ratio, internal_loss, pointing_error):
        self.wavelength = wavelength
        self.waist_radius = waist_radius
        self.obscuration_ratio = obscuration_ratio
        self.internal_loss = internal_loss
        self.pointing_error = pointing_error

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        if value <= 0:
            raise ValueError(f"wavelength must be positive, got {value}")
        self._wavelength = value

    @property
    def waist_radius(self):
        return self._waist_radius

    @waist_radius.setter
    def waist_radius(self, value):
        if value <= 0:
            raise ValueError(f"waist_radius must be positive, got {value}")
        self._waist_radius = value

    @property
    def obscuration_ratio(self):
        return self._obscuration_ratio

    @obscuration_ratio.setter
    def obscuration_ratio(self, value):
        if not 0 <= value < 1:
            raise ValueError(f"obscuration_ratio must be in [0, 1), got {value}")
        self._obscuration_ratio = value

    @property
    def internal_loss(self):
        return self._internal_loss

    @internal_loss.setter
    def internal_loss(self, value):
        if not 0 <= value <= 1:
            raise ValueError(f"internal_loss must be in [0, 1], got {value}")
        self._internal_loss = value

    @property
    def pointing_error(self):
        return self._pointing_error

    @pointing_error.setter
    def pointing_error(self, value):
        if value < 0:
            raise ValueError(f"pointing_error must be non-negative, got {value}")
        self._pointing_error = value
