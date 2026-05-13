"""Module containing classes related to the receiver of a free-space optical communication system."""

from abc import ABC, abstractmethod


class Receiver(ABC):
    """Abstract base class representing the receiver telescope of a free-space optical communication system.

    Parameters
    ----------
    wavelength : float
        The wavelength of the classical optical signal.
    aperture : float
        The diameter of the receiver aperture.
    obscuration_ratio : float
        The ratio of the obscured area to the total area of the aperture.
    internal_loss : float
        The internal loss in the receiver.
    """

    def __init__(self, wavelength, aperture, obscuration_ratio, internal_loss):
        self.wavelength = wavelength
        self.waist_radius = aperture
        self.obscuration_ratio = obscuration_ratio
        self.internal_loss = internal_loss

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        if value <= 0:
            raise ValueError(f"wavelength must be positive, got {value}")
        self._wavelength = value

    @property
    def aperture(self):
        return self._aperture

    @aperture.setter
    def aperture(self, value):
        if value <= 0:
            raise ValueError(f"aperture must be positive, got {value}")
        self._aperture = value

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
        if not 0 <= value < 1:
            raise ValueError(f"internal_loss must be in [0, 1), got {value}")
        self._internal_loss = value
