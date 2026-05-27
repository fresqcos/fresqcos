"""Module containing classes related to the receiver of a free-space optical
communication system.
"""

from abc import ABC


class Receiver(ABC):
    """Abstract base class representing the receiver telescope of a free-space optical
    communication system.
    """

    def __init__(
        self,
        wavelength: float,
        aperture: float,
        obscuration_ratio: float,
        internal_loss: float,
    ) -> None:
        """Initialize the receiver with the given parameters.

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
        self.wavelength = wavelength
        self.waist_radius = aperture
        self.obscuration_ratio = obscuration_ratio
        self.internal_loss = internal_loss

    @property
    def wavelength(self) -> float:
        """Return the wavelength of the classical optical signal.

        Must be positive.
        """
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"wavelength must be positive, got {value}")
        self._wavelength = value

    @property
    def aperture(self) -> float:
        """Return the diameter of the receiver aperture.

        Must be positive.
        """
        return self._aperture

    @aperture.setter
    def aperture(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"aperture must be positive, got {value}")
        self._aperture = value

    @property
    def obscuration_ratio(self) -> float:
        """Return the obscuration ratio of the receiver.

        Must be in [0, 1).
        """
        return self._obscuration_ratio

    @obscuration_ratio.setter
    def obscuration_ratio(self, value: float) -> None:
        if not 0 <= value < 1:
            raise ValueError(f"obscuration_ratio must be in [0, 1), got {value}")
        self._obscuration_ratio = value

    @property
    def internal_loss(self) -> float:
        """Return the internal loss in the receiver.

        Must be in [0, 1).
        """
        return self._internal_loss

    @internal_loss.setter
    def internal_loss(self, value: float) -> None:
        if not 0 <= value < 1:
            raise ValueError(f"internal_loss must be in [0, 1), got {value}")
        self._internal_loss = value
