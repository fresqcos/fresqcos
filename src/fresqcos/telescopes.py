"""Module containing classed related to the telescopes of a free-space optical
communication system."""


class Telescope:
    """Class representing a telescope in a free-space optical
    communication system."""

    def __init__(
        self,
        wavelength: float,
        obscuration_ratio: float,
        internal_loss: float,
    ) -> None:
        """Initialize the telescope with the given parameters.

        Parameters
        ----------
        wavelength : float
            The wavelength of the classical optical signal.
        obscuration_ratio : float
            The ratio of the obscured area to the total area of the aperture.
        internal_loss : float
            The internal loss in the telescope.
        """
        self.wavelength = wavelength
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
    def obscuration_ratio(self) -> float:
        """Return the obscuration ratio of the transmitter.

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
        """Return the internal loss of the transmitter.

        Must be in [0, 1].
        """
        return self._internal_loss

    @internal_loss.setter
    def internal_loss(self, value: float) -> None:
        if not 0 <= value <= 1:
            raise ValueError(f"internal_loss must be in [0, 1], got {value}")
        self._internal_loss = value


class Transmitter(Telescope):
    """Class representing the transmitter telescope of a free-space
    optical communication system."""

    def __init__(
        self,
        wavelength: float,
        waist_radius: float,
        obscuration_ratio: float,
        internal_loss: float,
        pointing_error_x: float = 0,
        pointing_error_y: float = 0,
    ) -> None:
        """Initialize the transmitter with the given parameters.

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
        pointing_error_x : float
            The pointing error of the transmitter in radians in the x-direction.
        pointing_error_y : float
            The pointing error of the transmitter in radians in the y-direction.
        """
        super().__init__(
            wavelength=wavelength,
            obscuration_ratio=obscuration_ratio,
            internal_loss=internal_loss,
        )
        self.waist_radius = waist_radius
        self.pointing_error_x = pointing_error_x
        self.pointing_error_y = pointing_error_y

    @property
    def waist_radius(self) -> float:
        """Return the radius of the beam waist.

        Must be positive.
        """
        return self._waist_radius

    @waist_radius.setter
    def waist_radius(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"waist_radius must be positive, got {value}")
        self._waist_radius = value

    @property
    def pointing_error_x(self) -> float:
        """Return the pointing error of the transmitter in the x-direction.

        Must be non-negative.
        """
        return self._pointing_error_x

    @pointing_error_x.setter
    def pointing_error_x(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"pointing_error_x must be non-negative, got {value}")

    @property
    def pointing_error_y(self) -> float:
        """Return the pointing error of the transmitter in the y-direction.

        Must be non-negative.
        """
        return self._pointing_error_y

    @pointing_error_y.setter
    def pointing_error_y(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"pointing_error_y must be non-negative, got {value}")
        self._pointing_error_y = value


class Receiver(Telescope):
    """Class representing the receiver telescope of a free-space optical
    communication system."""

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
        super().__init__(
            wavelength=wavelength,
            obscuration_ratio=obscuration_ratio,
            internal_loss=internal_loss,
        )
        self.aperture = aperture

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
