"""Module defining classes for stations or platforms in a free-space optical
communication system."""

from fresqcos.telescopes import Transmitter
from fresqcos.telescopes import Receiver


class Station:
    """Class representing a station/platform in a free-space optical
    communication system."""

    def __init__(
        self,
        name: str,
        altitude_km: float,
        latitude_deg: float | None = None,
        longitude_deg: float | None = None,
    ) -> None:
        """Initialize the station with the given parameters.

        Parameters
        ----------
        name : str
            The name of the station.
        altitude_km : float
            The altitude of the station in kilometers.
        latitude_deg : float
            The latitude of the station in degrees. Default is None.
        longitude_deg : float
            The longitude of the station in degrees. Default is None.
        """
        self.name = name
        self.altitude_km = altitude_km
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg

    @property
    def name(self) -> str:
        """Return the name of the station.

        Must be a string.
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if not isinstance(value, str):
            raise ValueError(f"name must be a string, got {type(value)}")
        self._name = value

    @property
    def altitude_km(self) -> float:
        """Return the altitude of the station in kilometers.

        Must be non-negative.
        """
        return self._altitude_km

    @property
    def altitude_m(self) -> float:
        """Return the altitude of the station in meters."""
        return self._altitude_km * 1e3

    @altitude_km.setter
    def altitude_km(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"altitude must be non-negative, got {value}")
        self._altitude_km = value

    @property
    def latitude_deg(self) -> float | None:
        """Return the latitude of the station in degrees.

        Must be in [-90, 90].
        """
        return self._latitude_deg

    @latitude_deg.setter
    def latitude_deg(self, value: float | None) -> None:
        if value is not None and not -90 <= value <= 90:
            raise ValueError(f"latitude must be in [-90, 90], got {value}")
        self._latitude_deg = value

    @property
    def longitude_deg(self) -> float | None:
        """Return the longitude of the station in degrees.

        Must be in [-180, 180].
        """
        return self._longitude_deg

    @longitude_deg.setter
    def longitude_deg(self, value: float | None) -> None:
        if value is not None and not -180 <= value <= 180:
            raise ValueError(f"longitude must be in [-180, 180], got {value}")
        self._longitude_deg = value


class TransmitterStation(Station):
    """Class representing a station hosting a transmitter in a free-space optical
    communication system."""

    def __init__(
        self,
        name: str,
        transmitter: Transmitter,
        altitude_km: float,
        latitude_deg: float | None = None,
        longitude_deg: float | None = None,
    ) -> None:
        """Initialize the transmitter station with the given parameters.

        Parameters
        ----------
        name : str
            The name of the station.
        transmitter : Transmitter
            The transmitter hosted by the station.
        altitude_km : float
            The altitude of the station in kilometers.
        latitude_deg : float
            The latitude of the station in degrees. Default is None.
        longitude_deg : float
            The longitude of the station in degrees. Default is None.
        """
        super().__init__(name, altitude_km, latitude_deg, longitude_deg)
        self.transmitter = transmitter

    @property
    def transmitter(self) -> Transmitter:
        """Return the transmitter hosted by the station.

        Must be an instance of Transmitter.
        """
        return self._transmitter

    @transmitter.setter
    def transmitter(self, value: Transmitter) -> None:
        if not isinstance(value, Transmitter):
            raise TypeError(
                f"transmitter must be an instance of Transmitter, got {type(value)}"
            )
        self._transmitter = value


class ReceiverStation(Station):
    """Class representing a station hosting a receiver in a free-space optical
    communication system."""

    def __init__(
        self,
        name: str,
        receiver: Receiver,
        altitude_km: float,
        latitude_deg: float | None = None,
        longitude_deg: float | None = None,
    ) -> None:
        """Initialize the receiver station with the given parameters.

        Parameters
        ----------
        name : str
            The name of the station.
        receiver : Receiver
            The receiver hosted by the station.
        latitude_deg : float
            The latitude of the station in degrees. Default is None.
        longitude_deg : float
            The longitude of the station in degrees. Default is None.
        altitude_km : float
            The altitude of the station in kilometers. Default is None.
        """
        super().__init__(name, altitude_km, latitude_deg, longitude_deg)
        self.receiver = receiver

    @property
    def receiver(self) -> Receiver:
        """Return the receiver hosted by the station.

        Must be an instance of Receiver.
        """
        return self._receiver

    @receiver.setter
    def receiver(self, value: Receiver) -> None:
        if not isinstance(value, Receiver):
            raise TypeError(
                f"receiver must be an instance of Receiver, got {type(value)}"
            )
        self._receiver = value
