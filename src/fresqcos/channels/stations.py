"""Module defining classes for stations or platforms in a free-space optical
communication system."""

from abc import ABC
from fresqcos.telescopes import Transmitter
from fresqcos.telescopes import Receiver


class Station(ABC):
    """Abstract base class representing a station/platform in a free-space optical
    communication system."""

    def __init__(
        self, name: str, latitude: float, longitude: float, altitude: float
    ) -> None:
        """Initialize the station with the given parameters.

        Parameters
        ----------
        name : str
            The name of the station.
        latitude : float
            The latitude of the station.
        longitude : float
            The longitude of the station.
        altitude : float
            The altitude of the station.
        """
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude

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
    def latitude(self) -> float:
        """Return the latitude of the station.

        Must be in [-90, 90].
        """
        return self._latitude

    @latitude.setter
    def latitude(self, value: float) -> None:
        if not -90 <= value <= 90:
            raise ValueError(f"latitude must be in [-90, 90], got {value}")
        self._latitude = value

    @property
    def longitude(self) -> float:
        """Return the longitude of the station.

        Must be in [-180, 180].
        """
        return self._longitude

    @longitude.setter
    def longitude(self, value: float) -> None:
        if not -180 <= value <= 180:
            raise ValueError(f"longitude must be in [-180, 180], got {value}")
        self._longitude = value

    @property
    def altitude(self) -> float:
        """Return the altitude of the station in meters.

        Must be non-negative.
        """
        return self._altitude

    @altitude.setter
    def altitude(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"altitude must be non-negative, got {value}")
        self._altitude = value


class TransmitterStation(Station, Transmitter):
    """Class representing a station hosting a transmitter in a free-space optical
    communication system."""

    def __init__(
        self,
        name: str,
        latitude: float,
        longitude: float,
        altitude: float,
        transmitter: Transmitter,
    ) -> None:
        """Initialize the transmitter station with the given parameters.

        Parameters
        ----------
        name : str
            The name of the station.
        latitude : float
            The latitude of the station.
        longitude : float
            The longitude of the station.
        altitude : float
            The altitude of the station.
        transmitter : Transmitter
            The transmitter hosted by the station.
        """
        super().__init__(name, latitude, longitude, altitude)
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
            raise ValueError(
                f"transmitter must be an instance of Transmitter, got {type(value)}"
            )
        self._transmitter = value


class ReceiverStation(Station, Receiver):
    """Class representing a station hosting a receiver in a free-space optical
    communication system."""

    def __init__(
        self,
        name: str,
        latitude: float,
        longitude: float,
        altitude: float,
        receiver: Receiver,
    ) -> None:
        """Initialize the receiver station with the given parameters.

        Parameters
        ----------
        name : str
            The name of the station.
        latitude : float
            The latitude of the station.
        longitude : float
            The longitude of the station.
        altitude : float
            The altitude of the station.
        receiver : Receiver
            The receiver hosted by the station.
        """
        super().__init__(name, latitude, longitude, altitude)
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
            raise ValueError(
                f"receiver must be an instance of Receiver, got {type(value)}"
            )
        self._receiver = value
