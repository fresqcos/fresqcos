"""Module defining classes for stations or platforms in a free-space optical communication system"""

from abc import ABC, abstractmethod
from transmitters import Transmitter
from receivers import Receiver


class Station(ABC):
    """Abstract base class representing a station/platform in a free-space optical communication system.

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

    def __init__(self, name, latitude, longitude, altitude):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise ValueError(f"name must be a string, got {type(value)}")
        self._name = value

    @property
    def latitude(self):
        return self._latitude

    @latitude.setter
    def latitude(self, value):
        if not -90 <= value <= 90:
            raise ValueError(f"latitude must be in [-90, 90], got {value}")
        self._latitude = value

    @property
    def longitude(self):
        return self._longitude

    @longitude.setter
    def longitude(self, value):
        if not -180 <= value <= 180:
            raise ValueError(f"longitude must be in [-180, 180], got {value}")
        self._longitude = value

    @property
    def altitude(self):
        return self._altitude

    @altitude.setter
    def altitude(self, value):
        if value < 0:
            raise ValueError(f"altitude must be non-negative, got {value}")
        self._altitude = value


class TransmitterStation(Station, Transmitter):
    """Class representing a station hosting a transmitter in a free-space optical communication system.

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

    def __init__(self, name, latitude, longitude, altitude, transmitter):
        super().__init__(name, latitude, longitude, altitude)
        self.transmitter = transmitter

    @property
    def transmitter(self):
        return self._transmitter

    @transmitter.setter
    def transmitter(self, value):
        if not isinstance(value, Transmitter):
            raise ValueError(f"transmitter must be an instance of Transmitter, got {type(value)}")
        self._transmitter = value


class ReceiverStation(Station, Receiver):
    """Class representing a station hosting a receiver in a free-space optical communication system.

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

    def __init__(self, name, latitude, longitude, altitude, receiver):
        super().__init__(name, latitude, longitude, altitude)
        self.receiver = receiver

    @property
    def receiver(self):
        return self._receiver

    @receiver.setter
    def receiver(self, value):
        if not isinstance(value, Receiver):
            raise ValueError(f"receiver must be an instance of Receiver, got {type(value)}")
        self._receiver = value
