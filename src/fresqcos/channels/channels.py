"""Module for defining different types of communication channels."""

from abc import ABC, abstractmethod
from fresqcos.channels.stations import ReceiverStation, TransmitterStation
from geometry import slant_range_from_coordinates


class Channel(ABC):
    """Abstract base class for any communication channel."""

    @abstractmethod
    def compute_channel_length(self) -> float:
        """Return the length of the channel in km."""
        pass

    @abstractmethod
    def compute_channel_losses(self) -> float:
        """Return the total losses of the channel in dB."""
        pass


class FiberChannel(Channel):
    """A fiber optic communication channel.

    Parameters
    ----------
    distance_km : float
        Length of the fiber in kilometers.
    loss_per_km : float
        Attenuation coefficient in dB/km.
    """

    def __init__(self, distance_km: float, loss_per_km: float):
        self.distance_km = distance_km
        self.loss_per_km = loss_per_km

    def compute_channel_length(self) -> float:
        return self.distance_km

    def compute_channel_losses(self) -> float:
        return self.distance_km * self.loss_per_km


class FreeSpaceChannel(Channel):
    """Abstract base class for free-space optical communication channels.

    Parameters
    ----------
    transmitter_station : TransmitterStation
        The station hosting the transmitter.
    receiver_station : ReceiverStation
        The station hosting the receiver.
    """

    def __init__(self, transmitter_station: TransmitterStation, receiver_station: ReceiverStation):
        self.transmitter_station = transmitter_station
        self.receiver_station = receiver_station

    def compute_channel_length(self) -> float:
        """Calculate the slant range between the two stations."""
        return slant_range_from_coordinates(
            self.transmitter_station.latitude,
            self.transmitter_station.longitude,
            self.transmitter_station.altitude,
            self.receiver_station.latitude,
            self.receiver_station.longitude,
            self.receiver_station.altitude,
        )

    @abstractmethod
    def compute_channel_losses(self) -> float:
        pass
