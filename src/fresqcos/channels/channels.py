"""Module for defining different types of free-space channels."""

from abc import ABC, abstractmethod
from fresqcos.channels.stations import ReceiverStation, TransmitterStation
from geometry import slant_range_from_coordinates


class Channel(ABC):
    """Abstract base class representing a free-space optical communication channel.

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

    @abstractmethod
    def compute_channel_length(self):
        """Calculate the length of the channel."""
        length = slant_range_from_coordinates(
            self.transmitter_station.latitude,
            self.transmitter_station.longitude,
            self.transmitter_station.altitude,
            self.receiver_station.latitude,
            self.receiver_station.longitude,
            self.receiver_station.altitude
        )
        return length

    @abstractmethod
    def compute_channel_losses(self):
        """Calculate the losses of the channel."""
        pass

class FiberChannel(Channel):
    """Class representing a fiber optic communication channel."""

    def __init__(self, loss_per_km):
        self.loss_per_km = loss_per_km

    def compute_channel_losses(self):
        """Calculate the losses of the fiber optic channel."""