"""Module for defining different types of free-space channels."""

from abc import ABC


class Channel(ABC):
    """Abstract base class representing a free-space optical communication channel.

    Parameters
    ----------
    transmitter_station : TransmitterStation
        The station hosting the transmitter.
    receiver_station : ReceiverStation
        The station hosting the receiver.
    """

    def __init__(self, transmitter_station, receiver_station):
        self.transmitter_station = transmitter_station
        self.receiver_station = receiver_station
