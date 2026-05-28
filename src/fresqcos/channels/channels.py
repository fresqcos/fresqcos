"""Module for defining different types of communication channels."""

from abc import ABC, abstractmethod
from fresqcos.channels.stations import ReceiverStation, TransmitterStation
from fresqcos.channels.geometry import (
    slant_range_from_coordinates,
    zenith_angle_from_coordinates,
    sec,
)
from fresqcos.channels.atmosphere import Atmosphere
from scipy.integrate import quad
import numpy as np


class Channel(ABC):
    """Abstract base class for any communication channel."""

    @abstractmethod
    def compute_channel_losses(self) -> float:
        """Return the total losses of the channel in dB."""
        pass


class FiberChannel(Channel):
    """A fiber optic communication channel."""

    def __init__(self, distance_km: float, loss_per_km: float):
        """Initialize the fiber channel with the given parameters.

        Parameters
        ----------
        distance_km : float
            Length of the fiber in kilometers.
        loss_per_km : float
            Attenuation coefficient in dB/km.
        """
        self.distance_km = distance_km
        self.loss_per_km = loss_per_km

    @property
    def distance_km(self) -> float:
        """Return the length of the fiber channel in km. Must be positive."""
        return self._distance_km

    @distance_km.setter
    def distance_km(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"distance_km must be positive, got {value}")
        self._distance_km = value

    @property
    def loss_per_km(self) -> float:
        """Return the attenuation coefficient of the fiber channel in dB/km. Must be positive."""
        return self._loss_per_km

    @loss_per_km.setter
    def loss_per_km(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"loss_per_km must be positive, got {value}")
        self._loss_per_km = value

    def compute_channel_losses(self) -> float:
        """Return the total losses of the fiber channel in dB."""
        return self.distance_km * self.loss_per_km


class FreeSpaceChannel(Channel):
    """Abstract base class for free-space optical communication channels."""

    def __init__(
        self,
        transmitter_station: TransmitterStation,
        receiver_station: ReceiverStation,
        atmospheric_channel: Atmosphere,
    ) -> None:
        """Initialize the free-space channel with the given parameters.

        Parameters
        ----------
        transmitter_station : TransmitterStation
            The station hosting the transmitter.
        receiver_station : ReceiverStation
            The station hosting the receiver.
        atmospheric_channel : Atmosphere
            The atmospheric effects on the channel.
        """
        self.transmitter_station = transmitter_station
        self.receiver_station = receiver_station
        self.atmospheric_channel = atmospheric_channel

    @property
    def transmitter_station(self) -> TransmitterStation:
        """Return the transmitter station. Must be an instance of TransmitterStation."""
        return self._transmitter_station

    @transmitter_station.setter
    def transmitter_station(self, value: TransmitterStation) -> None:
        if not isinstance(value, TransmitterStation):
            raise ValueError(
                f"transmitter_station must be an instance of TransmitterStation, got {type(value)}"
            )
        self._transmitter_station = value

    @property
    def receiver_station(self) -> ReceiverStation:
        """Return the receiver station. Must be an instance of ReceiverStation."""
        return self._receiver_station

    @receiver_station.setter
    def receiver_station(self, value: ReceiverStation) -> None:
        if not isinstance(value, ReceiverStation):
            raise ValueError(
                f"receiver_station must be an instance of ReceiverStation, got {type(value)}"
            )
        self._receiver_station = value

    @property
    def atmospheric_channel(self) -> Atmosphere:
        """Return the atmospheric channel. Must be an instance of Atmosphere."""
        return self._atmospheric_channel

    @atmospheric_channel.setter
    def atmospheric_channel(self, value: Atmosphere) -> None:
        if not isinstance(value, Atmosphere):
            raise ValueError(
                f"atmospheric_channel must be an instance of Atmosphere, got {type(value)}"
            )
        self._atmospheric_channel = value

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
        """Compute the total losses of the free-space channel in dB."""
        pass

    @abstractmethod
    def draw_channel_pdf_sample(self) -> float:
        """Draw a random sample from the channel loss distribution."""
        pass


class DownlinkChannel(FreeSpaceChannel):
    """A downlink free-space optical communication channel."""

    def __init__(
        self,
        transmitter_station: TransmitterStation,
        receiver_station: ReceiverStation,
        atmospheric_channel: Atmosphere,
    ) -> None:
        """Initialize the downlink channel with the given parameters.
        Parameters
        ----------
        transmitter_station : TransmitterStation
            The station hosting the transmitter.
        receiver_station : ReceiverStation
            The station hosting the receiver.
        atmospheric_channel : Atmosphere
            The atmospheric effects on the channel.
        """
        super().__init__(transmitter_station, receiver_station, atmospheric_channel)

    def compute_channel_losses(self) -> float:
        pass


class UplinkChannel(FreeSpaceChannel):
    """An uplink free-space optical communication channel."""

    def compute_channel_losses(self) -> float:
        pass


class HorizontalChannel(FreeSpaceChannel):
    """A horizontal free-space optical communication channel."""

    def compute_channel_losses(self) -> float:
        pass


class SatToAerialChannel(DownlinkChannel):
    """A satellite to aerial platform free-space optical communication channel."""

    def compute_channel_losses(self) -> float:
        pass


class AerialToSatChannel(UplinkChannel):
    """An aerial platform to satellite free-space optical communication channel."""

    def compute_channel_losses(self) -> float:
        pass


class AerialToGroundChannel(DownlinkChannel):
    """An aerial platform to ground free-space optical communication channel."""

    def compute_channel_losses(self) -> float:
        pass


class GroundToAerialChannel(UplinkChannel):
    """A ground to aerial platform free-space optical communication channel."""

    def compute_channel_losses(self) -> float:
        pass


class AerialToAerialChannel(HorizontalChannel):
    """An aerial platform to aerial platform free-space optical communication channel."""

    def compute_channel_losses(self) -> float:
        pass


class SatToSatChannel(HorizontalChannel):
    """A satellite to satellite free-space optical communication channel."""

    def compute_channel_losses(self) -> float:
        pass


class SatToGroundChannel(DownlinkChannel):
    """A satellite to ground free-space optical communication channel."""

    def compute_channel_losses(self) -> float:
        pass


class GroundToSatChannel(UplinkChannel):
    """A ground to satellite free-space optical communication channel."""

    def compute_channel_losses(self) -> float:
        pass


class SatToGroundChannel(DownlinkChannel):
    """A satellite to ground free-space optical communication channel."""

    def compute_channel_losses(self) -> float:
        pass


class SatToSatChannel(HorizontalChannel):
    """A satellite to satellite free-space optical communication channel."""

    def compute_channel_losses(self) -> float:
        pass
