"""Module for defining different types of communication channels."""

from abc import ABC, abstractmethod
from enum import Enum
from fresqcos.channels.stations import ReceiverStation, TransmitterStation
from fresqcos.channels.geometry import (
    slant_range_from_coordinates,
    compute_sec,
)
from fresqcos.channels.atmosphere import Atmosphere
from scipy.integrate import quad
import numpy as np


def output_plane_parameters(
    length: float,
    wavelength: float,
    waist_radius_in: float = np.inf,
    radius_of_curvature_in: float = np.inf,
) -> tuple[float, float]:
    """Calculate the output plane parameters for a beam.

    Parameters
    ----------
    length : float
        The propagation distance of the beam in meters.
    wavelength : float
        The wavelength of the beam in meters.
    waist_radius_in : float, optional
        The radius of the input beam waist in meters.
        Use np.inf for a plane wave, 0 for a spherical (point-source) wave.
        Defaults to np.inf (plane wave).
    radius_of_curvature_in : float, optional
        The radius of curvature of the input beam in meters.
        Use np.inf for a plane wave, 0 for a spherical (point-source) wave.
        Defaults to np.inf (plane wave).

    Returns
    -------
    tuple[float, float]
        A tuple containing the output beam parameters Theta and Lambda.
    """
    if length == 0:
        return 1, 0

    # Plane wave: infinite waist, infinite radius of curvature
    if np.isinf(waist_radius_in) and np.isinf(radius_of_curvature_in):
        return 1, 0

    # Spherical wave: point source, zero waist or zero radius of curvature
    if waist_radius_in == 0 or radius_of_curvature_in == 0:
        return 0, 0

    # General Gaussian beam
    k = 2 * np.pi / wavelength
    lambda_in = 2 * length / (k * waist_radius_in**2)
    theta_in = 1 - length / radius_of_curvature_in
    denom = theta_in**2 + lambda_in**2
    theta_out = theta_in / denom
    lambda_out = lambda_in / denom
    return theta_out, lambda_out


class LinkType(str, Enum):
    """Geometry of the free-space optical link."""

    HORIZONTAL = "horizontal"
    UPLINK = "uplink"
    DOWNLINK = "downlink"


class WaveType(str, Enum):
    """Wavefront model used for the Rytov approximation."""

    PLANE = "plane"
    SPHERICAL = "spherical"
    GAUSSIAN = "gaussian"


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

    def normalized_distance_variable(self, link_type: LinkType) -> callable:
        """Calculate the normalized distance variable xi for a given altitude h and link type.

        Parameters
        ----------
        link_type : LinkType
            Geometry of the link: UPLINK, or DOWNLINK.

        Returns
        -------
        callable
            The normalized distance variable xi as a function of altitude.
        """
        h_rx = self.receiver_station.altitude * 1e3
        h_tx = self.transmitter_station.altitude * 1e3
        if link_type == LinkType.DOWNLINK:
            xi = lambda h: (h - h_rx) / (h_tx - h_rx)
        elif link_type == LinkType.UPLINK:
            xi = lambda h: 1 - (h - h_rx) / (h_tx - h_rx)
        else:
            raise ValueError(f"Invalid link_type: {link_type}")
        return xi

    def compute_channel_losses(self) -> float:
        """Compute the total losses of the free-space channel in dB."""
        pass

    def draw_channel_pdf_sample(self) -> float:
        """Draw a random sample from the channel loss distribution."""
        pass

    def compute_rytov_variance(
        self, zenith_angle: float, link_type: LinkType, wave_type: WaveType
    ) -> float:
        """Compute the Rytov variance for this atmosphere given a link geometry.

        Parameters
        ----------
        zenith_angle : float
            The zenith angle of the link in degrees.
        link_type : LinkType
            Geometry of the link: HORIZONTAL, UPLINK, or DOWNLINK.
        wave_type : WaveType
            Wavefront model: PLANE, SPHERICAL, or GAUSSIAN.

        Returns
        -------
        float
            The Rytov variance.
        """
        k = 2 * np.pi / self.transmitter_station.transmitter.wavelength

        length = 1e3 * self.compute_channel_length()

        if wave_type == WaveType.PLANE:
            radius_of_curvature_in = np.inf
            waist_radius_in = np.inf
        elif wave_type == WaveType.SPHERICAL:
            radius_of_curvature_in = length
            waist_radius_in = 0
        elif wave_type == WaveType.GAUSSIAN:
            radius_of_curvature_in = np.inf
            waist_radius_in = self.transmitter_station.transmitter.waist_radius
        else:
            raise ValueError(f"Invalid wave_type: {wave_type}")

        theta_out, lambda_out = output_plane_parameters(
            length,
            self.transmitter_station.transmitter.wavelength,
            waist_radius_in,
            radius_of_curvature_in,
        )
        theta_bar_out = 1 - theta_out

        receiver_alt = self.receiver_station.altitude * 1e3
        transmitter_alt = self.transmitter_station.altitude * 1e3

        scale = 1e14  # Scale factor to avoid numerical issues in integration

        if link_type == LinkType.DOWNLINK or link_type == LinkType.UPLINK:
            if link_type == LinkType.DOWNLINK:
                xi = self.normalized_distance_variable(LinkType.DOWNLINK)
            else:
                xi = self.normalized_distance_variable(LinkType.UPLINK)
            integrand = lambda h: np.real(
                self.atmospheric_channel.cn2_profile(h)
                * scale
                * (
                    xi(h) ** (5 / 6)
                    * (lambda_out * xi(h) + 1j * (1 - theta_bar_out * xi(h))) ** (5 / 6)
                    - lambda_out ** (5 / 6) * xi(h) ** (5 / 3)
                )
            )
            integral = quad(integrand, receiver_alt, transmitter_alt)[0]
            rytov_var = (
                8.7
                * integral
                / scale
                * k ** (7 / 6)
                * (transmitter_alt - receiver_alt) ** (5 / 6)
                * compute_sec(zenith_angle) ** (11 / 6)
            )
        elif link_type == LinkType.HORIZONTAL:
            rytov_var_plane = (
                1.23
                * self.atmospheric_channel.cn2_profile
                * k ** (7 / 6)
                * length ** (11 / 6)
            )
            rytov_var = (
                3.86
                * rytov_var_plane
                * (
                    0.4
                    * ((1 + 2 * theta_out) ** 2 + 4 * lambda_out**2) ** (5 / 12)
                    * np.cos(5 / 6 * np.arctan((1 + 2 * theta_out) / (2 * lambda_out)))
                    - 11 / 16 * lambda_out ** (5 / 6)
                )
            )
        else:
            raise ValueError(f"Invalid link_type: {link_type}")

        return rytov_var


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

    def compute_rytov_variance(self, zenith_angle: float) -> float:
        """Compute rytov variance of a plane wave for a downlink channel [Andrews/Phillips, 2005, from Eqs 12.92 and 12.37].

        Returns
        -------
        rytov_var : float
            Rytov variance for plane wave propagating in the downlink channel.
        """
        rytov_var = super().compute_rytov_variance(
            zenith_angle=zenith_angle, link_type="downlink", wave_type="spherical"
        )
        return rytov_var

    def compute_channel_losses(self) -> float:
        pass

    def draw_channel_pdf_sample(self) -> float:
        """Draw a random sample from the channel loss distribution."""
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
