"""Module for defining different types of communication channels."""

from abc import ABC, abstractmethod
from enum import Enum

from tomlkit import value
from fresqcos.channels.stations import ReceiverStation, TransmitterStation
from fresqcos.channels.geometry import (
    slant_range_from_coordinates,
    compute_sec,
    zenith_angle_from_coordinates,
)
from fresqcos.channels.atmosphere import Atmosphere
from scipy.integrate import quad
import numpy as np


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


def input_plane_parameters(
    length: float,
    wavelength: float,
    waist_radius_in: float,
    radius_of_curvature_in: float,
) -> tuple[float, float]:
    """Calculate the input plane parameters for a beam.

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
        A tuple containing the input beam parameters Theta_0 and Lambda_0.
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
    return theta_in, lambda_in


def output_plane_parameters(
    length: float,
    wavelength: float,
    waist_radius_in: float,
    radius_of_curvature_in: float,
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
    theta_in, lambda_in = input_plane_parameters(
        length, wavelength, waist_radius_in, radius_of_curvature_in
    )
    denom = theta_in**2 + lambda_in**2
    theta_out = theta_in / denom
    lambda_out = lambda_in / denom
    return theta_out, lambda_out


def compute_slant_integral_1(
    cn2_profile: callable,
    h_1: float,
    h_2: float,
    theta_out: float,
    normalized_dist: callable,
) -> float:
    """Compute the slant integral denoted as mu_1 in [Andrews/Phillips, 2005, Eqs 12.18 and 12.25].

    Parameters
    ----------
    cn2_profile : callable
        The refractive index structure constant profile as a function of altitude.
    normalized_dist : callable
        The normalized distance variable as a function of altitude.
    theta_out : float
        The output beam parameter Theta.
    h_1 : float
        The lower altitude in meters.
    h_2 : float
        The upper altitude in meters.

    Returns
    -------
    float
        The value of the slant integral.
    """
    theta_bar_out = 1 - theta_out
    scale = 1e14  # Scale factor to avoid numerical issues in integration
    integrand = (
        lambda h: cn2_profile(h)
        * (theta_out + theta_bar_out * (1 - normalized_dist(h))) ** (5 / 3)
        * scale
    )
    integral = quad(integrand, h_1, h_2)[0] / scale
    return integral


def compute_slant_integral_2(
    cn2_profile: callable, h_1: float, h_2: float, normalized_dist: callable
) -> float:
    """Compute the slant integral denoted as mu_2 in [Andrews/Phillips, 2005, Eqs 12.19 and 12.26].

    Parameters
    ----------
    cn2_profile : callable
        The refractive index structure constant profile as a function of altitude.
    h_1 : float
        The lower altitude in meters.
    h_2 : float
        The upper altitude in meters.
    normalized_dist : callable
        The normalized distance variable as a function of altitude.

    Returns
    -------
    float
        The value of the slant integral.
    """
    scale = 1e14  # Scale factor to avoid numerical issues in integration
    integrand = lambda h: cn2_profile(h) * normalized_dist(h) ** (5 / 3) * scale
    integral = quad(integrand, h_1, h_2)[0] / scale
    return integral


def compute_slant_integral_3(
    cn2_profile: callable,
    h_1: float,
    h_2: float,
    normalized_dist: callable,
    theta_out: float,
    lambda_out: float,
) -> float:
    """Compute the slant integral denoted as mu_3 in [Andrews/Phillips, 2005, Eq 12.37].

    Parameters
    ----------
    cn2_profile : callable
        The refractive index structure constant profile as a function of altitude.
    h_1 : float
        The lower altitude in meters.
    h_2 : float
        The upper altitude in meters.
    normalized_dist : callable
        The normalized distance variable as a function of altitude.
    theta_out : float
        The output beam parameter.
    lambda_out : float
        The output beam parameter.
    radius_of_curvature_in : float
        The radius of curvature of the input beam in meters.

    Returns
    -------
    float
        The value of the slant integral.
    """
    theta_bar_out = 1 - theta_out
    scale = 1e14  # Scale factor to avoid numerical issues in integration
    integrand = lambda h: np.real(
        cn2_profile(h)
        * (
            normalized_dist(h) ** (5 / 6)
            * (
                lambda_out * normalized_dist(h)
                + 1j * (1 - theta_bar_out * normalized_dist(h))
            )
            ** (5 / 6)
            - lambda_out ** (5 / 6) * normalized_dist(h) ** (5 / 3)
        )
        * scale
    )
    integral = quad(integrand, h_1, h_2)[0] / scale
    return integral


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

    def compute_channel_losses(self) -> float:
        """Compute the total losses of the free-space channel in dB."""
        pass

    def draw_channel_pdf_sample(self) -> float:
        """Draw a random sample from the channel loss distribution."""
        pass


class SlantChannel(FreeSpaceChannel):
    """Abstract base class for slant free-space optical communication channels."""

    def __init__(
        self,
        transmitter_station: TransmitterStation,
        receiver_station: ReceiverStation,
        atmospheric_channel: Atmosphere,
        zenith_angle_deg: float = None,
    ) -> None:
        """Initialize the slant free-space channel with the given parameters.

        Parameters
        ----------
        transmitter_station : TransmitterStation
            The station hosting the transmitter.
        receiver_station : ReceiverStation
            The station hosting the receiver.
        atmospheric_channel : Atmosphere
            The atmospheric effects on the channel.
        zenith_angle_deg : float, optional
            The zenith angle of the link in degrees. If not provided,
            it will be computed from the station coordinates.
        """
        super().__init__(atmospheric_channel=atmospheric_channel)
        self.transmitter_station = transmitter_station
        self.receiver_station = receiver_station
        self.zenith_angle_deg = zenith_angle_deg

    @property
    def transmitter_station(self) -> TransmitterStation:
        """Return the transmitter station. Must be an instance of TransmitterStation,
        and either have all coordinates defined or the altitude must be provided,
        in case the zenith angle is given."""
        return self._transmitter_station

    @transmitter_station.setter
    def transmitter_station(self, value: TransmitterStation) -> None:
        if not isinstance(value, TransmitterStation):
            raise ValueError(
                f"transmitter_station must be an instance of TransmitterStation, got {type(value)}"
            )
        if self.zenith_angle_deg is None:
            if (
                value.latitude is None
                or value.longitude is None
                or value.altitude is None
            ):
                raise ValueError(
                    "transmitter_station must have latitude, longitude, and altitude "
                    "defined when zenith_angle_deg is not provided."
                )
        else:
            if value.altitude is None:
                raise ValueError(
                    "transmitter_station must have altitude defined when "
                    "zenith_angle_deg is provided."
                )
            value.latitude = 0
            value.longitude = 0
        self._transmitter_station = value

    @property
    def receiver_station(self) -> ReceiverStation:
        """Return the receiver station. Must be an instance of ReceiverStation,
        and either have all coordinates defined or the altitude must be provided,
        in case the zenith angle is given."""
        return self._receiver_station

    @receiver_station.setter
    def receiver_station(self, value: ReceiverStation) -> None:
        if not isinstance(value, ReceiverStation):
            raise ValueError(
                f"receiver_station must be an instance of ReceiverStation, got {type(value)}"
            )
        if self.zenith_angle_deg is None:
            if (
                value.latitude is None
                or value.longitude is None
                or value.altitude is None
            ):
                raise ValueError(
                    "receiver_station must have latitude, longitude, and altitude "
                    "defined when zenith_angle_deg is not provided."
                )
        else:
            if value.altitude is None:
                raise ValueError(
                    "receiver_station must have altitude defined when "
                    "zenith_angle_deg is provided."
                )
            value.latitude = 0
            value.longitude = 0
        self._receiver_station = value

    @property
    def zenith_angle_deg(self) -> float:
        """Return the zenith angle of the link in degrees.
        If not provided, it is computed from the station coordinates.
        """
        if self._zenith_angle_deg is not None:
            return self._zenith_angle_deg
        return zenith_angle_from_coordinates(
            self.transmitter_station.latitude,
            self.transmitter_station.longitude,
            self.transmitter_station.altitude,
            self.receiver_station.latitude,
            self.receiver_station.longitude,
            self.receiver_station.altitude,
        )

    @zenith_angle_deg.setter
    def zenith_angle_deg(self, value: float | None) -> None:
        if value is not None and not (0 <= value <= 180):
            raise ValueError(f"zenith_angle must be in [0, 180] degrees, got {value}")
        self._zenith_angle_deg = value

    @property
    def zenith_angle_rad(self) -> float:
        """Return the zenith angle in radians."""
        return np.deg2rad(self.zenith_angle_deg)

    @property
    def transmitter_altitude_m(self) -> float:
        """Return the altitude of the transmitter station in meters."""
        return self.transmitter_station.altitude * 1e3

    @property
    def receiver_altitude_m(self) -> float:
        """Return the altitude of the receiver station in meters."""
        return self.receiver_station.altitude * 1e3

    @property
    def channel_length_m(self) -> float:
        """Return the length of the free-space channel in meters."""
        return (
            slant_range_from_coordinates(
                self.transmitter_station.latitude,
                self.transmitter_station.longitude,
                self.transmitter_station.altitude,
                self.receiver_station.latitude,
                self.receiver_station.longitude,
                self.receiver_station.altitude,
            )
            * 1e3
        )

    def _normalized_distance_variable(self, link_type: LinkType) -> callable:
        """Calculate the normalized distance variable xi for a given altitude h and link type.

        Parameters
        ----------
        link_type : LinkType
            Geometry of the link: UPLINK or DOWNLINK.

        Returns
        -------
        callable
            The normalized distance variable xi as a function of altitude.
        """
        h_rx = self.receiver_altitude_m
        h_tx = self.transmitter_altitude_m
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

    def _wave_parameters_in(
        self, wave_type: WaveType, length: float
    ) -> tuple[float, float]:
        """Return wave parameters for the given wave_type.

        Parameters
        ----------
        wave_type : WaveType
            Wavefront model: PLANE, SPHERICAL, or GAUSSIAN.
        length : float
            The length of the link in meters.
        Returns
        -------
        tuple[float, float]
            The radius of curvature and waist radius for the input wave type.
        """
        if wave_type == WaveType.PLANE:
            return np.inf, np.inf
        elif wave_type == WaveType.SPHERICAL:
            return length, 0
        elif wave_type == WaveType.GAUSSIAN:
            return np.inf, self.transmitter_station.transmitter.waist_radius
        else:
            raise ValueError(f"Invalid wave_type: {wave_type}")

    def compute_rytov_variance(self, link_type: LinkType, wave_type: WaveType) -> float:
        """Compute the Rytov variance of the link [Andrews/Phillips, 2005, Eqs 12.92, 5.15 and 9.93].

        Parameters
        ----------
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
        length = self.channel_length_m

        radius_of_curvature_in, waist_radius_in = self._wave_parameters_in(
            wave_type, length
        )
        theta_out, lambda_out = output_plane_parameters(
            length,
            self.transmitter_station.transmitter.wavelength,
            waist_radius_in,
            radius_of_curvature_in,
        )

        if link_type == LinkType.DOWNLINK or link_type == LinkType.UPLINK:
            receiver_alt = self.receiver_altitude_m
            transmitter_alt = self.transmitter_altitude_m
            xi = self._normalized_distance_variable(link_type)
            integral = compute_slant_integral_3(
                self.atmospheric_channel.cn2_profile,
                receiver_alt,
                transmitter_alt,
                xi,
                theta_out,
                lambda_out,
            )
            rytov_var = (
                8.7
                * integral
                * k ** (7 / 6)
                * (transmitter_alt - receiver_alt) ** (5 / 6)
                * compute_sec(self.zenith_angle_deg) ** (11 / 6)
            )
        else:
            raise ValueError(f"Invalid link_type: {link_type}")

        return rytov_var

    def compute_coherence_width(
        self, link_type: LinkType, wave_type: WaveType
    ) -> float:
        """Compute the coherence width of the propagated beam [Andrews/Phillips, 2005, derived from Eq 12.27, Eq 6.132].

        Parameters
        ----------
        link_type : LinkType
            Geometry of the link: HORIZONTAL, UPLINK, or DOWNLINK.
        wave_type : WaveType
            Wavefront model: PLANE, SPHERICAL, or GAUSSIAN.

        Returns
        -------
        float
            The coherence width in meters.
        """
        k = 2 * np.pi / self.transmitter_station.transmitter.wavelength
        length = self.channel_length_m

        radius_of_curvature_in, waist_radius_in = self._wave_parameters_in(
            wave_type, length
        )
        theta_out, lambda_out = output_plane_parameters(
            length,
            self.transmitter_station.transmitter.wavelength,
            waist_radius_in,
            radius_of_curvature_in,
        )

        if length == 0:
            coherence_width = np.inf
        else:
            if link_type == LinkType.DOWNLINK or link_type == LinkType.UPLINK:
                xi = self._normalized_distance_variable(link_type)
                receiver_alt = self.receiver_altitude_m
                transmitter_alt = self.transmitter_altitude_m
                mu_1 = compute_slant_integral_1(
                    self.atmospheric_channel.cn2_profile,
                    receiver_alt,
                    transmitter_alt,
                    theta_out,
                    xi,
                )
                mu_2 = compute_slant_integral_2(
                    self.atmospheric_channel.cn2_profile,
                    receiver_alt,
                    transmitter_alt,
                    xi,
                )
                coherence_width = (
                    np.cos(self.zenith_angle_rad)
                    / (0.423 * k**2 * (mu_1 + 0.62 * mu_2 * lambda_out ** (11 / 6)))
                ) ** (3 / 5)
        return coherence_width

    def compute_wandering_variance(
        self, link_type: LinkType, wave_type: WaveType
    ) -> float:
        """Compute the beam wandering variance for a free-space optical channel.

        Parameters
        ----------
        link_type : LinkType
            Geometry of the link: HORIZONTAL, UPLINK, or DOWNLINK.
        wave_type : WaveType
            Type of the wave: PLANE, SPHERICAL OR GAUSSIAN.

        Returns
        -------
        float
            The beam wandering variance in square meters.
        """
        length = self.channel_length_m

        radius_of_curvature_in, waist_radius_in = self._wave_parameters_in(
            wave_type, length
        )
        theta_in, lambda_in = input_plane_parameters(
            length,
            self.transmitter_station.transmitter.wavelength,
            waist_radius_in,
            radius_of_curvature_in,
        )
        theta_bar_in = 1 - theta_in
        rytov_var = self.compute_rytov_variance(link_type, wave_type)
        tx_waist = self.transmitter_station.transmitter.waist_radius

        if link_type == LinkType.DOWNLINK or link_type == LinkType.UPLINK:
            receiver_alt = self.receiver_altitude_m
            transmitter_alt = self.transmitter_altitude_m
            xi = self._normalized_distance_variable(link_type)
            scale = 1e14  # Scale factor to avoid numerical issues in integration
            integrand = (
                lambda h: self.atmospheric_channel.cn2_profile(h)
                * xi(h) ** 2
                / (
                    (theta_in + theta_bar_in * xi(h)) ** 2
                    + 1.63 * rytov_var ** (6 / 5) * lambda_in * (1 - xi(h)) ** (16 / 5)
                )
                ** (1 / 6)
                * scale
            )
            integral = quad(integrand, receiver_alt, transmitter_alt)[0]
            wandering_variance = (
                7.25
                * compute_sec(self.zenith_angle_deg) ** 3
                * tx_waist ** (-1 / 3)
                * integral
                / scale
            )
        else:
            raise ValueError(f"Invalid link_type: {link_type}")

        return wandering_variance


class HorizontalChannel(FreeSpaceChannel):
    """A horizontal free-space optical communication channel."""

    def compute_channel_losses(self) -> float:
        pass

    def compute_rytov_variance(self, wave_type: WaveType) -> float:
        """Compute the Rytov variance of the link [Andrews/Phillips, 2005, Eqs 12.92, 5.15 and 9.93].

        Parameters
        ----------
        wave_type : WaveType
            Wavefront model: PLANE, SPHERICAL, or GAUSSIAN.

        Returns
        -------
        float
            The Rytov variance.
        """
        k = 2 * np.pi / self.transmitter_station.transmitter.wavelength
        length = self.channel_length_m

        radius_of_curvature_in, waist_radius_in = self._wave_parameters_in(
            wave_type, length
        )
        theta_out, lambda_out = output_plane_parameters(
            length,
            self.transmitter_station.transmitter.wavelength,
            waist_radius_in,
            radius_of_curvature_in,
        )

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

        return rytov_var

    def compute_coherence_width(self, wave_type: WaveType) -> float:
        """Compute the coherence width of the propagated beam [Andrews/Phillips, 2005, derived from Eq 12.27, Eq 6.132].

        Parameters
        ----------
        wave_type : WaveType
            Wavefront model: PLANE, SPHERICAL, or GAUSSIAN.

        Returns
        -------
        float
            The coherence width in meters.
        """
        k = 2 * np.pi / self.transmitter_station.transmitter.wavelength
        length = self.channel_length_m

        radius_of_curvature_in, waist_radius_in = self._wave_parameters_in(
            wave_type, length
        )
        theta_out, lambda_out = output_plane_parameters(
            length,
            self.transmitter_station.transmitter.wavelength,
            waist_radius_in,
            radius_of_curvature_in,
        )

        if length == 0:
            coherence_width = np.inf
        else:
            coherence_width_plane = (
                0.423 * k**2 * self.atmospheric_channel.cn2_profile * length
            ) ** (-3 / 5)
            alpha = (
                (1 - theta_out ** (8 / 3)) / (1 - theta_out)
                if theta_out >= 0
                else (1 + np.abs(theta_out) ** (8 / 3)) / (1 - theta_out)
            )
            coherence_width = (8 / (3 * (alpha + 0.618 * lambda_out ** (11 / 6)))) ** (
                3 / 5
            ) * coherence_width_plane
        return coherence_width

    def compute_wandering_variance(self, wave_type: WaveType) -> float:
        """Compute the beam wandering variance for a free-space optical channel.

        Parameters
        ----------
        wave_type : WaveType
            Type of the wave: PLANE, SPHERICAL OR GAUSSIAN.

        Returns
        -------
        float
            The beam wandering variance in square meters.
        """
        length = self.channel_length_m

        radius_of_curvature_in, waist_radius_in = self._wave_parameters_in(
            wave_type, length
        )
        theta_in, lambda_in = input_plane_parameters(
            length,
            self.transmitter_station.transmitter.wavelength,
            waist_radius_in,
            radius_of_curvature_in,
        )
        theta_bar_in = 1 - theta_in
        rytov_var = self.compute_rytov_variance(wave_type)
        tx_waist = self.transmitter_station.transmitter.waist_radius

        integrand = lambda xi: xi**2 / (
            (theta_in + theta_bar_in * xi) ** 2
            + 1.63 * rytov_var ** (6 / 5) * lambda_in * (1 - xi) ** (16 / 5)
        ) ** (1 / 6)
        integral = quad(integrand, 0, 1)[0]
        wandering_variance = (
            7.25
            * self.atmospheric_channel.cn2_profile
            * compute_sec(self.zenith_angle_deg) ** 3
            * length**3
            * tx_waist ** (-1 / 3)
            * integral
        )

        return wandering_variance


class DownlinkChannel(FreeSpaceChannel):
    """A downlink free-space optical communication channel."""

    def __init__(
        self,
        transmitter_station: TransmitterStation,
        receiver_station: ReceiverStation,
        atmospheric_channel: Atmosphere,
        zenith_angle_deg: float | None = None,
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
        super().__init__(
            transmitter_station, receiver_station, atmospheric_channel, zenith_angle_deg
        )

    def compute_rytov_variance(self) -> float:
        """Compute rytov variance of a spherical wave for a downlink channel.

        Returns
        -------
        rytov_var : float
            Rytov variance for spherical wave propagating in the downlink channel.
        """
        rytov_var = super().compute_rytov_variance(
            link_type="downlink", wave_type="spherical"
        )
        return rytov_var

    def compute_coherence_width(self) -> float:
        """Compute coherence width for a downlink channel.

        Returns
        -------
        coherence_width : float
            Coherence width for spherical wave propagating in the downlink channel.
        """
        coherence_width = super().compute_coherence_width(
            link_type="downlink", wave_type="spherical"
        )
        return coherence_width

    def compute_channel_losses(self) -> float:
        pass

    def draw_channel_pdf_sample(self) -> float:
        """Draw a random sample from the channel loss distribution."""
        pass


class UplinkChannel(FreeSpaceChannel):
    """An uplink free-space optical communication channel."""

    def __init__(
        self,
        transmitter_station: TransmitterStation,
        receiver_station: ReceiverStation,
        atmospheric_channel: Atmosphere,
    ) -> None:
        """Initialize the uplink channel with the given parameters.

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

    def compute_rytov_variance(self) -> float:
        """Compute rytov variance for an uplink channel.

        Returns
        -------
        rytov_var : float
            Rytov variance for spherical wave propagating in the uplink channel.
        """
        rytov_var = super().compute_rytov_variance(
            link_type="uplink", wave_type="spherical"
        )
        return rytov_var

    def compute_coherence_width(self) -> float:
        """Compute coherence width for an uplink channel.

        Returns
        -------
        coherence_width : float
            Coherence width for spherical wave propagating in the uplink channel.
        """
        coherence_width = super().compute_coherence_width(
            link_type="uplink", wave_type="spherical"
        )
        return coherence_width


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
