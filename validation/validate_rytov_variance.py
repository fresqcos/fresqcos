"""Validate rytov variance against plane/spherical reference formulas."""

from fresqcos.telescopes import Transmitter, Receiver
from fresqcos.channels.stations import TransmitterStation, ReceiverStation
from fresqcos.channels.atmosphere import Atmosphere
from fresqcos.channels.cn2 import hufnagel_valley
from fresqcos.channels.channels import (
    DownlinkChannel,
    GroundToSatChannel,
    SatToGroundChannel,
    UplinkChannel,
)
from fresqcos.channels.geometry import compute_sec
from functools import partial
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["font.size"] = 14


def rytov_variance_plane_downlink(channel: DownlinkChannel) -> float:
    """Compute rytov variance of a plane wave for a
    downlink channel [Andrews/Phillips, 2005, from Eqs 12.92 and 12.37].

    Parameters
    ----------
    channel : DownlinkChannel
        Downlink channel object.

    Returns
    -------
    rytov_var : float
        Rytov variance for plane wave propagating in the downlink channel.
    """
    lower_alt = channel.lower_altitude_m
    higher_alt = channel.higher_altitude_m
    k = 2 * np.pi / channel.transmitter_station.transmitter.wavelength
    scale = 1e14  # Scale factor to avoid numerical issues in integration
    integrand = (
        lambda h: channel.atmospheric_channel.cn2_profile(h)
        * scale
        * (h - lower_alt) ** (5 / 6)
    )
    rytov_var = (
        2.25
        * k ** (7 / 6)
        * compute_sec(channel.zenith_angle_deg) ** (11 / 6)
        * quad(integrand, lower_alt, higher_alt)[0]
        / scale
    )
    return rytov_var


def rytov_variance_plane_uplink(channel: UplinkChannel) -> float:
    """Compute rytov variance of a plane wave for a
    uplink channel.

    Parameters
    ----------
    channel : UplinkChannel
        Uplink channel object.

    Returns
    -------
    rytov_var :float
        Rytov variance for plane wave propagating in the uplink channel.
    """
    lower_alt = channel.lower_altitude_m
    higher_alt = channel.higher_altitude_m
    k = 2 * np.pi / channel.transmitter_station.transmitter.wavelength
    scale = 1e14  # Scale factor to avoid numerical issues in integration
    integrand = (
        lambda h: channel.atmospheric_channel.cn2_profile(h)
        * scale
        * (higher_alt - h) ** (5 / 6)
    )
    rytov_var = (
        2.25
        * k ** (7 / 6)
        * compute_sec(channel.zenith_angle_deg) ** (11 / 6)
        * quad(integrand, lower_alt, higher_alt)[0]
        / scale
    )
    return rytov_var


def rytov_variance_spherical_downlink(channel: DownlinkChannel) -> float:
    """Compute rytov variance of a spherical wave for a
    downlink channel [Andrews/Phillips, 2005, from Eqs 12.92 and 12.37].

    Parameters
    ----------
    channel : DownlinkChannel
        Downlink channel object.

    Returns
    -------
    rytov_var :float
        Rytov variance for spherical wave propagating in the downlink channel.
    """
    lower_alt = channel.lower_altitude_m
    higher_alt = channel.higher_altitude_m
    k = 2 * np.pi / channel.transmitter_station.transmitter.wavelength
    scale = 1e14  # Scale factor to avoid numerical issues in integration
    integrand = (
        lambda h: channel.atmospheric_channel.cn2_profile(h)
        * scale
        * (h - lower_alt) ** (5 / 6)
        * ((higher_alt - h) / (higher_alt - lower_alt)) ** (5 / 6)
    )
    rytov_var = (
        2.25
        * k ** (7 / 6)
        * compute_sec(channel.zenith_angle_deg) ** (11 / 6)
        * quad(integrand, lower_alt, higher_alt)[0]
        / scale
    )
    return rytov_var


if __name__ == "__main__":
    wvln = 1550e-9
    tx_waist_radius = 0.1
    tx_obscuration_ratio = 0
    tx_internal_loss = 0.1
    tx_pointing_error = 0
    rx_aperture = 0.4
    rx_obscuration_ratio = 0.3
    rx_internal_loss = 0.1
    platform_altitude = 400
    zenith_angle_deg = 80
    wind_rms = 21
    reference_cn2 = 1.7e-14
    altitude_ground = 0
    altitude_vector = np.linspace(5, platform_altitude, 100)

    tx_telescope = Transmitter(
        wavelength=wvln,
        waist_radius=tx_waist_radius,
        obscuration_ratio=tx_obscuration_ratio,
        internal_loss=tx_internal_loss,
        pointing_error=tx_pointing_error,
    )

    rx_telescope = Receiver(
        wavelength=wvln,
        aperture=rx_aperture,
        obscuration_ratio=rx_obscuration_ratio,
        internal_loss=rx_internal_loss,
    )

    satellite_tx_station = TransmitterStation(
        name="SatelliteTx",
        transmitter=tx_telescope,
        altitude_km=altitude_vector[0],
    )

    ground_rx_station = ReceiverStation(
        name="GroundRx",
        receiver=rx_telescope,
        altitude_km=altitude_ground,
    )

    ground_tx_station = TransmitterStation(
        name="GroundTx",
        transmitter=tx_telescope,
        altitude_km=altitude_ground,
    )

    satellite_rx_station = ReceiverStation(
        name="SatelliteRx",
        receiver=rx_telescope,
        altitude_km=altitude_vector[0],
    )

    atmosphere = Atmosphere(
        cn2_profile=partial(
            hufnagel_valley, wind_speed_rms=wind_rms, reference_ground=reference_cn2
        ),
        wind_speed=wind_rms,
        visibility=10,
    )

    downlink_channel = SatToGroundChannel(
        transmitter_station=satellite_tx_station,
        receiver_station=ground_rx_station,
        atmospheric_channel=atmosphere,
        zenith_angle_deg=zenith_angle_deg,
    )

    uplink_channel = GroundToSatChannel(
        transmitter_station=ground_tx_station,
        receiver_station=satellite_rx_station,
        atmospheric_channel=atmosphere,
        zenith_angle_deg=zenith_angle_deg,
    )

    rytov_variance_general_downlink_plane_list = []
    rytov_variance_general_downlink_spherical_list = []
    rytov_variance_general_downlink_gaussian_list = []
    rytov_variance_downlink_plane_list = []
    rytov_variance_width_downlink_spherical_list = []

    rytov_variance_general_uplink_plane_list = []
    rytov_variance_general_uplink_spherical_list = []
    rytov_variance_general_uplink_gaussian_list = []
    rytov_variance_uplink_plane_list = []
    rytov_variance_uplink_spherical_list = []

    for i in range(len(altitude_vector)):

        rytov_variance_general_downlink_plane = downlink_channel.compute_rytov_variance(
            wave_type="plane"
        )
        rytov_variance_general_downlink_spherical = (
            downlink_channel.compute_rytov_variance(wave_type="spherical")
        )
        rytov_variance_general_downlink_gaussian = (
            downlink_channel.compute_rytov_variance(wave_type="gaussian")
        )
        rytov_variance_downlink_plane = rytov_variance_plane_downlink(downlink_channel)
        rytov_variance_downlink_spherical = rytov_variance_spherical_downlink(
            downlink_channel
        )
        rytov_variance_general_downlink_plane_list.append(
            rytov_variance_general_downlink_plane
        )
        rytov_variance_general_downlink_spherical_list.append(
            rytov_variance_general_downlink_spherical
        )
        rytov_variance_general_downlink_gaussian_list.append(
            rytov_variance_general_downlink_gaussian
        )
        rytov_variance_downlink_plane_list.append(rytov_variance_downlink_plane)
        rytov_variance_width_downlink_spherical_list.append(
            rytov_variance_downlink_spherical
        )

        rytov_variance_general_uplink_plane = uplink_channel.compute_rytov_variance(
            wave_type="plane"
        )
        rytov_variance_general_uplink_spherical = uplink_channel.compute_rytov_variance(
            wave_type="spherical"
        )
        rytov_variance_general_uplink_gaussian = uplink_channel.compute_rytov_variance(
            wave_type="gaussian"
        )
        rytov_variance_uplink_plane = rytov_variance_plane_uplink(uplink_channel)
        rytov_variance_uplink_spherical = rytov_variance_downlink_spherical
        rytov_variance_general_uplink_plane_list.append(
            rytov_variance_general_uplink_plane
        )
        rytov_variance_general_uplink_spherical_list.append(
            rytov_variance_general_uplink_spherical
        )
        rytov_variance_general_uplink_gaussian_list.append(
            rytov_variance_general_uplink_gaussian
        )
        rytov_variance_uplink_plane_list.append(rytov_variance_uplink_plane)
        rytov_variance_uplink_spherical_list.append(rytov_variance_uplink_spherical)

        if i < len(altitude_vector) - 1:
            downlink_channel.transmitter_station.altitude_km = altitude_vector[i + 1]
            uplink_channel.receiver_station.altitude_km = altitude_vector[i + 1]


plt.figure()
plt.plot(
    altitude_vector,
    rytov_variance_general_downlink_plane_list,
    label="Plane Wave (General)",
)
plt.plot(
    altitude_vector,
    rytov_variance_general_downlink_spherical_list,
    label="Spherical Wave (General)",
)
plt.plot(
    altitude_vector,
    rytov_variance_general_downlink_gaussian_list,
    label="Gaussian Beam (General)",
)
plt.gca().set_prop_cycle(None)
plt.plot(
    altitude_vector,
    rytov_variance_downlink_plane_list,
    "o",
    label="Plane Wave (Reference)",
)
plt.plot(
    altitude_vector,
    rytov_variance_width_downlink_spherical_list,
    "o",
    label="Spherical Wave (Reference)",
)
plt.xlabel("Platform altitude (km)")
plt.ylabel("Rytov variance")
plt.title("Rytov variance in Downlink Channel")
plt.legend()
plt.grid()
plt.tight_layout()

plt.figure()
plt.semilogy(
    altitude_vector,
    rytov_variance_general_uplink_plane_list,
    label="Plane Wave (General)",
)
plt.semilogy(
    altitude_vector,
    rytov_variance_general_uplink_spherical_list,
    label="Spherical Wave (General)",
)
plt.semilogy(
    altitude_vector,
    rytov_variance_general_uplink_gaussian_list,
    label="Gaussian Beam (General)",
)
plt.gca().set_prop_cycle(None)
plt.plot(
    altitude_vector,
    rytov_variance_uplink_plane_list,
    "o",
    label="Plane Wave (Reference)",
)
plt.semilogy(
    altitude_vector,
    rytov_variance_uplink_spherical_list,
    "o",
    label="Spherical Wave (Reference)",
)
plt.xlabel("Platform altitude (km)")
plt.ylabel("Rytov variance")
plt.title("Rytov variance in Uplink Channel")
plt.legend()
plt.grid()
plt.tight_layout()

plt.show()
