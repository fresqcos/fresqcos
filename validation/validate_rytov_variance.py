"""Validate rytov variance against plane/spherical reference formulas."""

from fresqcos.telescopes import Transmitter, Receiver
from fresqcos.channels.stations import TransmitterStation, ReceiverStation
from fresqcos.channels.atmosphere import Atmosphere
from fresqcos.channels.cn2 import hufnagel_valley
from fresqcos.channels.channels import DownlinkChannel, SatToGroundChannel
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
    receiver_alt = channel.receiver_altitude_m
    transmitter_alt = channel.transmitter_altitude_m
    k = 2 * np.pi / channel.transmitter_station.transmitter.wavelength
    scale = 1e14  # Scale factor to avoid numerical issues in integration
    integrand = (
        lambda h: channel.atmospheric_channel.cn2_profile(h)
        * scale
        * (h - receiver_alt) ** (5 / 6)
    )
    rytov_var = (
        2.25
        * k ** (7 / 6)
        * compute_sec(channel.zenith_angle_deg) ** (11 / 6)
        * quad(integrand, receiver_alt, transmitter_alt)[0]
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
    receiver_alt = channel.receiver_altitude_m
    transmitter_alt = channel.transmitter_altitude_m
    k = 2 * np.pi / channel.transmitter_station.transmitter.wavelength
    scale = 1e14  # Scale factor to avoid numerical issues in integration
    integrand = (
        lambda h: channel.atmospheric_channel.cn2_profile(h)
        * scale
        * (h - receiver_alt) ** (5 / 6)
        * ((transmitter_alt - h) / (transmitter_alt - receiver_alt)) ** (5 / 6)
    )
    rytov_var = (
        2.25
        * k ** (7 / 6)
        * compute_sec(channel.zenith_angle_deg) ** (11 / 6)
        * quad(integrand, receiver_alt, transmitter_alt)[0]
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
    gs_altitude = 0
    zenith_angle = 80
    wind_rms = 10
    reference_cn2 = 1.7e-14
    altitude_vector = np.linspace(5, platform_altitude, 50)

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

    tx_station = TransmitterStation(
        name="Satellite",
        transmitter=tx_telescope,
        altitude_km=altitude_vector[0],
    )

    rx_station = ReceiverStation(
        name="Ground Station",
        receiver=rx_telescope,
        altitude_km=gs_altitude,
    )

    atmosphere = Atmosphere(
        cn2_profile=partial(
            hufnagel_valley, wind_speed_rms=wind_rms, reference_ground=reference_cn2
        ),
        wind_speed=wind_rms,
        visibility=10,
    )

    downlink_channel = SatToGroundChannel(
        transmitter_station=tx_station,
        receiver_station=rx_station,
        atmospheric_channel=atmosphere,
        zenith_angle_deg=zenith_angle,
    )

    rytov_variance_general_plane_list = []
    rytov_variance_general_spherical_list = []
    rytov_variance_general_gaussian_list = []
    rytov_variance_plane_list = []
    rytov_variance_width_spherical_list = []

    for i in range(len(altitude_vector)):

        rytov_variance_general_plane = downlink_channel.compute_rytov_variance(
            wave_type="plane"
        )
        rytov_variance_general_spherical = downlink_channel.compute_rytov_variance(
            wave_type="spherical"
        )
        rytov_variance_general_gaussian = downlink_channel.compute_rytov_variance(
            wave_type="gaussian"
        )
        rytov_variance_plane = rytov_variance_plane_downlink(downlink_channel)
        rytov_variance_spherical = rytov_variance_spherical_downlink(downlink_channel)
        rytov_variance_general_plane_list.append(rytov_variance_general_plane)
        rytov_variance_general_spherical_list.append(rytov_variance_general_spherical)
        rytov_variance_general_gaussian_list.append(rytov_variance_general_gaussian)
        rytov_variance_plane_list.append(rytov_variance_plane)
        rytov_variance_width_spherical_list.append(rytov_variance_spherical)

        if i < len(altitude_vector) - 1:
            downlink_channel.transmitter_station.altitude_km = altitude_vector[i + 1]


plt.figure()
plt.plot(
    altitude_vector, rytov_variance_general_plane_list, label="Plane Wave (General)"
)
plt.plot(
    altitude_vector,
    rytov_variance_general_spherical_list,
    label="Spherical Wave (General)",
)
plt.plot(
    altitude_vector,
    rytov_variance_general_gaussian_list,
    label="Gaussian Beam (General)",
)
plt.gca().set_prop_cycle(None)
plt.plot(
    altitude_vector, rytov_variance_plane_list, "o", label="Plane Wave (Reference)"
)
plt.plot(
    altitude_vector,
    rytov_variance_width_spherical_list,
    "o",
    label="Spherical Wave (Reference)",
)
plt.xlabel("Platform altitude (km)")
plt.ylabel("Rytov variance")
plt.title("Rytov variance in Downlink Channel")
plt.legend()
plt.grid()
plt.tight_layout()

plt.show()
