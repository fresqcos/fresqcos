"""Validate downlink coherence width against plane/spherical reference formulas."""

from fresqcos.telescopes import Transmitter, Receiver
from fresqcos.channels.stations import TransmitterStation, ReceiverStation
from fresqcos.channels.atmosphere import Atmosphere
from fresqcos.channels.cn2 import hufnagel_valley
from fresqcos.channels.channels import FreeSpaceChannel
from fresqcos.channels.geometry import compute_sec
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14


def wandering_variance_plane_horizontal(channel: FreeSpaceChannel) -> float:
    """Compute the beam wandering variance for a plane wave in a free-space optical channel.

    Returns
    -------
    float
        The beam wandering variance in square meters.
    """
    length = channel.channel_length_m
    tx_waist = channel.transmitter_station.transmitter.waist_radius

    wandering_variance = (
        2.42
        * channel.atmospheric_channel.cn2_profile
        * compute_sec(channel.zenith_angle_deg) ** 3
        * length**3
        * tx_waist ** (-1 / 3)
    )
    return wandering_variance


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
        latitude_deg=0,
        longitude_deg=0,
        altitude_km=0,
        transmitter=tx_telescope,
    )

    rx_station = ReceiverStation(
        name="Ground Station",
        latitude=0,
        longitude=0,
        altitude=gs_altitude,
        receiver=rx_telescope,
    )

    atmosphere = Atmosphere(
        cn2_profile=partial(hufnagel_valley, wind_speed_rms=wind_rms, reference_ground=reference_cn2),
        wind_speed=wind_rms,
        visibility=10,
    )

    free_space_channel = FreeSpaceChannel(
        transmitter_station=tx_station,
        receiver_station=rx_station,
        atmospheric_channel=atmosphere,
        zenith_angle_deg=zenith_angle,
    )

    wandering_variance_general_plane_list = []
    wandering_variance_general_spherical_list = []
    wandering_variance_general_gaussian_list = []
    wandering_variance_plane_list = []

    for i in range(len(altitude_vector)):

        free_space_channel.transmitter_station.altitude_km = altitude_vector[i]
    
        wandering_variance_general_plane = free_space_channel.compute_wandering_variance(
            link_type="horizontal", 
            wave_type="plane")
        wandering_variance_general_spherical = free_space_channel.compute_wandering_variance(
            link_type="horizontal",
            wave_type="spherical")
        wandering_variance_general_gaussian = free_space_channel.compute_wandering_variance(
            link_type="horizontal", 
            wave_type="gaussian")
        wandering_variance_plane_horizontal = wandering_variance_plane_horizontal(free_space_channel)
        wandering_variance_general_plane_list.append(wandering_variance_general_plane)
        wandering_variance_general_spherical_list.append(wandering_variance_general_spherical)
        wandering_variance_general_gaussian_list.append(wandering_variance_general_gaussian)
        wandering_variance_plane_list.append(wandering_variance_plane_horizontal)


plt.figure()
plt.plot(altitude_vector, wandering_variance_general_plane_list, label="Plane Wave (General)")
plt.plot(altitude_vector, wandering_variance_general_spherical_list, label="Spherical Wave (General)")
plt.plot(altitude_vector, wandering_variance_general_gaussian_list, label="Gaussian Beam (General)")
plt.gca().set_prop_cycle(None)
plt.plot(altitude_vector, wandering_variance_plane_list, "o", label="Plane Wave (Reference)")
plt.xlabel("Platform altitude (km)")
plt.ylabel("Wandering Variance (m^2)")
plt.title("Wandering Variance in Horizontal Channel")
plt.legend()
plt.grid()
plt.tight_layout()

plt.show()