"""Validate wandering variance against plane/spherical reference formulas."""

from fresqcos.telescopes import Transmitter, Receiver
from fresqcos.channels.stations import TransmitterStation, ReceiverStation
from fresqcos.channels.atmosphere import Atmosphere
from fresqcos.channels.cn2 import hufnagel_valley
from fresqcos.channels.channels import (
    HorizontalChannel,
    AerialToAerialChannel,
    UplinkChannel,
    GroundToSatChannel,
    SatToGroundChannel,
)
from fresqcos.channels.geometry import compute_sec
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["font.size"] = 14


def wandering_variance_plane_horizontal(channel: HorizontalChannel) -> float:
    """Compute the beam wandering variance for a plane wave in a horizontal free-space optical channel.

    Returns
    -------
    float
        The beam wandering variance in square meters.
    """
    length = channel.channel_length_m
    tx_waist = channel.transmitter_station.transmitter.waist_radius

    wandering_variance = 2.42 * channel.cn2 * length**3 * tx_waist ** (-1 / 3)
    return wandering_variance


def wandering_variance_plane_uplink(channel: UplinkChannel) -> float:
    """Compute the beam wandering variance for a plane wave in an uplink free-space optical channel.

    Returns
    -------
    float
        The beam wandering variance in square meters.
    """
    receiver_alt = channel.receiver_altitude_m
    transmitter_alt = channel.transmitter_altitude_m
    wvln = channel.transmitter_station.transmitter.wavelength
    tx_waist = channel.transmitter_station.transmitter.waist_radius
    coherence_width = channel.compute_coherence_width(wave_type="plane")

    wandering_variance = (
        0.73
        * (receiver_alt - transmitter_alt)
        * compute_sec(channel.zenith_angle_deg)
        * (wvln / (2 * tx_waist))
        * (2 * tx_waist / coherence_width) ** (5 / 6)
    ) ** 2
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
    altitude_aerial = 20
    altitude_ground = 0
    altitude_satellite = 400
    zenith_angle_deg = 70
    wind_rms = 10
    reference_cn2 = 1.7e-14
    length_vector = np.linspace(1, 20, 50)
    altitude_vector = np.linspace(1, altitude_satellite - 1, 50)

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

    aerial_tx_station = TransmitterStation(
        name="AerialTx",
        transmitter=tx_telescope,
        altitude_km=altitude_aerial,
    )

    aerial_rx_station = ReceiverStation(
        name="AerialRx",
        receiver=rx_telescope,
        altitude_km=altitude_aerial,
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

    atmosphere = Atmosphere(
        cn2_profile=partial(
            hufnagel_valley, wind_speed_rms=wind_rms, reference_ground=reference_cn2
        ),
        wind_speed=wind_rms,
        visibility=10,
    )

    horizontal_channel = AerialToAerialChannel(
        transmitter_station=aerial_tx_station,
        receiver_station=aerial_rx_station,
        atmospheric_channel=atmosphere,
        length_km=length_vector[0],
    )

    uplink_channel = GroundToSatChannel(
        transmitter_station=ground_tx_station,
        receiver_station=satellite_rx_station,
        atmospheric_channel=atmosphere,
        zenith_angle_deg=zenith_angle_deg,
    )

    downlink_channel = SatToGroundChannel(
        transmitter_station=satellite_tx_station,
        receiver_station=ground_rx_station,
        atmospheric_channel=atmosphere,
        zenith_angle_deg=zenith_angle_deg,
    )

    wandering_variance_general_horizontal_plane_list = []
    wandering_variance_horizontal_plane_list = []

    for i in range(len(length_vector)):

        wandering_variance_general_horizontal_plane = (
            horizontal_channel.compute_wandering_variance(wave_type="plane")
        )
        wandering_variance_horizontal_plane = wandering_variance_plane_horizontal(
            horizontal_channel
        )
        wandering_variance_general_horizontal_plane_list.append(
            wandering_variance_general_horizontal_plane
        )
        wandering_variance_horizontal_plane_list.append(
            wandering_variance_horizontal_plane
        )

        if i < len(length_vector) - 1:
            horizontal_channel.length_km = length_vector[i + 1]

    wandering_variance_general_uplink_plane_list = []
    wandering_variance_uplink_plane_list = []
    wandering_variance_general_downlink_plane_list = []

    for i in range(len(altitude_vector)):

        wandering_variance_general_uplink_plane = (
            uplink_channel.compute_wandering_variance(wave_type="plane")
        )
        wandering_variance_uplink_plane = wandering_variance_plane_uplink(
            uplink_channel
        )
        wandering_variance_general_uplink_plane_list.append(
            wandering_variance_general_uplink_plane
        )
        wandering_variance_uplink_plane_list.append(wandering_variance_uplink_plane)

        wandering_variance_general_downlink_plane = (
            downlink_channel.compute_wandering_variance(wave_type="plane")
        )
        wandering_variance_general_downlink_plane_list.append(
            wandering_variance_general_downlink_plane
        )

        if i < len(altitude_vector) - 1:
            uplink_channel.receiver_station.altitude_km = altitude_vector[i + 1]
            downlink_channel.transmitter_station.altitude_km = altitude_vector[i + 1]


plt.figure()
plt.plot(
    length_vector,
    wandering_variance_general_horizontal_plane_list,
    label="Plane Wave (General)",
)
plt.gca().set_prop_cycle(None)
plt.plot(
    length_vector,
    wandering_variance_horizontal_plane_list,
    "o",
    label="Plane Wave (Reference)",
)
plt.xlabel("Channel Length (km)")
plt.ylabel("Wandering Variance (m^2)")
plt.title("Wandering Variance in Horizontal Channel")
plt.legend()
plt.grid()
plt.tight_layout()

plt.figure()
plt.plot(
    altitude_vector,
    wandering_variance_general_uplink_plane_list,
    label="Plane Wave (General)",
)
plt.gca().set_prop_cycle(None)
plt.plot(
    altitude_vector,
    wandering_variance_uplink_plane_list,
    "o",
    label="Plane Wave (Reference)",
)
plt.xlabel("Platform altitude (km)")
plt.ylabel("Wandering Variance (m^2)")
plt.title("Wandering Variance in Uplink Channel")
plt.legend()
plt.grid()
plt.tight_layout()

plt.figure()
plt.plot(
    altitude_vector,
    wandering_variance_general_downlink_plane_list,
    label="Plane Wave (General)",
)
plt.xlabel("Platform altitude (km)")
plt.ylabel("Wandering Variance (m^2)")
plt.title("Wandering Variance in Downlink Channel")
plt.legend()
plt.grid()
plt.tight_layout()

plt.show()
