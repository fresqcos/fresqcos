from fresqcos.telescopes import Transmitter, Receiver
from fresqcos.channels.stations import TransmitterStation, ReceiverStation
from fresqcos.channels.atmosphere import Atmosphere
from fresqcos.channels.cn2 import hufnagel_valley
from fresqcos.channels.channels import DownlinkChannel, FreeSpaceChannel
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14


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
    altitude_vector = np.linspace(gs_altitude, platform_altitude, 1000)
    
    
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
        latitude=0,
        longitude=0,
        altitude=0,
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

    down_channel = DownlinkChannel(
        transmitter_station=tx_station,
        receiver_station=rx_station,
        atmospheric_channel=atmosphere,
    )

    free_space_channel = FreeSpaceChannel(
        transmitter_station=tx_station,
        receiver_station=rx_station,
        atmospheric_channel=atmosphere,
    )

    rytov_var_spherical_list = []
    rytov_var_plane_parent_list = []
    rytov_var_spherical_parent_list = []

    for i in range(len(altitude_vector)):

        down_channel.transmitter_station.altitude = altitude_vector[i]
        free_space_channel.transmitter_station.altitude = altitude_vector[i]

        rytov_var_spherical = down_channel.compute_rytov_variance(zenith_angle)
        rytov_var_plane_parent = free_space_channel.compute_rytov_variance(zenith_angle, link_type="downlink", wave_type="plane")
        rytov_var_spherical_parent = free_space_channel.compute_rytov_variance(zenith_angle, link_type="downlink", wave_type="spherical")
        rytov_var_spherical_list.append(rytov_var_spherical)
        rytov_var_plane_parent_list.append(rytov_var_plane_parent)
        rytov_var_spherical_parent_list.append(rytov_var_spherical_parent)

cn2_computed = hufnagel_valley(altitude_vector*1e3, wind_speed_rms=wind_rms, reference_ground=reference_cn2)

# save_data = np.column_stack((altitude_vector, rytov_var_plane_list, rytov_var_spherical_list, cn2_computed))   

plt.figure()
plt.plot(altitude_vector, rytov_var_spherical_list, label="Spherical Wave")
plt.plot(altitude_vector, rytov_var_plane_parent_list, label="Plane Wave (Parent)")
plt.plot(altitude_vector, rytov_var_spherical_parent_list, "o", label="Spherical Wave (Parent)")
plt.xlabel("Platform altitude (km)")
plt.ylabel("Rytov Variance")
plt.title("Rytov variance in downlink channel")
plt.legend()
plt.grid()
plt.tight_layout()

plt.figure()
plt.semilogx(cn2_computed, altitude_vector)
plt.xlabel("$C_n^2$ (m$^{-2/3}$)")
plt.ylabel("Altitude (km)")
plt.title("Hufnagel-Valley profile")
plt.grid()
plt.tight_layout()

plt.show()

