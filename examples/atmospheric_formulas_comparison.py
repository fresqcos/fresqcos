from fresqcos.telescopes import Transmitter, Receiver
from fresqcos.channels.stations import TransmitterStation, ReceiverStation
from fresqcos.channels.atmosphere import Atmosphere
from fresqcos.channels.cn2 import hufnagel_valley
from fresqcos.channels.channels import DownlinkChannel
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
    altitude_vector = np.linspace(gs_altitude, platform_altitude, 2000)
    
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

    rx_station = ReceiverStation(
        name="Ground Station",
        latitude=0,
        longitude=0,
        altitude=gs_altitude,
        receiver=rx_telescope,
    )

    rytov_var_plane_list = []
    rytov_var_spherical_list = []

    for i in range(len(altitude_vector)):

        tx_station = TransmitterStation(
            name="Satellite",
            latitude=0,
            longitude=0,
            altitude=altitude_vector[i],
            transmitter=tx_telescope,
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

        rytov_var_plane = down_channel._compute_rytov_variance_plane(zenith_angle)
        rytov_var_spherical = down_channel._compute_rytov_variance_spherical(zenith_angle)
        rytov_var_plane_list.append(rytov_var_plane)
        rytov_var_spherical_list.append(rytov_var_spherical)

cn2_computed = hufnagel_valley(altitude_vector*1e3, wind_speed_rms=wind_rms, reference_ground=reference_cn2)

save_data = np.column_stack((altitude_vector, rytov_var_plane_list, rytov_var_spherical_list, cn2_computed))   

plt.figure()
plt.plot(altitude_vector, rytov_var_plane_list, label="Plane Wave")
plt.plot(altitude_vector, rytov_var_spherical_list, label="Spherical Wave")
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

