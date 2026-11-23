"""Validate collection efficiency."""

from fresqcos.telescopes import Transmitter, Receiver
from fresqcos.channels.stations import TransmitterStation, ReceiverStation
from fresqcos.channels.atmosphere import Atmosphere
from fresqcos.channels.cn2 import hufnagel_valley
from fresqcos.channels.channels import (
    SatToAerialChannel,
)
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["font.size"] = 14


if __name__ == "__main__":
    wvln = 1550e-9
    tx_waist_radius = 0.1
    tx_obscuration_ratio = 0
    tx_internal_loss = 0.1
    rx_aperture = 0.4
    rx_obscuration_ratio = 0.4
    rx_internal_loss = 0.1
    zenith_angle_deg = 0
    wind_rms = 21
    reference_cn2 = 1.7e-14
    altitude_aerial = 0
    altitude_satellite = 20

    tx_telescope = Transmitter(
        wavelength=wvln,
        waist_radius=tx_waist_radius,
        obscuration_ratio=tx_obscuration_ratio,
        internal_loss=tx_internal_loss,
        pointing_error_x=0e-6,
        pointing_error_y=0e-6,
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
        altitude_km=altitude_satellite,
    )

    aerial_rx_station = ReceiverStation(
        name="AerialRx",
        receiver=rx_telescope,
        altitude_km=altitude_aerial,
    )

    atmosphere = Atmosphere(
        cn2_profile=partial(
            hufnagel_valley, wind_speed_rms=wind_rms, reference_ground=reference_cn2
        ),
        wind_speed=wind_rms,
        visibility=10,
    )

    downlink_channel = SatToAerialChannel(
        transmitter_station=satellite_tx_station,
        receiver_station=aerial_rx_station,
        atmospheric_channel=atmosphere,
        zenith_angle_deg=zenith_angle_deg,
    )

    eta = np.linspace(1e-7, 1, 1000)
    efficiency_pdf = downlink_channel.compute_collection_efficiency_pdf(eta)
    efficiency_pdf_annular = downlink_channel.compute_collection_efficiency_pdf_annular(eta)


plt.figure()
plt.plot(eta, efficiency_pdf, label="Circular Aperture")
plt.plot(eta, efficiency_pdf_annular, label="Annular Aperture", linestyle="--")
plt.xlabel("Collection Efficiency")
plt.ylabel("Probability Density")
plt.grid()
plt.legend()
plt.tight_layout()

plt.show()
