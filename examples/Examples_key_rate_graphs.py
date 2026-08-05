""" I put where the values are coming from and the results in the section 'Example' of the latex doc"""

from matplotlib import pyplot as plt
import numpy as np

from fresqcos.sources.sources import Attenuated_Laser, Single_Photon_Source, Multiplexed_Heralded_Photon_Source, Symmetric_Multiplexed_Heralded_Photon_Source, Asymmetric_Multiplexed_Heralded_Photon_Source, Entangled_PDC_Source

from fresqcos.detectors.detectors import Threshold_detector

from fresqcos.protocols.protocols import BB84, BBM92

from fresqcos.receiver import Receiver

from fresqcos.channel import Channel

## Plot

def key_rate_distance_km_bb84(*, min, max, values_number, source: Source, detector: Detector, channel: FiberChannel, receiver: Receiver, correction_efficiency: float, title: str):
    x_values = np.linspace(min, max, values_number)
    y1_values = []
    for x in x_values:
        channel.distance_km = x
        protocol = BB84(source=source, detector=detector, channel=channel, receiver=receiver, correction_efficiency=correction_efficiency)
        y1 = protocol.key_rate_decoy_state_inf_key()
        y1_values.append(y1)

    plt.plot(x_values, y1_values, color='blue', label="With active decoy state")
    plt.yscale('log')
    plt.xlabel("distance_km in km")
    plt.ylabel("Key rate in bpp")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def key_rate_distance_km_pulsed_bbm92(*, min, max, values_number, source: Source, detector1: Detector, detector2: Detector, channel_1: FiberChannel, channel_2: FiberChannel, receiver1: Receiver, receiver2: Receiver, correction_efficiency: float, title: str):
    x_values = np.linspace(min, max, values_number)
    x_axis = []
    y1_values = []
    for x in x_values:
        channel_1.distance_km = x
        if channel_2 is None:
            ch2 = channel_1
        else:
            ch2 = channel_2

        x_axis.append(channel_1.distance_km+ch_2.distance_km)

        protocol = Pulsed_BBM92(source=source, detector1=detector1, channel_1=channel_1, channel_2 = ch_2, receiver1=receiver1, correction_efficiency=correction_efficiency, detector2=detector2, receiver2=receiver2)
        y1 = protocol.key_rate()
        y1_values.append(y1)

    plt.plot(x_values, y1_values, color='red')
    plt.yscale('log')
    plt.xlabel("distance_km in km")
    plt.ylabel("Key rate in bpp")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def key_rate_loss_bb84(*, min, max, values_number, source: Source, detector: Detector, channel: FiberChannel, receiver: Receiver, correction_efficiency: float, title: str):
    x_values = np.linspace(min, max, values_number)
    horiz_axis = np.linspace(min * channel.loss_per_km, max * channel.loss_per_km, values_number)
    y1_values = []
    for x in x_values:
        channel.distance_km = x
        protocol = BB84(source=source, detector=detector, channel=channel, receiver=receiver, correction_efficiency=correction_efficiency)
        y1 = protocol.key_rate_decoy_state_inf_key()
        y1_values.append(y1)

    plt.plot(horiz_axis, y1_values, color='blue', label="With active decoy state")
    plt.yscale('log')
    plt.xlabel("Loss in dB")
    plt.ylabel("Key rate in bpp")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def key_rate_loss_pulsed_bbm92(*, min, max, values_number, source: Source, detector1: Detector, channel_1: FiberChannel, receiver1: Receiver, correction_efficiency: float, title: str, detector2: Optional[Detector] = None, channel_2: Optional[FiberChannel] = None, receiver2: Optional[Receiver] = None):
    x_values = np.linspace(min, max, values_number)
    horiz_axis = []
    y1_values = []

    for x in x_values:
        channel_1.distance_km = x
        if channel_2 is None:
            ch2 = channel_1
        else:
            ch2 = channel_2
        horiz_axis.append(channel_1.distance_km * channel_1.loss_per_km + ch2.distance_km * ch2.loss_per_km)

        y1_values.append(Pulsed_BBM92(source=source, detector1=detector1, detector2=detector2, channel_1=channel_1, channel_2=ch2, receiver1=receiver1, receiver2=receiver2, correction_efficiency=correction_efficiency).key_rate())

    plt.plot(horiz_axis, y1_values, color='red')
    plt.yscale('log')
    plt.xlabel("Loss in dB")
    plt.ylabel("Key rate in bits/s")
    plt.title(title)
    plt.grid(True)
    plt.show()

def key_rate_loss_bbm92_continuous(*, min, max, values_number, source: Source, detector1: Detector, channel_1: FiberChannel, receiver1: Receiver, correction_efficiency: float, title: str, coincidence_time: float, detector2: Optional[Detector] = None, channel_2: Optional[FiberChannel] = None, receiver2: Optional[Receiver] = None):
    x_values = np.linspace(min, max, values_number)
    horiz_axis = []
    y1_values = []


    for x in x_values:
        channel_1.distance_km = x
        if channel_2 is None:
            ch2 = channel_1
        else:
            ch2 = channel_2

        protocol = BBM92_continuous_wave_pumped_source(source=source, detector1=detector1, detector2=detector2, channel_1=channel_1, channel_2=ch2, receiver1=receiver1, receiver2=receiver2, correction_efficiency=correction_efficiency, coincidence_time=coincidence_time)
        y1 = protocol.key_rate()
        y1_values.append(y1)
        horiz_axis.append(-10 * np.log10((protocol.heralding_efficiency_1_x() + protocol.heralding_efficiency_1_z()) / 2 * (protocol.heralding_efficiency_2_x() + protocol.heralding_efficiency_2_z()) / 2))

    plt.plot(horiz_axis, y1_values, color='red')
    plt.yscale('log')
    plt.xlabel("Loss in dB")
    plt.ylabel("Key rate in bits/s")
    plt.title(title)
    plt.grid(True)
    plt.show()


# Test 1:

source_1 = Attenuated_Laser(mean_photon_number=0.48, repetition_rate=0)

detector_1 = Threshold_detector(dark_count_rate=0.17, efficiency=5/100, time_window=10**(-5), after_pulsing=0) #Y_0 = 1.7*10**(-6)

receiver_1 = Receiver(transmittance=0.9)

FiberChannel_1 = FiberChannel(loss_per_km=0.21, distance_km=0, detection_error=0.033)

f_1 = 1.22

key_rate_distance_km_bb84(min=0, max=160, values_number=300, source=source_1, detector=detector_1, channel=FiberChannel_1, receiver=receiver_1, correction_efficiency=f_1, title="Evolution of the key rate with the distance_km for an attenuated laser")


# Test 2:

source_2 = Symmetric_Multiplexed_Heralded_Photon_Source(mean_photon_number=0.48, repetition_rate=0, sources_num=32, transmittance=0.5, efficiency=0.7)

detector_2 = Threshold_detector(dark_count_rate=20, efficiency=0.25, time_window=10**(-8), after_pulsing=0)

receiver_2 = Receiver(transmittance=1)

FiberChannel_2 = FiberChannel(loss_per_km=0.2, distance_km=0, detection_error=0.005)

f_2 = 1.05

key_rate_loss_bb84(min=0, max=275, values_number=300, source=source_2, detector=detector_2, channel=FiberChannel_2, receiver=receiver_2, correction_efficiency=f_2, title="Evolution of the key rate with the loss for SMHPS")


# Test 3:

source_3 = Asymmetric_Multiplexed_Heralded_Photon_Source(mean_photon_number=0.6, repetition_rate=0, sources_num=32, transmittance=0.5, efficiency=0.7)

detector_3 = Threshold_detector(dark_count_rate=20, efficiency=0.25, time_window=10**(-8), after_pulsing=0)

receiver_3 = Receiver(transmittance=1)

FiberChannel_3 = FiberChannel(loss_per_km=0.2, distance_km=0, detection_error=0.005)

f_3 = 1.05

key_rate_loss_bb84(min=0, max=275, values_number=300, source=source_3, detector=detector_3, channel=FiberChannel_3, receiver=receiver_3, correction_efficiency=f_3, title="Evolution of the key rate with the loss for AMHPS")


# Test 4:

source_4 = Entangled_PDC_Source(mean_photon_number=0.053, repetition_rate=0)

detector_4 = Threshold_detector(dark_count_rate=6.02, efficiency=14.5/100, time_window=10**(-6), after_pulsing=0)

receiver_4 = Receiver(transmittance=1)

FiberChannel_4 = FiberChannel(loss_per_km=0.21, distance_km=0, detection_error=0.015)

f_4 = 1.22

key_rate_loss_pulsed_bbm92(min=0, max=170, values_number=300, source=source_4, detector1=detector_4, detector2=detector_4, channel_1=FiberChannel_4, receiver1=receiver_4, receiver2=receiver_4, correction_efficiency=f_4, title="Evolution of the key rate with the loss for an entangled PDC source")

# Test 6
## Testing BBM92 continuous, Voigt profile

def g2_source_6(x):
    return voigt_profile(x, 123.2*10**(-12), 99.3*10**(-12))

source_6 = Continuous_Wave_Pumped_Source(brightness=1646*(10**5), g2_profile= g2_source_6, optical_losses = 4.5)

detector_6 = Threshold_detector(dark_count_rate=350, efficiency=0.76, time_window=310*10**(-12), after_pulsing=0)

channel_6 = FiberChannel(loss_per_km=0.1, distance_km=0, detection_error=0.005)

receiver_6 = Receiver(transmittance=1, x_basis_loss = 6, z_basis_loss = 3)

f_6 = 1.2

#graph_proba(0,3,source_6, "Spiral resonator source statistic")

#print(source_6.coincidence_window_efficiency(310*10**(-12)))

key_rate_loss_bbm92_continuous(min=0, max=275, values_number=300, source=source_6, detector1=detector_6, channel_1=channel_6, channel_2 = channel_6, receiver1=receiver_6, correction_efficiency=f_6, title="Key rate evolution with the loss in dB",coincidence_time =310*10**(-12))

# Test 7
## Testing BBM92 continuous, Gaussian profile


def g2_source_7(t):
    t_delta = 10**(-10)
    return (2.0 / t_delta) * np.sqrt(np.log(2.0) / np.pi)*np.exp(-4.0 * np.log(2.0) * (t ** 2) / (t_delta**2))

source_7 = Continuous_Wave_Pumped_Source(brightness=0.05*(10**9), g2_profile = g2_source_7)

detector_7 = Threshold_detector(dark_count_rate=250, efficiency=0.76, time_window=46*10**(-12), after_pulsing=0)

channel_7 = FiberChannel(loss_per_km=0.2, distance_km=0, detection_error=0.01)

receiver_7 = Receiver(transmittance=1)

f_7 = 1.2

key_rate_loss_bbm92_continuous(min=0, max=400, values_number=400, source=source_7, detector1=detector_7, channel_1=channel_7, receiver1=receiver_7, correction_efficiency=f_7, title="Key rate evolution with the loss in dB",coincidence_time =46*10**(-12))
