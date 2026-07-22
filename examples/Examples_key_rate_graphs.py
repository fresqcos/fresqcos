""" I put where the values are coming from and the results in the section 'Example' of the latex doc"""

from matplotlib import pyplot as plt
import numpy as np

from fresqcos.sources.sources import Attenuated_Laser, Single_Photon_Source, Multiplexed_Heralded_Photon_Source, Symmetric_Multiplexed_Heralded_Photon_Source, Asymmetric_Multiplexed_Heralded_Photon_Source, Entangled_PDC_Source

from fresqcos.detectors.detectors import Threshold_detector

from fresqcos.protocols.protocols import BB84, BBM92

from fresqcos.receiver import Receiver

from fresqcos.channel import Channel

## Plot

def key_rate_distance_km_bb84(min, max, values_number, source: Source, detector: Detector, channel: Channel, receiver: Receiver, correction_efficiency: float, title: str):
    x_values = np.linspace(min, max, values_number)
    y1_values = []
    for x in x_values:
        protocol = BB84(source, detector, channel, receiver, correction_efficiency, x)
        y1 = protocol.key_rate_decoy_state_inf_key()
        y1_values.append(y1)

    plt.plot(x_values, y1_values, color = 'blue', label = "With active decoy state")
    plt.yscale('log')
    plt.xlabel("distance_km in km")
    plt.ylabel("Key rate in bpp")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def key_rate_distance_km_bbm92(min, max, values_number, source: Source, detector1: Detector,detector2: Detector, channel1: Channel,channel2: Channel, receiver1: Receiver,receiver2: Receiver, correction_efficiency: float, title: str):
    x_values = np.linspace(min, max, values_number)
    y1_values = []
    for x in x_values:
        protocol = BBM92(source, detector1,detector2, channel1,channel2, receiver1,receiver2, correction_efficiency, x,0)
        y1 = protocol.key_rate()
        y1_values.append(y1)

    plt.plot(x_values, y1_values, color = 'red')
    plt.yscale('log')
    plt.xlabel("distance_km in km")
    plt.ylabel("Key rate in bpp")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def key_rate_loss_bb84(min, max, values_number, source: Source, detector: Detector, channel: Channel, receiver: Receiver, correction_efficiency: float, title: str):
    x_values = np.linspace(min, max, values_number)
    horiz_axis = np.linspace(min*channel.loss_coef, max*channel.loss_coef, values_number)
    y1_values = []
    for x in x_values:
        protocol = BB84(source, detector, channel, receiver, correction_efficiency, x)
        y1 = protocol.key_rate_decoy_state_inf_key()
        y1_values.append(y1)

    plt.plot(horiz_axis, y1_values, color = 'blue', label = "With active decoy state")
    plt.yscale('log')
    plt.xlabel("Loss in dB")
    plt.ylabel("Key rate in bpp")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def key_rate_loss_bbm92(min, max, values_number, source: Source, detector1: Detector,detector2: Detector, channel1: Channel,channel2: Channel, receiver1: Receiver,receiver2: Receiver, correction_efficiency: float, title: str):
    x_values = np.linspace(min, max, values_number)
    horiz_axis = np.linspace(min*channel1.loss_coef, max*channel1.loss_coef, values_number)
    y1_values = []
    for x in x_values:
        protocol = BBM92(source, detector1,detector2, channel1,channel2, receiver1,receiver2, correction_efficiency, x,0)
        y1 = protocol.key_rate()
        y1_values.append(y1)

    plt.plot(horiz_axis, y1_values, color = 'red')
    plt.yscale('log')
    plt.xlabel("Loss in dB")
    plt.ylabel("Key rate in bpp")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


## Test 1:

source_1 = Attenuated_Laser(0.48, 0)

detector_1 = Threshold_detector(0.17,5/100,10**(-5),0) #Y_0 = 1.7*10**(-6)

receiver_1 = Receiver(0.9)

channel_1 = FiberChannel(0.21,0.934)

f_1 = 1.22

#key_rate_distance_km_bb84(0,160,300,source_1,detector_1,channel_1,receiver_1,f_1,"Evolution of the key rate with the distance_km for an attenuated laser")

#graph_proba(0,25,source_1, "Attenuated laser statistic")


## Test 2:

source_2 = Symmetric_Multiplexed_Heralded_Photon_Source(0.48,0,32, 0.5, 0.7)

detector_2 = Threshold_detector(20,0.25,10**(-8),0)

receiver_2 = Receiver(1)

channel_2 = FiberChannel(0.2,0.99)

f_2 = 1.05

#key_rate_loss_bb84(0,275,300,source_2,detector_2,channel_2,receiver_2,f_2,"Evolution of the key rate with the loss for SMHPS")

#graph_proba(0,25,source_2, "SMHPS statistic")


## Test 3:

source_3 = Asymmetric_Multiplexed_Heralded_Photon_Source(0.6,0,32, 0.5, 0.7)

detector_3 = Threshold_detector(20,0.25,10**(-8),0)

receiver_3 = Receiver(1)

channel_3 = FiberChannel(0.2,0.99)

f_3 = 1.05

#key_rate_loss_bb84(0,275,300,source_3,detector_3,channel_3,receiver_3,f_3,"Evolution of the key rate with the loss for AMHPS")

#graph_proba(0,25,source_3, "AMHPS statistic")

## Test 4:

source_4 = Entangled_PDC_Source(0.053, 0)

detector_4 = Threshold_detector(6.02,14.5/100,10**(-6),0)

receiver_4 = Receiver(1)

channel_4 = FiberChannel(0.21,0.97)

f_4 = 1.22

#key_rate_loss_bbm92(0,170,300, source_4, detector_4, detector_4, channel_4, channel_4, receiver_4, receiver_4, f_4, "Evolution of the key rate with the loss for an entangled PDC source")

#graph_proba(0,25,source_4, "Entangles PDC source statistic")