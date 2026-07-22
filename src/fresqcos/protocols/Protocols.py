from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt
import math

from fresqcos.sources.sources import Attenuated_Laser, Single_Photon_Source, Multiplexed_Heralded_Photon_Source, Symmetric_Multiplexed_Heralded_Photon_Source, Asymmetric_Multiplexed_Heralded_Photon_Source, Entangled_PDC_Source

from fresqcos.detectors.detectors import Threshold_detector

from fresqcos.protocols.protocols import BB84, BBM92

from fresqcos.receiver import Receiver

from fresqcos.channel import Channel

## Functions

def binary_shannon_entropy(x):
    if x<=0 or x>=1:
        return 0
    return -x*math.log2(x)-(1-x)*math.log2(1-x)

## Protocols

class Protocol(ABC):

    def __init__(self, source: Source, detector: Detector, channel: Channel, receiver: Receiver, correction_efficiency: float, distance_km: float):

        self.source = source
        self.detector = detector
        self.channel = channel
        self.receiver = receiver
        self.correction_efficiency = correction_efficiency
        self.distance_km = distance_km

class BB84(Protocol):

    def __init__(self, source: Source, detector: Detector, channel: Channel, receiver: Receiver, correction_efficiency: float, distance_km: float):

        self.source = source
        self.detector = detector
        self.channel = channel
        self.receiver = receiver
        self.correction_efficiency = correction_efficiency
        self.distance_km = distance_km

    def transmittance_i_photon_state(self, i):

        return 1-(1-self.detector.efficiency*self.receiver.transmittance*self.channel.transmittance(self.distance_km))**i

    def yield_i_photon_state(self, i):
        #probability for Bob to have a detection assuming that Alice sent an i-photon state
        return  self.detector.background_rate() + self.transmittance_i_photon_state(i)*(1+self.detector.after_pulsing)

    def gain_i_photon_state(self, i):
        #probability for Alice to send an i-photon state and for Bob to have a detection
        return self.yield_i_photon_state(i)*self.source.probability_sending_i_state(i)

    def overall_gain(self):
        gain = 0
        for i in range(0,50):
            gain += self.gain_i_photon_state(i)
        return gain


    def quantum_bit_error_rate(self,i):
        return (1/2 * self.detector.background_rate() + (self.channel.probability_hitting_wrong_detector()+1/2 *self.detector.after_pulsing ) * self.transmittance_i_photon_state(i))/self.yield_i_photon_state(i)

    def overall_quantum_bit_error_rate(self):
        qber = 0
        for i in range(0,50):
            qber += (self.quantum_bit_error_rate(i)*self.yield_i_photon_state(i)*self.source.probability_sending_i_state(i))
        qber = qber/self.overall_gain()
        return qber

    def key_rate_decoy_state_inf_key(self):
        return self.source.probability_sending_i_state(0)*self.detector.background_rate() + self.source.probability_sending_i_state(1)*self.yield_i_photon_state(1)*(1-binary_shannon_entropy(self.quantum_bit_error_rate(1)))-self.overall_gain()*self.correction_efficiency*binary_shannon_entropy(self.overall_quantum_bit_error_rate())

    def key_rate_no_decoy_state_inf_key(self):

        delta = (1-self.source.probability_sending_i_state(0)-self.source.probability_sending_i_state(1))/self.overall_gain()

        return self.overall_gain()*((1-delta)*(1-binary_shannon_entropy(self.overall_quantum_bit_error_rate()/(1-delta)))-self.correction_efficiency*binary_shannon_entropy(self.overall_quantum_bit_error_rate()))


class BBM92(Protocol):

    def __init__(self, source: Source, detector1: Detector, detector2: Detector, channel1: Channel, channel2: Channel, receiver1: Receiver, receiver2: Receiver, correction_efficiency: float, distance_km1: float, distance_km2: float):

        self.source = source
        self.detector1 = detector1
        self.detector2 = detector2
        self.channel1 = channel1
        self.channel2 = channel2
        self.receiver1 = receiver1
        self.receiver2 = receiver2
        self.correction_efficiency = correction_efficiency
        self.distance_km1 = distance_km1
        self.distance_km2 = distance_km2

    def transmittance_i_photon_state1(self, i):

        return 1-(1-self.detector1.efficiency*self.receiver1.transmittance*self.channel1.transmittance(self.distance_km1)*(1+self.detector1.after_pulsing))**i

    def transmittance_i_photon_state2(self, i):

        return 1-(1-self.detector2.efficiency*self.receiver2.transmittance*self.channel2.transmittance(self.distance_km2)*(1+self.detector2.after_pulsing))**i

    def yield_i_photon_state(self, i):

        return  (1-(1-self.detector1.background_rate())*(1-self.transmittance_i_photon_state1(i)))*(1-(1-self.detector2.background_rate())*(1-self.transmittance_i_photon_state2(i)))

    def gain_i_photon_state(self, i):

        return self.yield_i_photon_state(i)*self.source.probability_sending_i_state(i)

    def overall_gain(self):
        gain = 0
        for i in range(0,50):
            gain += self.gain_i_photon_state(i)
        return gain


    def entanglement_error(self,n,m):

        return 1/2-((1/2-((self.channel1.probability_hitting_wrong_detector()+self.channel2.probability_hitting_wrong_detector()+(self.detector1.after_pulsing+self.detector2.after_pulsing)/4)/(1+(self.detector1.after_pulsing+self.detector2.after_pulsing)/2)))/self.yield_i_photon_state(n))*(-self.transmittance_i_photon_state1(n-m)+self.transmittance_i_photon_state1(m))*(-self.transmittance_i_photon_state2(n-m)+self.transmittance_i_photon_state2(m))


    def quantum_bit_error_rate(self,i):
        qber = 0
        for n in range(0,i+1):
            qber = qber + self.entanglement_error(i,n)
        return qber/(1+i)


    def overall_quantum_bit_error_rate(self):
        qber = 0
        for i in range(0,50):
            qber += (self.quantum_bit_error_rate(i)*self.yield_i_photon_state(i)*self.source.probability_sending_i_state(i))
        qber = qber/self.overall_gain()
        return max(0,qber)

    def key_rate(self):

        return (self.overall_gain()/2)*(1-binary_shannon_entropy(self.overall_quantum_bit_error_rate())*(1+self.correction_efficiency))
