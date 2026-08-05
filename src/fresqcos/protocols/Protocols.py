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

def no_x_event_i_time(x,i):

    return (1-x)**i

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

    def __init__(self,*, source: Source, detector: Detector, channel: FiberChannel, receiver: Receiver, correction_efficiency: float):

        self.source = source
        self.detector = detector
        self.channel = channel
        self.receiver = receiver
        self.correction_efficiency = correction_efficiency

## X basis

    def transmittance_i_photon_state_x(self, i):

        return 1-(1-self.detector.efficiency*self.receiver.x_basis_transmittance()*self.channel.transmittance()*self.source.optical_efficiency())**i

    def yield_i_photon_state_x(self, i):
        return  self.detector.background_rate() + self.transmittance_i_photon_state_x(i)*(1+self.detector.after_pulsing)

    def gain_i_photon_state_x(self, i):
        return self.yield_i_photon_state_x(i)*self.source.probability_sending_i_state(i)

    def overall_gain_x(self):
        gain = 0
        for i in range(0,50):
            gain += self.gain_i_photon_state_x(i)
        return gain

    def quantum_bit_error_rate_x(self,i):
        return (1/2 * self.detector.background_rate() + (self.channel.detection_error+1/2 *self.detector.after_pulsing ) * self.transmittance_i_photon_state_x(i))/self.yield_i_photon_state_x(i)

    def overall_quantum_bit_error_rate_x(self):
        qber = 0
        for i in range(0,50):
            qber += (self.quantum_bit_error_rate_x(i)*self.yield_i_photon_state_x(i)*self.source.probability_sending_i_state(i))
        qber = qber/self.overall_gain_x()
        return qber

## Z basis

    def transmittance_i_photon_state_z(self, i):

        return 1-(1-self.detector.efficiency*self.receiver.z_basis_transmittance()*self.channel.transmittance()*self.source.optical_efficiency())**i

    def yield_i_photon_state_z(self, i):
        return  self.detector.background_rate() + self.transmittance_i_photon_state_z(i)*(1+self.detector.after_pulsing)

    def gain_i_photon_state_z(self, i):
        return self.yield_i_photon_state_z(i)*self.source.probability_sending_i_state(i)

    def overall_gain_z(self):
        gain = 0
        for i in range(0,50):
            gain += self.gain_i_photon_state_z(i)
        return gain

    def quantum_bit_error_rate_z(self,i):
        return (1/2 * self.detector.background_rate() + (self.channel.detection_error+1/2 *self.detector.after_pulsing ) * self.transmittance_i_photon_state_z(i))/self.yield_i_photon_state_z(i)

    def overall_quantum_bit_error_rate_z(self):
        qber = 0
        for i in range(0,50):
            qber += (self.quantum_bit_error_rate_z(i)*self.yield_i_photon_state_z(i)*self.source.probability_sending_i_state(i))
        qber = qber/self.overall_gain_z()
        return qber

## Key rates
    def gain(self):
        return (self.overall_gain_x()+self.overall_gain_z())/2

    def key_rate_decoy_state_inf_key(self):
        return self.source.probability_sending_i_state(0)*self.detector.background_rate() + self.source.probability_sending_i_state(1)*(self.yield_i_photon_state_x(1)+self.yield_i_photon_state_z(1))/2*(1-binary_shannon_entropy(self.quantum_bit_error_rate_x(1)))-self.overall_gain_z()*self.correction_efficiency*binary_shannon_entropy(self.overall_quantum_bit_error_rate_z())

    def key_rate_no_decoy_state_inf_key(self):

        delta = (1-self.source.probability_sending_i_state(0)-self.source.probability_sending_i_state(1))/self.gain()

        return self.gain()*((1-delta)*(1-binary_shannon_entropy(self.overall_quantum_bit_error_rate_x()/(1-delta)))-self.correction_efficiency*binary_shannon_entropy(self.overall_quantum_bit_error_rate_z()))


class Pulsed_BBM92(Protocol):

    def __init__(self, *, source: Source, detector1: Detector, channel_1: FiberChannel, receiver1: Receiver, correction_efficiency: float, detector2: Optional[Detector] = None, channel_2: Optional[FiberChannel] = None, receiver2: Optional[Receiver] = None):

        self.source = source
        self.detector1 = detector1

        if detector2 is None:
                    self.detector2 = detector1
        else:
            self.detector2 = detector2

        self.channel_1 = channel_1

        if channel_2 is None:
            self.channel_2 = channel_1
        else:
            self.channel_2 = channel_2

        self.receiver1 = receiver1

        if receiver2 is None:
            self.receiver2 = receiver1
        else:
            self.receiver2 = receiver2

        self.correction_efficiency = correction_efficiency

## Basis Z
    def transmittance_i_photon_state1_Z(self, i):

        return 1-(1-self.detector1.efficiency*self.receiver1.z_basis_transmittance()*self.channel_1.transmittance()*self.source.optical_efficiency()*(1+self.detector1.after_pulsing))**i

    def transmittance_i_photon_state2_Z(self, i):

        return 1-(1-self.detector2.efficiency*self.receiver2.z_basis_transmittance()*self.channel_2.transmittance()*self.source.optical_efficiency()*(1+self.detector2.after_pulsing))**i

    def yield_i_photon_stateZ(self, i):

        return  (1-(1-self.detector1.background_rate())*(1-self.transmittance_i_photon_state1_Z(i)))*(1-(1-self.detector2.background_rate())*(1-self.transmittance_i_photon_state2_Z(i)))

    def gain_i_photon_stateZ(self, i):

        return self.yield_i_photon_stateZ(i)*self.source.probability_sending_i_state(i)

    def overall_gainZ(self):
        gain = 0
        for i in range(0,50):
            gain += self.gain_i_photon_stateZ(i)
        return gain


    def entanglement_errorZ(self,n,m):

        return 1/2-((1/2-((self.channel_1.detection_error+self.channel_2.detection_error+(self.detector1.after_pulsing+self.detector2.after_pulsing)/4)/(1+(self.detector1.after_pulsing+self.detector2.after_pulsing)/2)))/self.yield_i_photon_stateZ(n))*(-self.transmittance_i_photon_state1_Z(n-m)+self.transmittance_i_photon_state1_Z(m))*(-self.transmittance_i_photon_state2_Z(n-m)+self.transmittance_i_photon_state2_Z(m))


    def quantum_bit_error_rateZ(self,i):
        qber = 0
        for n in range(0,i+1):
            qber = qber + self.entanglement_errorZ(i,n)
        return qber/(1+i)


    def overall_quantum_bit_error_rateZ(self):
        qber = 0
        for i in range(0,50):
            qber += (self.quantum_bit_error_rateZ(i)*self.yield_i_photon_stateZ(i)*self.source.probability_sending_i_state(i))
        qber = qber/self.overall_gainZ()
        return max(0,qber)

## Basis X
    def transmittance_i_photon_state1_X(self, i):

        return 1-(1-self.detector1.efficiency*self.receiver1.x_basis_transmittance()*self.channel_1.transmittance()*self.source.optical_efficiency()*(1+self.detector1.after_pulsing))**i

    def transmittance_i_photon_state2_X(self, i):

        return 1-(1-self.detector2.efficiency*self.receiver2.x_basis_transmittance()*self.channel_2.transmittance()*self.source.optical_efficiency()*(1+self.detector2.after_pulsing))**i

    def yield_i_photon_stateX(self, i):

        return  (1-(1-self.detector1.background_rate())*(1-self.transmittance_i_photon_state1_X(i)))*(1-(1-self.detector2.background_rate())*(1-self.transmittance_i_photon_state2_X(i)))

    def gain_i_photon_stateX(self, i):

        return self.yield_i_photon_stateX(i)*self.source.probability_sending_i_state(i)

    def overall_gainX(self):
        gain = 0
        for i in range(0,50):
            gain += self.gain_i_photon_stateX(i)
        return gain


    def entanglement_errorX(self,n,m):

        return 1/2-((1/2-((self.channel_1.detection_error+self.channel_2.detection_error+(self.detector1.after_pulsing+self.detector2.after_pulsing)/4)/(1+(self.detector1.after_pulsing+self.detector2.after_pulsing)/2)))/self.yield_i_photon_stateX(n))*(-self.transmittance_i_photon_state1_X(n-m)+self.transmittance_i_photon_state1_X(m))*(-self.transmittance_i_photon_state2_X(n-m)+self.transmittance_i_photon_state2_X(m))


    def quantum_bit_error_rateX(self,i):
        qber = 0
        for n in range(0,i+1):
            qber = qber + self.entanglement_errorX(i,n)
        return qber/(1+i)


    def overall_quantum_bit_error_rateX(self):
        qber = 0
        for i in range(0,50):
            qber += (self.quantum_bit_error_rateX(i)*self.yield_i_photon_stateX(i)*self.source.probability_sending_i_state(i))
        qber = qber/self.overall_gainX()
        return max(0,qber)

## Final gain and key rate
    def final_gain(self):
        return (self.overall_gainX()+self.overall_gainZ())/2

    def key_rate(self):

        return (self.final_gain()/2)*(1-binary_shannon_entropy(self.overall_quantum_bit_error_rateX())-binary_shannon_entropy(self.overall_quantum_bit_error_rateZ())*self.correction_efficiency)


## BBM92 with continuous-wave pumped entangled photon sources

class BBM92_continuous_wave_pumped_source(Protocol):

    def __init__(self, *, source: Source, detector1: Detector, channel_1: FiberChannel, receiver1: Receiver, correction_efficiency: float, coincidence_time: float, detector2: Optional[Detector] = None, channel_2: Optional[FiberChannel] = None, receiver2: Optional[Receiver] = None):

        self.source = source
        self.detector1 = detector1

        if detector2 is None:
                    self.detector2 = detector1
        else:
            self.detector2 = detector2

        self.channel_1 = channel_1

        if channel_2 is None:
            self.channel_2 = channel_1
        else:
            self.channel_2 = channel_2

        self.receiver1 = receiver1

        if receiver2 is None:
            self.receiver2 = receiver1
        else:
            self.receiver2 = receiver2

        self.correction_efficiency = correction_efficiency

        self.coincidence_time = coincidence_time



## Overall detector error

    def overall_detector_error(self):
        return (self.channel_1.detection_error+self.channel_2.detection_error)/2

## Basis X

    def heralding_efficiency_1_x(self):
        return self.detector1.efficiency*self.receiver1.x_basis_transmittance()*self.channel_1.transmittance()*self.source.optical_efficiency()

    def heralding_efficiency_2_x(self):
        return self.detector2.efficiency*self.receiver2.x_basis_transmittance()*self.channel_2.transmittance()*self.source.optical_efficiency()

    def true_coincidence_rate_i_n_x(self,i,n):
        return self.heralding_efficiency_1_x()*self.heralding_efficiency_2_x()*no_x_event_i_time(self.heralding_efficiency_1_x(),i-1)*no_x_event_i_time(self.heralding_efficiency_2_x(),i-1)*no_x_event_i_time(self.heralding_efficiency_2_x()/2,n-i)*no_x_event_i_time(self.heralding_efficiency_1_x()/2,n-i)*no_x_event_i_time(self.detector1.background_rate()/2,1)*no_x_event_i_time(self.detector2.background_rate()/2,1)

    def true_coincidence_rate_x_i(self,i):
        sum = 0
        for n in range(1,i+1):
            sum = sum + self.true_coincidence_rate_i_n_x(n,i)
        return sum

    def true_coincidence_rate_x(self):
        sum = 0
        for i in range(1,5):
            sum = sum+self.true_coincidence_rate_x_i(i)*self.source.probability_sending_i_state(i, self.coincidence_time)
        return sum/self.coincidence_time

    def miss_coincidence_x_i(self,i):
        return no_x_event_i_time(self.heralding_efficiency_1_x(),i)+no_x_event_i_time(self.heralding_efficiency_2_x(),i)-no_x_event_i_time(self.heralding_efficiency_1_x(),i)*no_x_event_i_time(self.heralding_efficiency_2_x(),i)-no_x_event_i_time(self.heralding_efficiency_1_x(),i)*(1-no_x_event_i_time(self.heralding_efficiency_2_x(),i))*self.detector1.background_rate()-no_x_event_i_time(self.heralding_efficiency_2_x(),i)*(1-no_x_event_i_time(self.heralding_efficiency_1_x(),i))*self.detector2.background_rate()-no_x_event_i_time(self.heralding_efficiency_1_x(),i)*no_x_event_i_time(self.heralding_efficiency_2_x(),i)*self.detector1.background_rate()*self.detector2.background_rate()

    def miss_coincidence_x(self):
        sum = 0
        for i in range(1,5):
            sum = sum + self.miss_coincidence_x_i(i)*self.source.probability_sending_i_state(i, self.coincidence_time)
        return sum

    def accidental_coincidence_rate_x(self):

        return (1-self.true_coincidence_rate_x()*self.coincidence_time-self.miss_coincidence_x()-self.source.probability_sending_i_state(0, self.coincidence_time) *(1-self.detector1.background_rate()*self.detector2.background_rate()))/self.coincidence_time

    def measured_coincidence_rate_x(self):

        return (self.source.coincidence_window_efficiency(self.coincidence_time)*self.true_coincidence_rate_x()+self.accidental_coincidence_rate_x())

    def coincidence_error_rate_x(self):

        return self.source.coincidence_window_efficiency(self.coincidence_time)*self.true_coincidence_rate_x()*self.overall_detector_error()+self.accidental_coincidence_rate_x()/2

    def qber_x(self):

        return self.coincidence_error_rate_x()/self.measured_coincidence_rate_x()

## Basis Z

    def heralding_efficiency_1_z(self):
        return self.detector1.efficiency*self.receiver1.z_basis_transmittance()*self.channel_1.transmittance()*self.source.optical_efficiency()

    def heralding_efficiency_2_z(self):
        return self.detector2.efficiency*self.receiver2.z_basis_transmittance()*self.channel_2.transmittance()*self.source.optical_efficiency()

    def true_coincidence_rate_i_n_z(self,i,n):
        return self.heralding_efficiency_1_z()*self.heralding_efficiency_2_z()*no_x_event_i_time(self.heralding_efficiency_1_z(),i-1)*no_x_event_i_time(self.heralding_efficiency_2_z(),i-1)*no_x_event_i_time(self.heralding_efficiency_2_z()/2,n-i)*no_x_event_i_time(self.heralding_efficiency_1_z()/2,n-i)*no_x_event_i_time(self.detector1.background_rate()/2,1)*no_x_event_i_time(self.detector2.background_rate()/2,1)

    def true_coincidence_rate_z_i(self,i):
        sum = 0
        for n in range(1,i+1):
            sum = sum + self.true_coincidence_rate_i_n_z(n,i)
        return sum

    def true_coincidence_rate_z(self):
        sum = 0
        for i in range(1,5):
            sum = sum+self.true_coincidence_rate_z_i(i)*self.source.probability_sending_i_state(i, self.coincidence_time)
        return sum/self.coincidence_time

    def miss_coincidence_z_i(self,i):
        return no_x_event_i_time(self.heralding_efficiency_1_z(),i)+no_x_event_i_time(self.heralding_efficiency_2_z(),i)-no_x_event_i_time(self.heralding_efficiency_1_z(),i)*no_x_event_i_time(self.heralding_efficiency_2_z(),i)-no_x_event_i_time(self.heralding_efficiency_1_z(),i)*(1-no_x_event_i_time(self.heralding_efficiency_2_z(),i))*self.detector1.background_rate()-no_x_event_i_time(self.heralding_efficiency_2_z(),i)*(1-no_x_event_i_time(self.heralding_efficiency_1_z(),i))*self.detector2.background_rate()-no_x_event_i_time(self.heralding_efficiency_1_z(),i)*no_x_event_i_time(self.heralding_efficiency_2_z(),i)*self.detector1.background_rate()*self.detector2.background_rate()

    def miss_coincidence_z(self):
        sum = 0
        for i in range(1,5):
            sum = sum + self.miss_coincidence_z_i(i)*self.source.probability_sending_i_state(i, self.coincidence_time)
        return sum

    def accidental_coincidence_rate_z(self):

        return (1-self.true_coincidence_rate_z()*self.coincidence_time-self.miss_coincidence_z()-self.source.probability_sending_i_state(0, self.coincidence_time) *(1-self.detector1.background_rate()*self.detector2.background_rate()))/self.coincidence_time

    def measured_coincidence_rate_z(self):

        return self.source.coincidence_window_efficiency(self.coincidence_time)*self.true_coincidence_rate_z()+self.accidental_coincidence_rate_z()

    def coincidence_error_rate_z(self):

        return self.source.coincidence_window_efficiency(self.coincidence_time)*self.true_coincidence_rate_z()*self.overall_detector_error()+self.accidental_coincidence_rate_z()/2

    def qber_z(self):

        return self.coincidence_error_rate_z()/self.measured_coincidence_rate_z()

## Key rate

    def overall_measured_coincidence(self):

        return (self.measured_coincidence_rate_z()+self.measured_coincidence_rate_x())/2

    def key_rate(self):
        return (self.overall_measured_coincidence()/2)*(1-binary_shannon_entropy(self.qber_x())-self.correction_efficiency*binary_shannon_entropy(self.qber_z()))
