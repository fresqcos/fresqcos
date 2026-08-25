from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.stats import poisson
import scipy

class Source(ABC):
    pass
## Pulsed sources


class Attenuated_Laser(Source):

    def __init__(self,*, mean_photon_number: float, repetition_rate: float, optical_losses: Optional[float] = None):

        self.mean_photon_number = mean_photon_number
        self.repetition_rate = repetition_rate

        if optical_losses is None:
            self.optical_losses = 0
        else:
            self.optical_losses = optical_losses

    @property
    def repetition_rate(self) -> float:
        """ Return the pulse number per second.

        Must be non-negative
        """
        return self._repetition_rate

    @repetition_rate.setter
    def repetition_rate(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"repetition_rate must be non-negative, got {value}")

        self._repetition_rate = float(value)

    @property
    def mean_photon_number(self) -> float:
        """ Return the mean photon number per pulse.

        Must be non-negative
        """
        return self._mean_photon_number

    @mean_photon_number.setter
    def mean_photon_number(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"mean_photon_number must be non-negative, got {value}")

        self._mean_photon_number = float(value)


    def probability_sending_i_state(self,i) -> float:
        return poisson.pmf(i, self.mean_photon_number)

    def optical_efficiency(self):
        return 10**(-self.optical_losses/10)


class Multiplexed_Heralded_Photon_Source(Source):

    def __init__(self,*, mean_photon_number: float, repetition_rate: float, sources_num: int, optical_losses: Optional[float] = None):

        self.mean_photon_number = mean_photon_number
        self.sources_num = sources_num
        self.repetition_rate = repetition_rate

        if optical_losses is None:
            self.optical_losses = 0
        else:
            self.optical_losses = optical_losses

    @property
    def repetition_rate(self) -> float:
        """ Return the pulse number per second.

        Must be non-negative
        """
        return self._repetition_rate

    @repetition_rate.setter
    def repetition_rate(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"repetition_rate must be non-negative, got {value}")

        self._repetition_rate = float(value)

    @property
    def mean_photon_number(self) -> float:
        """ Return the mean photon number per pulse.

        Must be non-negative
        """
        return self._mean_photon_number

    @mean_photon_number.setter
    def mean_photon_number(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"mean_photon_number must be non-negative, got {value}")

        self._mean_photon_number = float(value)

    @property
    def sources_num(self) -> int:
        """ Return the HS units number.

        Must be non-negative
        """
        return self._sources_num

    @sources_num.setter
    def sources_num(self, value: int) -> None:
        if value <= 0 :
            raise ValueError(f"sources_num must be non-negative, got {value}")
        self._sources_num = int(value)

    def probability_sending_i_state(self,i):

        if i==0:
            return np.exp(-self.mean_photon_number*self.sources_num)

        else:
            return poisson.pmf(i, self.mean_photon_number)*(1-np.exp(-self.mean_photon_number*self.sources_num))/np.exp(-self.mean_photon_number)

    def optical_efficiency(self):
        return 10**(-self.optical_losses/10)


class Symmetric_Multiplexed_Heralded_Photon_Source(Source):

    def __init__(self,*, mean_photon_number: float,repetition_rate: float, sources_num: int, transmittance: float, efficiency: float, optical_losses: Optional[float] = None):

        self.mean_photon_number = mean_photon_number
        self.repetition_rate = repetition_rate
        self.sources_num = sources_num
        self.transmittance = transmittance
        self.efficiency = efficiency

        if optical_losses is None:
            self.optical_losses = 0
        else:
            self.optical_losses = optical_losses

    @property
    def repetition_rate(self) -> float:
        """ Return the pulse number per second.

        Must be non-negative
        """
        return self._repetition_rate

    @repetition_rate.setter
    def repetition_rate(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"repetition_rate must be non-negative, got {value}")

        self._repetition_rate = float(value)

    @property
    def mean_photon_number(self) -> float:
        """ Return the mean photon number per pulse.

        Must be non-negative
        """
        return self._mean_photon_number

    @mean_photon_number.setter
    def mean_photon_number(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"mean_photon_number must be non-negative, got {value}")

        self._mean_photon_number = float(value)

    @property
    def sources_num(self) -> int:
        """ Return the HS units number.

        Must be non-negative
        """
        return self._sources_num

    @sources_num.setter
    def sources_num(self, value: int) -> None:
        if value <= 0 or (value & (value - 1)) != 0:
            raise ValueError(f"sources_num must be non-negative and a power of 2  , got {value}")
        self._sources_num = int(value)

    def probability_sending_i_state(self,i):
        k = math.log2(self.sources_num)
        return (1-self.efficiency)*np.exp(-(1-self.efficiency)*self.mean_photon_number)*np.exp(-self.efficiency*self.mean_photon_number*(2/self.transmittance)**k)/math.factorial(i)+poisson.pmf(i, self.mean_photon_number)*(1-((1-self.efficiency)**i)*np.exp(-self.efficiency*self.mean_photon_number*(-1+1/self.transmittance**i)))*(1-np.exp(-self.efficiency*self.mean_photon_number*(2/self.transmittance)**k))/(1-np.exp(-self.efficiency*self.mean_photon_number/(self.transmittance**k)))

    def optical_efficiency(self):
        return 10**(-self.optical_losses/10)


class Asymmetric_Multiplexed_Heralded_Photon_Source(Source):

    def __init__(self,*, mean_photon_number: float, repetition_rate: float, sources_num: int, transmittance: float, efficiency: float, optical_losses: Optional[float] = None):

        self.mean_photon_number = mean_photon_number
        self.repetition_rate = repetition_rate
        self.sources_num = sources_num
        self.transmittance = transmittance
        self.efficiency = efficiency

        if optical_losses is None:
            self.optical_losses = 0
        else:
            self.optical_losses = optical_losses

    @property
    def repetition_rate(self) -> float:
        """ Return the pulse number per second.

        Must be non-negative
        """
        return self._repetition_rate

    @repetition_rate.setter
    def repetition_rate(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"repetition_rate must be non-negative, got {value}")

        self._repetition_rate = float(value)

    @property
    def mean_photon_number(self) -> float:
        """ Return the mean photon number per pulse.

        Must be non-negative
        """
        return self._mean_photon_number

    @mean_photon_number.setter
    def mean_photon_number(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"mean_photon_number must be non-negative, got {value}")

        self._mean_photon_number = float(value)

    @property
    def sources_num(self) -> int:
        """ Return the HS units number.

        Must be non-negative
        """
        return self._sources_num

    @sources_num.setter
    def sources_num(self, value: int) -> None:
        if value <= 0:
            raise ValueError(f"sources_num must be non-negative, got {value}")
        self._sources_num = int(value)

    def probability_sending_i_state(self,i):
        sum = 0
        for k in range(1, self.sources_num):
            if k == self.sources_num:
                k = self.sources_num -1
            sum += np.exp(-self.efficiency*self.mean_photon_number*((self.transmittance**(1-k)-1)/(1-self.transmittance)))*(1-((1-self.efficiency)**k))*np.exp(self.efficiency*self.mean_photon_number-self.efficiency*self.mean_photon_number/self.transmittance**k)

        return poisson.pmf(i, self.mean_photon_number)*sum + (1-self.efficiency)*np.exp(-(1-self.efficiency)*self.mean_photon_number)*np.exp(-self.efficiency*self.mean_photon_number*(((2-self.transmittance)*self.transmittance**(1-self.sources_num))-1)/(1-self.transmittance))/math.factorial(i)

    def optical_efficiency(self):
        return 10**(-self.optical_losses/10)


class Single_Photon_Source(Source):

    def __init__(self,*, repetition_rate: float, brightness: float, g2: float, optical_losses: Optional[float] = None):

        self.repetition_rate = repetition_rate
        self.brightness = brightness
        self.g2 = g2

        if optical_losses is None:
            self.optical_losses = 0
        else:
            self.optical_losses = optical_losses

    @property
    def repetition_rate(self) -> float:
        """ Return the pulse number per second.

        Must be non-negative
        """
        return self._repetition_rate

    @repetition_rate.setter
    def repetition_rate(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"repetition_rate must be non-negative, got {value}")

        self._repetition_rate = float(value)

    @property
    def brightness(self) -> float:
        """ Return the brightness, the probability of a detection.

        Must be non-negative and less than one.
        """
        return self._brightness

    @brightness.setter
    def brightness(self, value: float) -> None:
        if value <= 0 or value>1 :
            raise ValueError(f"brightness must be non-negative and less than 1, got {value}")
        self._brightness = float(value)

    @property
    def g2(self) -> float:
        """ Return the g2(0).

        Must be non-negative
        """
        return self._g2

    @g2.setter
    def g2(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"g2 must be non-negative, got {value}")
        self._g2 = float(value)


    def probability_sending_i_state(self,i):
        if i<0 or i>2:
            return 0

        elif i==0:
            return 1-self.brightness

        else:
            p2 = (1-self.g2*self.brightness-np.sqrt(1-2*self.g2*self.brightness))/self.g2

            if i==1:
                return self.brightness-p2

            else:
                return p2

    def optical_efficiency(self):
        return 10**(-self.optical_losses/10)


class Entangled_PDC_Source(Source):

    def __init__(self,*, mean_photon_number: float, repetition_rate: float, optical_losses: Optional[float] = None):

        self.mean_photon_number = mean_photon_number
        self.repetition_rate = repetition_rate

        if optical_losses is None:
            self.optical_losses = 0
        else:
            self.optical_losses = optical_losses

    @property
    def repetition_rate(self) -> float:
        """ Return the pulse number per second.

        Must be non-negative
        """
        return self._repetition_rate

    @repetition_rate.setter
    def repetition_rate(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"repetition_rate must be non-negative, got {value}")

        self._repetition_rate = float(value)

    @property
    def mean_photon_number(self) -> float:
        """ Return the mean photon number per pulse.

        Must be non-negative
        """
        return self._mean_photon_number

    @mean_photon_number.setter
    def mean_photon_number(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"mean_photon_number must be non-negative, got {value}")

        self._mean_photon_number = float(value)

    def brightness_parameter(self):
        return self.mean_photon_number/2


    def probability_sending_i_state(self,i) -> float:
        return ((i+1)*self.brightness_parameter()**i)/(self.brightness_parameter()+1)**(i+2)

    def optical_efficiency(self):
        return 10**(-self.optical_losses/10)

## Continuous wave pumped

class Continuous_Wave_Pumped_Source(Source):

    def __init__(self,*, brightness: float, g2_profile: Callable, optical_losses: Optional[float] = None):

        self.brightness = brightness
        self.g2_profile = g2_profile

        if optical_losses is None:
            self.optical_losses = 0
        else:
            self.optical_losses = optical_losses


    @property
    def brightness(self) -> float:
        """ Return the brightness, the probability of a detection.

        Must be non-negative and less than one.
        """
        return self._brightness

    @brightness.setter
    def brightness(self, value: float) -> None:
        if value <= 0 :
            raise ValueError(f"brightness must be non-negative, got {value}")
        self._brightness = float(value)

    def probability_sending_i_state(self,i, coincidence_time) -> float:
        return poisson.pmf(i, self.brightness*coincidence_time)

    def coincidence_window_efficiency(self, coincidence_time):

        return quad(self.g2_profile, -coincidence_time, coincidence_time)[0]

    def optical_efficiency(self):
        return 10**(-self.optical_losses/10)

## Entanglement swapping sources

class Sagnac_Sources(Source):

    def __init__(self,*, mean_photon_number: float, optical_losses: Optional[float] = None):

        self.mean_photon_number = mean_photon_number

        if optical_losses is None:
            self.optical_losses = 0
        else:
            self.optical_losses = optical_losses

    @property
    def mean_photon_number(self) -> float:
        """ Return the mean photon number per pulse.

        Must be non-negative
        """
        return self._mean_photon_number

    @mean_photon_number.setter
    def mean_photon_number(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"mean_photon_number must be non-negative, got {value}")

        self._mean_photon_number = float(value)

    def optical_efficiency(self):
        return 10**(-self.optical_losses/10)

    def two_modes_squeezed_vacuum_states(self, sources_number):
        c_1 = 2*self.mean_photon_number+1
        c_2 = 2*np.sqrt(self.mean_photon_number*(self.mean_photon_number+1))

        partial_x_quadrature_matrix = np.array([[c_1,c_2],
                                                [c_2,c_1]])

        partial_p_quadrature_matrix = np.array([[c_1,-c_2],
                                                [-c_2,c_1]])

        x_quadrature = np.kron(np.eye(2),partial_x_quadrature_matrix)

        p_quadrature = np.kron(np.eye(2),partial_p_quadrature_matrix)

        overall_x = np.kron(np.eye(sources_number), x_quadrature)

        overall_p = np.kron(np.eye(sources_number), p_quadrature)

        return scipy.linalg.block_diag(overall_x, overall_p)

    def emitted_pairs_covariance_matrix(self, sources_number):

        partial_swapping_matrix = np.array([[1,0,0,0],
                                            [0,0,0,1],
                                            [0,0,1,0],
                                            [0,1,0,0]])

        swapping_matrix = np.kron(np.eye(2*sources_number), partial_swapping_matrix)

        return np.dot(swapping_matrix.T, np.dot(self.two_modes_squeezed_vacuum_states(sources_number),swapping_matrix))

