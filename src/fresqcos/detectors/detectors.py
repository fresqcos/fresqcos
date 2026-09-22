"Module for defining different single-photon detectors."

from abc import ABC, abstractmethod


class Detector(ABC):
    "Abstract base class for single-photon detectors."

    def __init__(
        self,
        dark_count_rate: float,
        efficiency: float,
        time_window: float,
        after_pulsing: float,
    ) -> None:
        """Initialize the detector with the given parameters.

        Parameters
        ----------
        dark_count_rate : float
            The dark count rate of the detector in Hz.
        efficiency : float
            The efficiency of the detector (between 0 and 1).
        time_window : float
            The time window of the detector in seconds.
        after_pulsing : float
            The after pulsing probability of the detector (between 0 and 1).
        """
        self.dark_count_rate = dark_count_rate
        self.efficiency = efficiency
        self.time_window = time_window
        self.after_pulsing = after_pulsing

    @property
    def dark_count_rate(self) -> float:
        """Return the dark count rate in Hz. Must be non-negative."""
        return self._dark_count_rate

    @dark_count_rate.setter
    def dark_count_rate(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"dark_count_rate must be non-negative, got {value}")
        self._dark_count_rate = float(value)

    @property
    def efficiency(self) -> float:
        """Return the efficiency of the detector. Must be between 0 and 1."""
        return self._efficiency

    @efficiency.setter
    def efficiency(self, value: float) -> None:
        if value < 0 or value > 1:
            raise ValueError(f"efficiency must be between 0 and 1, got {value}")
        self._efficiency = float(value)

    @property
    def time_window(self) -> float:
        """Return the time window of the detector in s. Must be non-negative."""
        return self._time_window

    @time_window.setter
    def time_window(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"time_window must be non-negative, got {value}")

        self._time_window = float(value)

    @property
    def after_pulsing(self) -> float:
        """Return the after pulsing probability of the detector. Must be between 0 and 1."""
        return self._after_pulsing

    @after_pulsing.setter
    def after_pulsing(self, value: float) -> None:
        if value < 0 or value > 1:
            raise ValueError(f"after_pulsing must be between 0 and 1, got {value}")
        self._after_pulsing = float(value)

    @abstractmethod
    def compute_dark_count_probability(self) -> float:
        """Compute the dark count probability."""
        pass

    @abstractmethod
    def compute_background_rate(self) -> float:
        """Compute the overall background rate."""
        pass


class ThresholdDetector(Detector):
    "Class for threshold single-photon detectors."

    def __init__(
        self,
        dark_count_rate: float,
        efficiency: float,
        time_window: float,
        after_pulsing: float,
    ) -> None:
        """Initialize the threshold detector with the given parameters.

        Parameters
        ----------
        dark_count_rate : float
            The dark count rate of the detector in Hz.
        efficiency : float
            The efficiency of the detector (between 0 and 1).
        time_window : float
            The time window of the detector in seconds.
        after_pulsing : float
            The after pulsing probability of the detector (between 0 and 1).
        """
        super().__init__(dark_count_rate, efficiency, time_window, after_pulsing)

    def compute_dark_count_probability(self) -> float:
        """Compute the dark count probability of the threshold detector.

        Returns
        -------
        dark_count_prob : float
            The dark count probability, which is the minimum of 1 and the product of
            dark count rate and time window.
        """
        probability = self.dark_count_rate * self.time_window
        dark_count_prob = min(1, probability)
        return dark_count_prob

    def compute_background_rate(self) -> float:
        """Compute the overall background rate of the threshold detector.

        Returns
        -------
        background_rate : float
            The overall background rate, which is the minimum of 1 and twice the product
            of dark count probability and (1 + after pulsing).
        """
        rate = 2 * self.compute_dark_count_probability() * (1 + self.after_pulsing)
        background_rate = min(1, rate)
        return background_rate
