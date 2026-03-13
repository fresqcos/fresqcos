"""Module for free-space communication channels."""

import numpy as np
from scipy.special import i0, i1
from scipy.integrate import quad, quad_vec
from probability_distributions import lognegative_weibull_pdf, truncated_lognormal_pdf
import fiber_coupling as smf
import pandas as pd
import cn2
import matplotlib.pyplot as plt

# Zernike indices look-up table
RE = 6371  # Earth's radius [km]
MAX_N_WFS = 150  # Maximum radial index n returned by WFS
array_of_zernike_index = smf.get_zernikes_index_range(MAX_N_WFS)
lut_zernike_index_pd = pd.DataFrame(array_of_zernike_index[1:], columns=["n", "m", "j"])
lut_zernike_index_pd["j_Noll"] = smf.calculate_j_noll(
    lut_zernike_index_pd["n"], lut_zernike_index_pd["m"]
)


def compute_channel_length(ground_station_alt, aerial_platform_alt, zenith_angle):
    """Compute channel length that corresponds to a particular ground station altitude, aerial
    platform altitude and zenith angle.

    ## Parameters
        `ground_station_alt` : float
            Altitude of the ground station [km].
        `aerial_platform_alt` : float
            Altitude of the aerial platform [km].
        `zenith_angle` : float
            Zenith angle of aerial platform [degrees].
    ## Returns
    `length` : float
        Length of the channel [km].
    """
    zenith_angle = np.deg2rad(zenith_angle)
    RA = RE + aerial_platform_alt
    RG = RE + ground_station_alt
    length = np.sqrt(RA**2 + RG**2 * (np.cos(zenith_angle) ** 2 - 1)) - RG * np.cos(zenith_angle)
    return length


def compute_height_min_horiz(length, height):
    """Compute minimal height of a horizontal channel between two ballons at the same height.

    ## Parameters
        `length` : float
            length of the horizontal channel [km]
        `height̀` : float
            height of the balloons [km]
    ## Returns
    `hmin` : float
        Minimal height of the channel [km]
    """
    RS = RE + height
    theta = np.arcsin((length) / (2 * RS))
    hmin = np.cos(theta) * RS - RE
    return hmin


def sec(theta):
    """Compute secant of angle theta.

    ## Parameters
    `theta` : float
        Angle for which secant will be calculated [degrees].
    ## Returns
    `sec` : float
        Secant result.
    """
    theta = np.deg2rad(theta)
    sec = 1 / np.cos(theta)
    return sec


class SatToSatChannel:
    """Model for photon loss on a horizontal free-space channel.

    Uses Weibull probability density of atmospheric transmittance (PDT)
    from [Vasylyev et al., 2012] to sample the loss probability of the photon.

    ## Parameters
    ----------
    `w0` : float
        Waist radius of the beam at the transmitter [m].
    `rx_aperture` : float
        Diameter of the receiving telescope [m].
    `obs_ratio` : float
        Obscuration ratio of the receiving telescope.
    `cn2` : float
        Index of refraction structure constant [m**(-2/3)].
    `wavelength` : float
        Wavelength of the radiation [m].
    `pointing_error` : float
        Pointing error [rad].
    `tracking_efficiency` : float
        Efficiency of the coarse tracking mechanism.
    `t_atm` : float
        Atmospheric transmittance (square of the transmission coefficient).
    """

    def __init__(
        self,
        w0,
        rx_aperture,
        obs_ratio,
        cn2,
        wavelength,
        pointing_error=0,
        tracking_efficiency=0,
        t_atm=1,
    ):
        self.w0 = w0
        self.rx_aperture = rx_aperture
        self.obs_ratio = obs_ratio
        self.cn2 = cn2
        self.wavelength = wavelength
        self.pointing_error = pointing_error
        self.tracking_efficiency = tracking_efficiency
        self.t_atm = t_atm
        self.required_properties = ["length"]

    def _compute_rytov_variance(self, length):
        """Compute rytov variance for a horizontal channel [Andrews/Phillips, 2005].

        ## Parameters
        `length` : float
            Length of the channel [m].
        ## Returns
        `rytov_var` : float
            Rytov variance for given length.
        """
        k = 2 * np.pi / self.wavelength
        rytov_var = 1.23 * self.cn2 * k ** (7 / 6) * length ** (11 / 6)
        return rytov_var

    def _compute_wandering_variance(self, length):
        """Compute beam wandering variance for a horizontal channel [Andrews/Phillips, 2005].

        ## Parameters
        `length` : float
            Length of the channel [m].
        ## Returns
        `wandering_var` : float
            Beam wandering variance for given length [m^2].
        """
        k = 2 * np.pi / self.wavelength
        lambda_0 = 2 * length / (k * self.w0**2)
        theta_0 = 1
        rytov_var = self._compute_rytov_variance(length)

        def f(xi):
            return (theta_0 + (1 - theta_0) * xi) ** 2 + 1.63 * (rytov_var) ** (
                6 / 5
            ) * lambda_0 * (1 - xi) ** (16 / 5)

        def integrand(xi):
            return xi**2 / f(xi) ** (1 / 6)

        wandering_var = 7.25 * self.cn2 * self.w0 ** (-1 / 3) * length**3 * quad(integrand, 0, 1)[0]
        return wandering_var

    def _compute_scintillation_index_plane(self, rytov_var, length):
        """Compute aperture-averaged scintillation index of plane
        wave for a horizontal channel [Andrews/Phillips, 2005].

        ## Parameters
        `length` : float
            Length of the channel [m].
        `rytov_var` : float
            Rytov variance.
        ## Returns
        `scint_index` : float
            Scintillation index for requested input parameters.
        """
        k = 2 * np.pi / self.wavelength
        d = np.sqrt(k * self.rx_aperture**2 / (4 * length))
        first_term = 0.49 * rytov_var / (1 + 0.65 * d**2 + 1.11 * rytov_var ** (6 / 5)) ** (7 / 6)
        second_term = (
            0.51
            * rytov_var
            * (1 + 0.69 * rytov_var ** (6 / 5)) ** (-5 / 6)
            / (1 + 0.9 * d**2 + 0.62 * d**2 * rytov_var ** (6 / 5))
        )
        scint_index = np.exp(first_term + second_term) - 1
        return scint_index

    def _compute_scintillation_index_spherical(self, rytov_var, length):
        """Compute aperture-averaged scintillation index of spherical
        wave for a horizontal channel [Andrews/Phillips, 2005].

        ## Parameters
        `length` : float
            Length of the channel [m].
        `rytov_var` : float
            Rytov variance.
        ## Returns
        `scint_index` : float
            Scintillation index for requested input parameters.
        """
        k = 2 * np.pi / self.wavelength
        d = np.sqrt(k * self.rx_aperture**2 / (4 * length))
        beta_0_sq = 0.4065 * rytov_var
        first_term = 0.49 * beta_0_sq / (1 + 0.18 * d**2 + 0.56 * beta_0_sq ** (6 / 5)) ** (7 / 6)
        second_term = (
            0.51
            * beta_0_sq
            * (1 + 0.69 * beta_0_sq ** (6 / 5)) ** (-5 / 6)
            / (1 + 0.9 * d**2 + 0.62 * d**2 * beta_0_sq ** (6 / 5))
        )
        return np.exp(first_term + second_term) - 1

    def _compute_coherence_width_plane(self, length):
        """Compute coherence width of plane wave for a horizontal channel [Andrews/Phillips, 2005].

        ## Parameters
        `length` : float
            Length of the channel [m].
        ## Returns
        `coherence_width` : float
            Coherence width for requested input parameters.
        """
        k = 2 * np.pi / self.wavelength
        coherence_width = (0.42 * length * self.cn2 * k**2) ** (-3 / 5)
        return coherence_width

    def _compute_coherence_width_spherical(self, length):
        """Compute coherence width of spherical wave for a horizontal
        channel [Andrews/Phillips, 2005].

        ## Parameters
        `length` : float
            Length of the channel [m].
        ## Returns
        `coherence_width` : float
            Coherence width for requested input parameters.
        """
        k = 2 * np.pi / self.wavelength
        coherence_width = (0.16 * length * self.cn2 * k**2) ** (-3 / 5)
        return coherence_width

    def _compute_coherence_width_gaussian(self, length):
        """Compute coherence width of gaussian wave for a horizontal channel (valid also for the
        strong tubulence regime) [Andrews/Phillips, 2005].

        ## Parameters
        `length` : float
            Length of the channel [m].
        ## Returns
        `coherence_width` : float
            Coherence width for requested input parameters.
        """
        k = 2 * np.pi / self.wavelength
        lambda_0 = 2 * length / (k * self.w0**2)
        lambda_curv = lambda_0 / (1 + lambda_0**2)
        theta = 1 / (1 + lambda_0**2)
        rho_plane = (1.46 * self.cn2 * length * k**2) ** (-3 / 5)
        q = length / (k * rho_plane**2)
        theta_e = (theta - 2 * q * lambda_curv / 3) / (1 + 4 * q * lambda_curv / 3)
        lambda_e = lambda_curv / (1 + 4 * q * lambda_curv / 3)
        if theta_e >= 0:
            a_e = (1 - theta_e ** (8 / 3)) / (1 - theta_e)
        else:
            a_e = (1 + np.abs(theta_e) ** (8 / 3)) / (1 - theta_e)
        coherence_width = (
            2.1 * rho_plane * (8 / (3 * (a_e + 0.618 * lambda_e ** (11 / 6)))) ** (3 / 5)
        )
        return coherence_width

    def _compute_long_term_beam_size_at_receiver(self, rytov_var, length):
        """Compute long-term beamsize at receiver for a horizontal channel [Andrews/Phillips, 2005].

        ## Parameters
        `length` : float
            Length of the channel [m].
        `rytov_var` : float
            Rytov variance.
        ## Returns
        `w_lt` : float
            Long-term beamsize at receiver for requested input parameters [m].
        """
        k = 2 * np.pi / self.wavelength
        w_lt = self.w0 * np.sqrt(
            1
            + (self.wavelength * length / (np.pi * self.w0**2)) ** 2
            + 1.63 * rytov_var ** (6 / 5) * 2 * length / (k * self.w0**2)
        )
        return w_lt

    def _compute_short_term_beam_size_at_receiver(self, long_term_beamsize, wandering_var):
        """Compute short-term beamsize at receiver for a
        horizontal channel [Andrews/Phillips, 2005].

        ## Parameters
        `long_term_beamsize` : float
            Long-term beamsize at the receiver [m].
        `wandering_var` : float
            Beam wandering variance at receiver [m^2].
        ## Returns
        `w_st` : float
            Short-term beamsize at receiver for requested input parameters [m].
        """
        w_st = np.sqrt(long_term_beamsize**2 - wandering_var)
        return w_st

    def _compute_pdt(self, eta, length):
        """Compute probability distribution of atmospheric
        transmittance (PDT) [Vasylyev et al., 2018].

        ## Parameters
        `eta` : np.ndarray
            Input random variable values to calculate PDT for.
        `length` : float
            Length of the channel [km].
        ## Returns
        `integral` : np.ndarray
            PDT function for input eta.
        """
        z = length * 1e3
        rx_radius = self.rx_aperture / 2
        rytov_var = self._compute_rytov_variance(z)
        pointing_var = (self.pointing_error * z) ** 2
        wandering_var = (self._compute_wandering_variance(z) + pointing_var) * (
            1 - self.tracking_efficiency
        )
        wandering_percent = 100 * np.sqrt(wandering_var) / rx_radius
        print(f"Wandering percent [%]: {wandering_percent}, Lenth [km]: {length}")
        if wandering_percent > 100:
            print(
                "Warning ! The total wandering is larger than the aperture of "
                "the receiver. Use smaller values of pointing error."
            )

        w_lt = self._compute_long_term_beam_size_at_receiver(rytov_var, z)
        w_st = self._compute_short_term_beam_size_at_receiver(w_lt, wandering_var)

        xi = (rx_radius / w_st) ** 2
        t_0 = np.sqrt(1 - np.exp(-2 * xi))
        l = (
            8
            * xi
            * np.exp(-4 * xi)
            * i1(4 * xi)
            / (1 - np.exp(-4 * xi) * i0(4 * xi))
            / np.log(2 * t_0**2 / (1 - np.exp(-4 * xi) * i0(4 * xi)))
        )
        r = rx_radius * np.log(2 * t_0**2 / (1 - np.exp(-4 * xi) * i0(4 * xi))) ** (-1.0 / l)

        if wandering_var >= 1e-7:
            pdt = lognegative_weibull_pdf(eta, t_0, wandering_var, r, l)
        else:
            pdt = np.zeros(np.size(eta))
            delta_eta = np.abs(eta[1] - eta[0])
            pdt[np.abs(eta - t_0) < delta_eta] = 1
        return pdt

    def _compute_channel_pdf(self, eta_ch, length):
        """Compute probability density function (PDF) of free-space channel efficiency.

        ## Parameters
        `eta_ch` : np.ndarray
            Input random variable values to calculate pdf for.
        `length` : float
            Length of the channel [km].
        ## Returns
        `ch_pdf` : np.ndarray
            Channel PDF for input eta.
        """
        pdt = self._compute_pdt(eta_ch, length)
        pdt = pdt / np.sum(pdt)

        z = length * 1e3
        n = lut_zernike_index_pd["n"]
        n = np.array(lut_zernike_index_pd["n"].values)
        rytov_var = self._compute_rytov_variance(z)
        r0 = self._compute_coherence_width_gaussian(z)
        eta_s = (1 + rytov_var) ** (-1 / 4)
        bj2 = smf.bn2(self.rx_aperture, r0, n, self.obs_ratio)
        beta_opt = smf.beta_opt(self.obs_ratio)
        eta_smf_max = smf.eta_smf_max(self.obs_ratio, beta_opt)
        # ch_pdf = pdt*self.Tatm*smf.eta_ao(bj2)*eta_s*eta_smf_max
        ch_pdf = self._compute_pdt(
            eta_ch / (self.t_atm * smf.eta_ao(bj2) * eta_s * eta_smf_max), length
        ) / (self.t_atm * smf.eta_ao(bj2) * eta_s * eta_smf_max)
        return ch_pdf

    def _compute_mean_channel_efficiency(self, eta_ch, length, detector_efficiency=1):
        """Compute mean channel efficiency, including losses at the detector.

        ## Parameters
        `eta_ch` : np.ndarray
            Input random variable values to calculate pdf for.
        `length` : float
            Length of the channel [km].
        `detector_efficiency` : float
            Efficiency of detector at receiver (default 1).
        ## Returns
        `ch_pdf` : np.ndarray
            Channel PDF for input eta.
        """
        pdt = self._compute_pdt(eta_ch, length)
        pdt = pdt / np.sum(pdt)
        z = length * 1e3
        n = lut_zernike_index_pd["n"]
        n = np.array(lut_zernike_index_pd["n"].values)
        rytov_var = self._compute_rytov_variance(z)
        r0 = self._compute_coherence_width_gaussian(z)
        eta_s = (1 + rytov_var) ** (-1 / 4)
        bj2 = smf.bn2(self.rx_aperture, r0, n, self.obs_ratio)
        beta_opt = smf.beta_opt(self.obs_ratio)
        eta_smf_max = smf.eta_smf_max(self.obs_ratio, beta_opt)
        mean_transmittance = (
            np.sum(eta_ch * pdt)
            * self.t_atm
            * smf.eta_ao(bj2)
            * eta_s
            * eta_smf_max
            * detector_efficiency
        )
        return mean_transmittance

    def _draw_pdt_sample(self, length, n_samples):
        """Draw random sample from probability distribution of atmospheric transmittance (PDT).

        ## Parameters
        `length` : float
            Length of the channel [km].
        `n_samples` : int
            Number of samples to return.
        ## Returns
        `samples` : float
            Random samples of PDT.
        """
        eta = np.linspace(1e-7, 1, 1000)
        pdt = self._compute_pdt(eta, length)
        pdt = np.abs(pdt / np.sum(pdt))
        samples = np.random.choice(eta, n_samples, p=pdt)
        return samples

    def _draw_channel_pdf_sample(self, length, n_samples):
        """Draw random sample from free-space channel probability distribution.

        ## Parameters
        `length` : float
            Length of the channel [km].
        `n_samples` : int
            Number of samples to return.
        ## Returns
        `samples` : float
            Random samples of channel PDF.
        """
        z = length * 1e3
        eta = np.linspace(1e-7, 1, 1000)
        ch_pdf = self._compute_channel_pdf(eta, length)
        ch_pdf = np.abs(ch_pdf / np.sum(ch_pdf))
        ch_pdf_samples = np.random.choice(eta, n_samples, p=ch_pdf)
        rytov_var = self._compute_rytov_variance(z)
        scint_index = self._compute_scintillation_index_spherical(rytov_var, z)
        eta_s = (1 + scint_index) ** (-1 / 4)
        beta_opt = smf.beta_opt(self.obs_ratio)
        eta_smf_max = smf.eta_smf_max(self.obs_ratio, beta_opt)

        return self.t_atm * ch_pdf_samples * eta_smf_max * eta_s


class SatToGroundChannel:
    """Model for photon loss on a downlink free-space channel.

    Uses probability density of atmospheric transmittance (PDT) from [Vasylyev et al., 2018] to
    sample the loss probability of the photon.

    ## Parameters
    ----------
    `W0` : float
        Waist radius of the beam at the transmitter [m].
    `rx_aperture` : float
        Diameter of the receiving telescope [m].
    `obs_ratio` : float
        Obscuration ratio of the receiving telescope.
    `n_max` : int
        Maximum radial index of correction of AO system.
    `Cn0` : float
        Reference index of refraction structure constant at ground level [m**(-2/3)].
    `wind_speed` : float
        Rms speed of the wind [m/s].
    `wavelength` : float
        Wavelength of the radiation [m].
    `ground_station_alt` : float
        Altitude of the ground station [km].
    `aerial_platform_alt` : float
        Altitude of the aerial platform [km].
    `zenith_angle` : float
        Zenith angle of aerial platform [degrees].
    `pointing_error` : float
        Pointing error [rad].
    `tracking_efficiency` : float
        Efficiency of the coarse tracking mechanism.
    `Tatm` : float
        Atmospheric transmittance (square of the transmission coefficient).
    `integral_gain: float`
        Integral gain of the AO system integral controller.
    `control_delay: float`
        Delay of the AO system loop [s].
    `integration_time: float`
        Integration time of the AO system integral controller [s].
    """

    def __init__(
        self,
        W0,
        rx_aperture,
        obs_ratio,
        n_max,
        Cn0,
        wind_speed,
        wavelength,
        ground_station_alt,
        aerial_platform_alt,
        zenith_angle,
        pointing_error=0,
        tracking_efficiency=0,
        Tatm=1,
        integral_gain=1,
        control_delay=13.32e-4,
        integration_time=6.66e-4,
    ):
        super().__init__()
        self.W0 = W0
        self.rx_aperture = rx_aperture
        self.obs_ratio = obs_ratio
        self.n_max = n_max
        self.Cn2 = Cn0
        self.wind_speed = wind_speed
        self.wavelength = wavelength
        self.ground_station_alt = ground_station_alt
        self.aerial_platform_alt = aerial_platform_alt
        self.zenith_angle = zenith_angle
        self.pointing_error = pointing_error
        self.integral_gain = integral_gain
        self.control_delay = control_delay
        self.integration_time = integration_time
        self.tracking_efficiency = tracking_efficiency
        self.Tatm = Tatm
        self.required_properties = ["length"]

    def _compute_Cn2(self, h):
        """Compute index of refraction structure constant [Andrews/Phillips, 2005].
        Uses the Hufnagel-Valley (HV) model.

        ## Parameters
        `h` : np.ndarray
            Values of h corresponding to slant path of the channel to integrate over [m].
        ## Returns
        `Cn2` : float
            Index of refraction structure constant [m^(-2/3)].
        """
        Cn2 = cn2.hufnagel_valley(h, self.wind_speed, self.Cn0)
        return Cn2

    def _compute_rytov_variance_plane(self):
        """Compute rytov variance of a plane wave for a downlink channel [Andrews/Phillips, 2005].

        ## Returns
        `rytov_var` : float
            Rytov variance for given length.
        """
        ground_station_alt = self.ground_station_alt * 1e3
        aerial_platform_alt = self.aerial_platform_alt * 1e3
        k = 2 * np.pi / self.wavelength
        integrand = lambda h: self._compute_Cn2(h) * (h - ground_station_alt) ** (5 / 6)
        rytov_var = (
            2.25
            * k ** (7 / 6)
            * sec(self.zenith_angle) ** (11 / 6)
            * quad(integrand, ground_station_alt, aerial_platform_alt)[0]
        )
        return rytov_var

    def _compute_rytov_variance_spherical(self):
        """Compute rytov variance of a spherical wave for a downlink channel [Andrews/Phillips, 2005].

        ## Returns
        `rytov_var` :float
            Rytov variance for given length.
        """
        ground_station_alt = self.ground_station_alt * 1e3
        aerial_platform_alt = self.aerial_platform_alt * 1e3
        k = 2 * np.pi / self.wavelength
        integrand = (
            lambda h: self._compute_Cn2(h)
            * (h - ground_station_alt) ** (5 / 6)
            * ((aerial_platform_alt - h) / (aerial_platform_alt - ground_station_alt)) ** (5 / 6)
        )
        rytov_var = (
            2.25
            * k ** (7 / 6)
            * sec(self.zenith_angle) ** (11 / 6)
            * quad(integrand, ground_station_alt, aerial_platform_alt)[0]
        )
        return rytov_var

    def _compute_wandering_variance(self):
        """Compute beam wandering variance for a downlink channel [Andrews/Phillips, 2005].

        ## Returns
        `wandering_var` : float
            Beam wandering variance for given length [m^2].
        """
        ground_station_alt = self.ground_station_alt * 1e3
        aerial_platform_alt = self.aerial_platform_alt * 1e3
        k = 2 * np.pi / self.wavelength
        length = 1e3 * compute_channel_length(
            self.ground_station_alt, self.aerial_platform_alt, self.zenith_angle
        )
        Lambda_0 = 2 * length / (k * self.W0**2)
        Theta_0 = 1
        rytov_var = self._compute_rytov_variance_spherical()
        f = lambda h: (
            Theta_0
            + (1 - Theta_0) * (h - ground_station_alt) / (aerial_platform_alt - ground_station_alt)
        ) ** 2 + 1.63 * (rytov_var) ** (6 / 5) * Lambda_0 * (
            (aerial_platform_alt - h) / (aerial_platform_alt - ground_station_alt)
        ) ** (
            16 / 5
        )
        integrand = lambda h: self._compute_Cn2(h) * (h - ground_station_alt) ** 2 / f(h) ** (1 / 6)
        wandering_var = (
            7.25
            * sec(self.zenith_angle) ** 3
            * self.W0 ** (-1 / 3)
            * quad(integrand, ground_station_alt, aerial_platform_alt)[0]
        )
        return wandering_var

    def _compute_scintillation_index_plane(self, rytov_var, length):
        """Compute aperture-averaged scintillation index of plane wave for a downlink channel [Andrews/Phillips, 2005].

        ## Parameters
        `length` : float
            Length of the channel [m].
        `rytov_var` : float
            Rytov variance.
        ## Returns
        `scint_index` : float
            Scintillation index for requested input parameters.
        """
        k = 2 * np.pi / self.wavelength
        d = np.sqrt(k * self.rx_aperture**2 / (4 * length))
        first_term = 0.49 * rytov_var / (1 + 0.65 * d**2 + 1.11 * rytov_var ** (6 / 5)) ** (7 / 6)
        second_term = (
            0.51
            * rytov_var
            * (1 + 0.69 * rytov_var ** (6 / 5)) ** (-5 / 6)
            / (1 + 0.9 * d**2 + 0.62 * d**2 * rytov_var ** (6 / 5))
        )
        return np.exp(first_term + second_term) - 1

    def _compute_scintillation_index_spherical(self, rytov_var, length):
        """Compute aperture-averaged scintillation index of spherical wave for a downlink channel [Andrews/Phillips, 2005].

        ## Parameters
        `length` : float
            Length of the channel [m].
        `rytov_var` : float
            Rytov variance.
        ## Returns
        `scint_index` : float
            Scintillation index for requested input parameters.
        """
        k = 2 * np.pi / self.wavelength
        d = np.sqrt(k * self.rx_aperture**2 / (4 * length))
        beta_0_sq = 0.4065 * rytov_var
        first_term = 0.49 * beta_0_sq / (1 + 0.18 * d**2 + 0.56 * beta_0_sq ** (6 / 5)) ** (7 / 6)
        second_term = (
            0.51
            * beta_0_sq
            * (1 + 0.69 * beta_0_sq ** (6 / 5)) ** (-5 / 6)
            / (1 + 0.9 * d**2 + 0.62 * d**2 * beta_0_sq ** (6 / 5))
        )
        return np.exp(first_term + second_term) - 1

    def _compute_coherence_width_plane(self):
        """Compute coherence width of plane wave for a downlink channel [Andrews/Phillips, 2005].

        ## Returns
        `coherence_width` : float
            Coherence width for requested input parameters.
        """
        ground_station_alt = self.ground_station_alt * 1e3
        aerial_platform_alt = self.aerial_platform_alt * 1e3
        k = 2 * np.pi / self.wavelength
        integrand = lambda h: self._compute_Cn2(h)
        coherence_width = (
            0.42
            * k**2
            * sec(self.zenith_angle)
            * quad(integrand, ground_station_alt, aerial_platform_alt)[0]
        ) ** (-3 / 5)
        return coherence_width

    def _compute_coherence_width_spherical(self):
        """Compute coherence width of spherical wave for a downlink channel [Andrews/Phillips, 2005].

        ## Returns
        `coherence_width` : float
            Coherence width for requested input parameters.
        """
        ground_station_alt = self.ground_station_alt * 1e3
        aerial_platform_alt = self.aerial_platform_alt * 1e3
        k = 2 * np.pi / self.wavelength
        integrand = lambda h: self._compute_Cn2(h) * (
            (aerial_platform_alt - h) / (aerial_platform_alt - ground_station_alt)
        ) ** (5 / 3)
        coherence_width = (
            0.42
            * k**2
            * sec(self.zenith_angle)
            * quad(integrand, ground_station_alt, aerial_platform_alt)[0]
        ) ** (-3 / 5)
        return coherence_width

    def _compute_coherence_width_gaussian(self, length):
        """Compute coherence width of gaussian wave for an downlink channel [Andrews/Phillips, 2005].

        ## Parameters
        `length` : float
            Length of the channel [km].
        ## Returns
        `coherence_width` : float
            Coherence width for requested input parameters.
        """
        ground_station_alt = self.ground_station_alt * 1e3
        aerial_platform_alt = self.aerial_platform_alt * 1e3
        k = 2 * np.pi / self.wavelength
        z = length * 1e3
        Lambda_0 = 2 * z / (k * self.W0**2)
        Lambda = Lambda_0 / (1 + Lambda_0**2)
        Theta = 1 / (1 + Lambda_0**2)
        Theta_bar = 1 - Theta
        integrand_1 = lambda h: self._compute_Cn2(h) * (
            Theta
            + Theta_bar * (aerial_platform_alt - h) / (aerial_platform_alt - ground_station_alt)
        ) ** (5 / 3)
        mu_1d = quad(integrand_1, ground_station_alt, aerial_platform_alt)[0]
        integrand_2 = lambda h: self._compute_Cn2(h) * (
            (h - ground_station_alt) / (aerial_platform_alt - ground_station_alt)
        ) ** (5 / 3)
        mu_2d = quad(integrand_2, ground_station_alt, aerial_platform_alt)[0]
        coherence_width = (
            np.cos(np.deg2rad(self.zenith_angle))
            / (0.423 * k**2 * (mu_1d + 0.622 * mu_2d * Lambda ** (11 / 6)))
        ) ** (3 / 5)
        return coherence_width

    def _compute_long_term_beam_size_at_receiver(self, rytov_var, length):
        """Compute long-term beamsize at receiver for a downlink channel [Andrews/Phillips, 2005].

        ## Parameters
        `length` : float
            Length of the channel [m].
        `rytov_var` : float
            Rytov variance.
        ## Returns
        `W_LT` : float
            Long-term beamsize at receiver for requested input parameters [m].
        """
        k = 2 * np.pi / self.wavelength
        return self.W0 * np.sqrt(
            1
            + (self.wavelength * length / (np.pi * self.W0**2)) ** 2
            + 1.63 * rytov_var ** (6 / 5) * 2 * length / (k * self.W0**2)
        )

    def _compute_short_term_beam_size_at_receiver(self, long_term_beamsize, wandering_var):
        """Compute short-term beamsize at receiver for a downlink channel [Andrews/Phillips, 2005].

        ## Parameters
        `long_term_beamsize` : float
            Long-term beamsize at the receiver [m].
        `wandering_var` : float
            Beam wandering variance at receiver [m^2].
        ## Returns
        `W_ST` : float
            Short-term beamsize at receiver for requested input parameters [m].
        """
        return np.sqrt(long_term_beamsize**2 - wandering_var)

    def _compute_lognormal_parameters(self, r, R, l, short_term_beamsize, scint_index):
        """Compute mean and standard deviation of lognormal distribution [Vasylyev et al., 2018].

        ## Parameters
        `r` : float
            Deflection radius from center of receiver aperture [m].
        `R` : float
            Weibull distribution R parameter.
        `l` : float
            Weibull distribution l parameter.
        `short_term_beamsize` : float
            Short-term beamsize at receiver [m].
        `scint_index` : float
            Scintillation index of horizontal channel.
        ## Returns
        `mu, sigma` : tuple (float, float)
            Mean value (mu) and standard deviation (sigma) of lognormal distribution.
        """
        rx_radius = self.rx_aperture / 2
        eta_0 = 1 - np.exp(-2 * rx_radius**2 / short_term_beamsize**2)
        eta_mean = eta_0 * np.exp(-((r / R) ** l))
        eta_var = (1 + scint_index) * eta_mean**2
        mu = -np.log(eta_mean**2 / np.sqrt(eta_var))
        sigma = np.sqrt(np.log(eta_var / eta_mean**2))
        return mu, sigma

    def _compute_pdt_parameters(self, length):
        """Compute parameters useful for the calculation of the probability distribution
        of atmospheric transmittance [Vasylyev et al., 2018].

        ## Parameters
        `length` : float
            Length of the channel [km].
        ## Returns
        `lognormal_params` : function
            Output of _compute_lognormal_parameters. When evaluated at specific deflection radius r, returns
            mean value (mu) and standard deviation (sigma) of lognormal distribution at r.
        `wandering_var` : float
            Beam wandering variance at receiver [m^2].
        `W_LT` : float
            Long-term beamsize at receiver for requested input parameters [m].
        """
        z = length * 1e3
        rx_radius = self.rx_aperture / 2
        pointing_var = (self.pointing_error * z) ** 2
        rytov_var = self._compute_rytov_variance_spherical()
        scint_index = self._compute_scintillation_index_spherical(rytov_var, z)
        W_LT = self._compute_long_term_beam_size_at_receiver(rytov_var, z)
        wandering_var = (self._compute_wandering_variance() + pointing_var) * (
            1 - self.tracking_efficiency
        )
        print("Wandering variance [m^2]: ", wandering_var)
        wandering_percent = 100 * np.sqrt(wandering_var) / rx_radius
        print(f"Wandering percent [%]: {wandering_percent}, Lenth [km]: {length}")

        if wandering_percent > 100:
            print(
                "Warning ! The total wandering is larger than the aperture of the receiver. Use smaller values of pointing error."
            )

        W_ST = self._compute_short_term_beam_size_at_receiver(W_LT, wandering_var)
        print("Short-term beam size at receiver [m]: ", W_ST)

        X = (rx_radius / W_ST) ** 2
        T0 = np.sqrt(1 - np.exp(-2 * X))
        l = (
            8
            * X
            * np.exp(-4 * X)
            * i1(4 * X)
            / (1 - np.exp(-4 * X) * i0(4 * X))
            / np.log(2 * T0**2 / (1 - np.exp(-4 * X) * i0(4 * X)))
        )
        R = rx_radius * np.log(2 * T0**2 / (1 - np.exp(-4 * X) * i0(4 * X))) ** (-1.0 / l)

        lognormal_params = lambda r: self._compute_lognormal_parameters(r, R, l, W_ST, scint_index)

        return lognormal_params, wandering_var, W_LT

    def _compute_pdt(self, eta, length):
        """Compute probability distribution of atmospheric transmittance (PDT) [Vasylyev et al., 2018].

        ## Parameters
        `eta` : np.ndarray
            Input random variable values to calculate PDT for.
        `length` : float
            Length of the channel [km].
        ## Returns
        `integral` : np.ndarray
            PDT function for input eta.
        """
        lognormal_params, wandering_var, W_LT = self._compute_pdt_parameters(length)
        if wandering_var == 0:
            pdt = truncated_lognormal_pdf(eta, lognormal_params(0)[0], lognormal_params(0)[1])
        else:
            integrand = (
                lambda r: r
                * truncated_lognormal_pdf(eta, lognormal_params(r)[0], lognormal_params(r)[1])
                * np.exp(-(r**2) / (2 * wandering_var))
                / wandering_var
            )
            pdt = quad_vec(integrand, 0, self.rx_aperture / 2 + W_LT)[0]
        return pdt

    def _compute_conversion_matrix(self, j_max):
        """Compute conversion matrix [Canuet et al., 2019]."""
        Z = smf.compute_zernike(j_max)
        CZZ = smf.calculate_CZZ(
            self.rx_aperture / 2, self.rx_aperture / 2, Z, j_max, self.obs_ratio
        )
        M = smf.compute_conversion_matrix(CZZ)
        return M

    def _compute_attenuation_factors(self):
        """Compute attenuation factors of turbulent phase mode variances up to maximum order of correction n_max [Roddier, 1999].

        ## Returns
        `gamma_j` : np.ndarray
            Attenuation factors.
        """

        n = lut_zernike_index_pd["n"]
        n = np.array(lut_zernike_index_pd["n"].values)
        n_corrected = n[n <= self.n_max]
        open_loop_tf = (
            lambda v: self.integral_gain
            * np.exp(-self.control_delay * v)
            * (1 - np.exp(-self.integration_time * v))
            / (self.integration_time * v) ** 2
        )
        e_error = lambda v: 1 / (1 + open_loop_tf(v))
        gamma_j = np.ones_like(n, dtype=float)
        cutoff_freq = 0.3 * (n_corrected + 1) * self.wind_speed / self.rx_aperture
        for index in range(0, np.size(n_corrected)):
            if n_corrected[index] == 1:
                PSD_turbulence = lambda v: (
                    v ** (-2 / 3) if v <= cutoff_freq[index] else v ** (-17 / 3)
                )
            else:
                PSD_turbulence = lambda v: v ** (0) if v <= cutoff_freq[index] else v ** (-17 / 3)
            gamma_j[index] = (
                quad(lambda v: e_error(v) ** 2 * PSD_turbulence(v), 1e-2, np.inf)[0]
                / quad(PSD_turbulence, 1e-2, np.inf)[0]
            )

        return gamma_j

    def _compute_smf_coupling_pdf(self, eta_smf, eta_max, length):
        """Compute probability density function (PDF) of single mode fiber (SMF) coupling efficiency [Canuet et al., 2018].

        ## Parameters
        `eta_smf` : np.ndarray
            Input random variable values to calculate pdf for.
        `eta_max` : float
            Theoretical maximum coupling efficiency.
        `length` : float
            Length of the channel [km].
        ## Returns
        `smf_pdf` : np.ndarray
            SMF PDF for input eta.
        """
        z = length * 1e3
        n = np.array(lut_zernike_index_pd["n"].values)
        j_Noll_as_index = np.array(lut_zernike_index_pd["j_Noll"].values) - 2
        rytov_var = self._compute_rytov_variance_spherical()

        # Check of the condition for aperture averaging
        if rytov_var < 1:
            check = np.sqrt(self.wavelength * length * 1e3)
        else:
            check = 0.36 * np.sqrt(self.wavelength * length * 1e3) * (rytov_var ** (-3 / 5))
        if self.rx_aperture < check:
            print(
                f"Warning ! The aperture averaging hypothesis is not valid for this set of parameters (correlation width: {check}). Use bigger values of receiving aperture size"
            )

        scint_index = self._compute_scintillation_index_spherical(rytov_var, z)
        r0 = self._compute_coherence_width_gaussian(z)
        eta_s = np.exp(-np.log(1 + scint_index))
        bj2 = smf.bn2(self.rx_aperture, r0, n, self.obs_ratio)
        gamma_j = self._compute_attenuation_factors()
        bj2 = bj2 * gamma_j

        # Check if we are below the Rayleigh criterion
        bj_wvln = np.sqrt(bj2) / (2 * np.pi)
        bj_wlvn_max = np.max(bj_wvln)
        if bj_wlvn_max > 0.05:
            print(
                f" Warning ! The maximum Zernike coefficient std in wavelenghts is {bj_wlvn_max}. The SMF PDF is accurate below the Rayleigh criterion (0.05). You may need to use higher order of correction or smaller integration time of the AO system."
            )

        beta_opt = smf.beta_opt(self.obs_ratio)
        eta_smf_max = smf.eta_0(self.obs_ratio, beta_opt)
        eta_max = eta_max * eta_s * eta_smf_max
        smf_pdf = smf.compute_eta_smf_probability_distribution(eta_smf, eta_max, bj2)
        return smf_pdf

    def _compute_channel_pdf(self, eta_ch, length):
        """Compute probability density function (PDF) of free-space channel efficiency [Scriminich et al., 2022].

        ## Parameters
        `eta_ch` : np.ndarray
            Input random variable values to calculate pdf for.
        `length` : float
            Length of the channel [km].
        ## Returns
        `ch_pdf` : np.ndarray
            Channel PDF for input eta.
        """
        N = 10
        pdt = self._compute_pdt(eta_ch, length)
        pdt = pdt / np.sum(pdt)
        eta_rx = np.random.choice(eta_ch, N, p=pdt)
        integral = 0
        for index in range(0, N):
            integral = integral + self._compute_smf_coupling_pdf(eta_ch, eta_rx[index], length)
        ch_pdf = integral
        ch_pdf = self.Tatm * ch_pdf / np.sum(ch_pdf)
        return ch_pdf

    def _compute_mean_channel_efficiency(self, eta_ch, length, detector_efficiency=1):
        """Compute mean channel efficiency, including losses at the detector.

        ## Parameters
        `eta_ch` : np.ndarray
            Input random variable values to calculate pdf for.
        `length` : float
            Length of the channel [km].
        `detector_efficiency` : float
            Efficiency of detector at receiver (default 1).
        ## Returns
        `ch_pdf` : np.ndarray
            Channel PDF for input eta.
        """
        pdt = self._compute_pdt(eta_ch, length)
        pdt = pdt / np.sum(pdt)

        z = length * 1e3
        n = np.array(lut_zernike_index_pd["n"].values)
        j_Noll_as_index = np.array(lut_zernike_index_pd["j_Noll"].values) - 2

        rytov_var = self._compute_rytov_variance_spherical()

        # Check of the condition for aperture averaging
        if rytov_var < 1:
            check = np.sqrt(self.wavelength * length * 1e3)
        else:
            check = 0.36 * np.sqrt(self.wavelength * length * 1e3) * (rytov_var ** (-3 / 5))
        if self.rx_aperture < check:
            print(
                "Warning ! The aperture averaging hypothesis is not valid for this set of parameters. Use bigger values of receiving aperture size"
            )

        scint_index = self._compute_scintillation_index_spherical(rytov_var, z)
        r0 = self._compute_coherence_width_gaussian(z)
        eta_s = np.exp(-np.log(1 + scint_index))
        bj2 = smf.bn2(self.rx_aperture, r0, n, self.obs_ratio)

        gamma_j = self._compute_attenuation_factors()
        bj2 = bj2 * gamma_j

        # Check if we are below the Rayleigh criterion
        bj_wvln = np.sqrt(bj2) / (2 * np.pi)
        bj_wlvn_max = np.max(bj_wvln)
        if bj_wlvn_max > 0.05:
            print(
                f" Warning ! The maximum Zernike coefficient std in wavelenghts is {bj_wlvn_max}. The SMF PDF is accurate below the Rayleigh criterion (0.05). You may need to use higher order of correction or smaller integration time of the AO system."
            )

        beta_opt = smf.beta_opt(self.obs_ratio)
        eta_smf_max = smf.eta_0(self.obs_ratio, beta_opt)

        mean_transmittance = (
            np.sum(eta_ch * pdt)
            * self.Tatm
            * eta_s
            * eta_smf_max
            * detector_efficiency
            * smf.eta_ao(bj2)
        )
        return mean_transmittance

    def _draw_pdt_sample(self, length):
        """Draw random sample from probability distribution of atmospheric transmittance (PDT).

        ## Parameters
        `length` : float
            Length of the channel [km].
        ## Returns
        `sample` : float
            Random sample of PDT.
        """
        eta = np.linspace(1e-7, 1, 1000)
        pdt = self._compute_pdt(eta, length)
        pdt = np.abs(pdt / np.sum(pdt))
        sample = np.random.choice(eta, 1, p=pdt)
        return sample

    def _draw_smf_pdf_sample(self, length):
        """Draw random sample from probability distribution of single-mode fiber (SMF) coupling efficiency [Canuet et al., 2018].

        ## Parameters
        `length` : float
            Length of the channel [km].
        ## Returns
        `sample` : float
            Random sample of PDT.
        """
        eta = np.linspace(1e-7, 1, 1000)
        smf_pdf = self._compute_smf_coupling_pdf(eta, 1, length)
        smf_pdf = np.abs(smf_pdf / np.sum(smf_pdf))
        plt.figure()
        plt.plot(eta, smf_pdf)
        plt.show()
        sample = np.random.choice(eta, 1, p=smf_pdf)
        return sample

    def _draw_channel_pdf_sample(self, length, n_samples):
        """Draw random sample from free-space channel probability distribution [Scriminich et al., 2022].
        To be more efficient, the sample is calculated as the product of a sample from the PDT and the SMF coupling efficiency PDF,
        instead of the function of the channel PDF.

        ## Parameters
        `length` : float
            Length of the channel [km].
        `n_samples` : int
            Number of samples to return.
        ## Returns
        `sample` : float
            Random sample of channel PDF.
        """
        eta = np.linspace(1e-4, 1, 1000)
        pdt = self._compute_pdt(eta, length)
        pdt = np.abs(pdt / np.sum(pdt))
        pdt /= pdt.sum()
        smf_pdf = self._compute_smf_coupling_pdf(eta, 1, length)
        smf_pdf = np.abs(smf_pdf / np.sum(smf_pdf))
        smf_pdf /= smf_pdf.sum()
        pdt_sample = np.random.choice(eta, n_samples, p=pdt)
        smf_sample = np.random.choice(eta, n_samples, p=smf_pdf)
        sample = self.Tatm * pdt_sample * smf_sample
        return sample

    def _compute_loss_probability(self, length, n_samples):
        """Compute loss probability of photon in downlink channel, taking all losses into account.

        ## Parameters
        `length` : float
            Length of the channel [km].
        `n_samples` : int
            Number of samples to return.
        ## Returns
        `prob_loss` : float
            Probability that a photon is lost in the channel.
        """
        T = self._draw_channel_pdf_sample(length, n_samples)
        prob_loss = 1 - T
        return prob_loss
