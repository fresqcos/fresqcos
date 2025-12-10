"""Module for free-space communication channels."""

import numpy as np
from scipy.special import i0, i1
from scipy.integrate import quad
from probability_distributions import lognegative_weibull_pdf
import fiber_coupling as smf
import pandas as pd

# Zernike indices look-up table
MAX_N_WFS = 150  # Maximum radial index n returned by WFS
array_of_zernike_index = smf.get_zernikes_index_range(MAX_N_WFS)
lut_zernike_index_pd = pd.DataFrame(array_of_zernike_index[1:], columns=["n", "m", "j"])
lut_zernike_index_pd["j_Noll"] = smf.calculate_j_noll(
    lut_zernike_index_pd["n"], lut_zernike_index_pd["m"]
)


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
