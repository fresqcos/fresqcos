"""Module for useful probability distributions"""

import numpy as np
from scipy.special import erf


def lognormal_pdf(eta, mu, sigma):
    """Compute lognormal distribution probability density function (PDF).

    ## Parameters
    `eta` : np.ndarray
        Input random variable values to calculate PDF for.
    `mu` : float
        Mean value of lognormal distribution.
    `sigma` : float
        Standard deviation of lognormal distribution.
    ## Returns
    `pdf` : np.ndarray
        PDF of lognormal distribution for values of eta.
    """

    pdf = np.exp(-((np.log(eta) + mu) ** 2) / (2 * sigma**2)) / (eta * sigma * np.sqrt(2 * np.pi))
    return pdf


def lognormal_cdf(eta, mu, sigma):
    """Compute lognormal distribution cumulative density function (CDF).

    ## Parameters
    `eta` : np.ndarray
        Input random variable values to calculate CDF for.
    `mu` : float
        Mean value of lognormal distribution.
    `sigma` : float
        Standard deviation of lognormal distribution.
    ## Returns
    `cdf` : np.ndarray
        CDF of lognormal distribution for values of eta.
    """
    cdf = (1 + erf((np.log(eta) + mu) / (sigma * np.sqrt(2)))) / 2
    return cdf


def truncated_lognormal_pdf(eta, mu, sigma):
    """Compute truncated lognormal distribution probability density function
    (PDF) according to [Vasylyev et al., 2018].

    ## Parameters
    `eta` : np.ndarray
        Input random variable values to calculate PDF for.
    `mu` : float
        Mean value of truncated lognormal distribution.
    `sigma` : float
        Standard deviation of truncated lognormal distribution.
    ## Returns
    `pdf` : np.ndarray
        PDF of truncated lognormal distribution for values of eta.
    """
    lognormal_cdf_dif = lognormal_cdf(1, mu, sigma)
    if np.size(eta) == 1:
        if eta < 0 or eta > 1:
            pdf = 0
        else:
            pdf = lognormal_pdf(eta, mu, sigma) / lognormal_cdf_dif
    else:
        pdf = np.zeros(np.size(eta))
        eta_domain = (eta >= 0) & (eta <= 1)
        pdf[eta_domain] = lognormal_pdf(eta[eta_domain], mu, sigma) / lognormal_cdf_dif
    return pdf


def lognegative_weibull_pdf(eta, eta_0, wandering_variance, r, l):
    """Compute log-negative Weiibull distribution probability density
    function (PDF) according to [Vasylyev et al., 2018].

    ## Parameters
    `eta` : np.ndarray
        Input random variable values to calculate PDF for.
    `eta_0` : float
        Maximal transmittance of the Gaussian beam.
    `wandering_variance` : float
        Wandering variance of the Gaussian beam.
    `r` : float
        Scale parameter of distribution.
    `l` : float
        Shape parameter of distribution.
    ## Returns
    `pdf` : np.ndarray
        PDF of log-negative Weibull distribution for values of eta.
    """
    if np.size(eta) == 1:
        if eta < 0 or eta > eta_0:
            pdf = 0
        else:
            pdf = (
                (r**2 / (wandering_variance * eta * l))
                * ((np.log(eta_0 / eta)) ** (2 / l - 1))
                * np.exp(-(r**2 / (2 * wandering_variance)) * (np.log(eta_0 / eta)) ** (2 / l))
            )
    else:
        pdf = np.zeros(np.size(eta))
        eta_domain = (eta >= 0) & (eta <= eta_0)
        pdf[eta_domain] = (
            (r**2 / (wandering_variance * eta[eta_domain] * l))
            * ((np.log(eta_0 / eta[eta_domain])) ** (2 / l - 1))
            * np.exp(
                -(r**2 / (2 * wandering_variance)) * (np.log(eta_0 / eta[eta_domain])) ** (2 / l)
            )
        )
    return pdf
