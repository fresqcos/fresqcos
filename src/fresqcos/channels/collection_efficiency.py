"""Module for calculating the collection efficiency of a free-space optical communication system."""

import numpy as np
from scipy.special import i0, i1


def compute_circular_aperture_parameters(
    rx_radius: float, waist_radius: float
) -> tuple[float, float, float]:
    """
    Compute the parameters useful to calculate the collection efficiency for
    a circular, unobstructed aperture.

    Parameters
    ----------
    rx_radius : float
        The radius of the receiver aperture.
    waist_radius : float
        The short-term radius of the beam waist at the receiver.

    Returns
    -------
    maximal_transmittance : float
        The maximal transmittance of the circular aperture.
    scale : float
        The scale parameter.
    shape : float
        The shape parameter.
    """
    x = (rx_radius / waist_radius) ** 2
    maximal_transmittance = np.sqrt(1 - np.exp(-2 * x))
    scale = (
        8
        * x
        * np.exp(-4 * x)
        * i1(4 * x)
        / (1 - np.exp(-4 * x) * i0(4 * x))
        / np.log(2 * maximal_transmittance**2 / (1 - np.exp(-4 * x) * i0(4 * x)))
    )
    shape = rx_radius * np.log(
        2 * maximal_transmittance**2 / (1 - np.exp(-4 * x) * i0(4 * x))
    ) ** (-1.0 / scale)

    return maximal_transmittance, scale, shape


def compute_collection_efficiency_pdf_circular_weak_turbulence(
    eta: np.ndarray, radial_pdf: callable, rx_radius: float, waist_radius: float
) -> np.ndarray:
    """
    Compute the collection efficiency probability density function (PDF) for a
    circular, unobstructed aperture under weak turbulence conditions.

    Parameters
    ----------
    eta : np.ndarray
        Input random variable values to calculate PDF for.
    radial_pdf : callable
        The radial PDF of the beam wandering.
    rx_radius : float
        The radius of the receiver aperture.
    waist_radius : float
        The short-term radius of the beam waist at the receiver.

    Returns
    -------
    pdf : np.ndarray
        PDF of collection efficiency for values of eta.
    """
    maximal_transmittance, scale, shape = compute_circular_aperture_parameters(
        rx_radius, waist_radius
    )
    x = 2 * np.log(maximal_transmittance / eta)
    pdf = (
        2 * scale / (shape * eta) * x ** (1 / scale - 1) * radial_pdf(x ** (1 / scale))
    )
    return pdf
