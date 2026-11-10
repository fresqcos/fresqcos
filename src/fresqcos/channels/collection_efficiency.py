"""Module for calculating the collection efficiency of a free-space optical communication system."""

import numpy as np
from scipy.special import i0e, i1e


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
    shape : float
        The shape parameter.
    scale : float
        The scale parameter.
    """
    x = (rx_radius / waist_radius) ** 2
    maximal_transmittance = 1 - np.exp(-2 * x)
    shape = (
        8
        * x
        * i1e(4 * x)
        / (1 - i0e(4 * x))
        / np.log(2 * maximal_transmittance / (1 - i0e(4 * x)))
    )
    scale = rx_radius * np.log(2 * maximal_transmittance / (1 - i0e(4 * x))) ** (
        -1 / shape
    )

    return maximal_transmittance, shape, scale


def compute_collection_efficiency_pdf_circular_weak_turbulence(
    eta: np.ndarray, radial_pdf: callable, rx_radius: float, waist_radius: float
) -> np.ndarray:
    """
    Compute the collection efficiency probability density function (PDF) for a
    circular, unobstructed aperture under weak turbulence conditions, when pointing
    error is prevalent.

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
    eta_max, shape, scale = compute_circular_aperture_parameters(
        rx_radius, waist_radius
    )
    pdf = np.zeros_like(eta)
    mask = (eta > 0) & (eta < eta_max)
    eta_valid = eta[mask]
    x = np.log(eta_max / eta_valid)
    pdf[mask] = (
        scale
        / (shape * eta_valid)
        * x ** (1 / shape - 1)
        * radial_pdf(scale * x ** (1 / shape))
    )
    return pdf
