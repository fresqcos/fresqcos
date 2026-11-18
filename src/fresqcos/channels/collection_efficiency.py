"""Module for calculating the collection efficiency of a free-space optical communication system."""

import numpy as np
from scipy.special import i0e, i1e
from scipy.integrate import quad


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


def compute_collection_efficiency_pdf_annular_weak_turbulence(
    eta: np.ndarray,
    radial_pdf: callable,
    rx_radius: float,
    obstr_radius: float,
    waist_radius: float,
    r_max: float = 5,
    n_r: int = 400_000,
) -> np.ndarray:
    """
    Compute the collection efficiency probability density function (PDF) for an
    annular aperture with a central obstruction under weak turbulence conditions,
    when pointing error is
    prevalent.

    Parameters
    ----------
    eta : np.ndarray
        Input random variable values to calculate PDF for.
    radial_pdf : callable
        The radial PDF of the beam wandering.
    rx_radius : float
        The radius of the receiver aperture.
    obstr_radius : float
        The radius of the central obstruction.
    waist_radius : float
        The short-term radius of the beam waist at the receiver.
    r_max : float, optional
        The maximum radial distance to consider for the integration. Default is 5.
    n_r : int, optional
        The number of radial points to use for the integration. Default is 400_000.

    Returns
    -------
    pdf : np.ndarray
        PDF of collection efficiency for values of eta.
    """
    eta_max_outer, shape_outer, scale_outer = compute_circular_aperture_parameters(
        rx_radius, waist_radius
    )
    if obstr_radius > 0:
        eta_max_inner, shape_inner, scale_inner = compute_circular_aperture_parameters(
            obstr_radius, waist_radius
        )
    outer = lambda r: eta_max_outer * np.exp(
        -((r / rx_radius / scale_outer) ** shape_outer)
    )
    if obstr_radius == 0:
        eta_max = outer
    else:
        inner = lambda r: eta_max_inner * np.exp(
            -((r / obstr_radius / scale_inner) ** shape_inner)
        )
        eta_max = lambda r: outer(r) - inner(r)

    r = np.linspace(0, r_max, n_r)
    dr = r[1] - r[0]
    density = radial_pdf(r)
    weights = density * dr

    print("Radial probability:", np.trapezoid(density, r))

    weights[0] *= 0.5
    weights[-1] *= 0.5

    eta_r = eta_max(r)

    print("Minimum transmission:", np.min(eta_r))
    print("Maximum transmission:", np.max(eta_r))

    if np.any(eta_r < -1e-10):
        raise ValueError("Annular transmission becomes significantly negative.")

    eta_r = np.clip(eta_r, 0, 1)

    edges = np.empty(len(eta) + 1)
    edges[1:-1] = (eta[:-1] + eta[1:]) / 2
    edges[0] = max(0, eta[0] - (edges[1] - eta[0]))
    edges[-1] = min(1, eta[-1] + (eta[-1] - edges[-2]))
    hist, edges = np.histogram(eta_r, bins=edges, weights=weights)

    pdf = hist / np.diff(edges)

    return pdf
