"""Module for calculating the collection efficiency of a free-space optical communication system."""

import numpy as np
from scipy.special import i0e, i1e
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq


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

    eta = np.asarray(eta, dtype=float)
    pdf = np.zeros_like(eta)

    mask = (eta > 0) & (eta < eta_max)
    eta_valid = eta[mask]

    x = np.log(eta_max / eta_valid)

    # Physical radial scale
    b = rx_radius * scale

    pdf[mask] = (
        b
        / (shape * eta_valid)
        * x ** (1 / shape - 1)
        * radial_pdf(b * x ** (1 / shape))
    )

    return pdf


def compute_collection_efficiency_pdf_annular_weak_turbulence(
    eta: np.ndarray,
    radial_pdf: callable,
    rx_radius: float,
    obstr_radius: float,
    waist_radius: float,
    r_max: float = 5,
    n_r: int = 10000,
) -> np.ndarray:
    """
    Compute the collection efficiency PDF for an annular aperture
    using the Dirac delta root-finding method.

    Parameters
    ----------
    eta : np.ndarray
        Collection efficiency values in [0, 1].
    radial_pdf : callable
        Normalized radial PDF, e.g. Hoyt, Rayleigh or Rice.
    rx_radius : float
        Outer receiver aperture radius.
    obstr_radius : float
        Central obstruction radius.
    waist_radius : float
        Short-term beam waist radius at the receiver.
    r_max : float
        Maximum radial displacement considered.
    n_r : int
        Number of sampling points used to locate stationary
        points of the transmission function.

    Returns
    -------
    pdf : np.ndarray
        PDF of collection efficiency for values of eta.
    """

    eta = np.asarray(eta, dtype=float)

    if not 0 <= obstr_radius < rx_radius:
        raise ValueError("Require 0 <= obstr_radius < rx_radius.")

    if waist_radius <= 0 or r_max <= 0 or n_r < 100:
        raise ValueError("Invalid waist_radius, r_max or n_r.")

    eta_maxer_outer, shape_outer, scale_outer = (
        compute_circular_aperture_parameters(
            rx_radius, waist_radius
        )
    )

    b_outer = rx_radius * scale_outer

    if obstr_radius > 0:
        eta_max_inner, shape_inner, scale_inner = (
            compute_circular_aperture_parameters(
                obstr_radius, waist_radius
            )
        )
        b_inner = obstr_radius * scale_inner

    def eta_max(r):
        outer = eta_maxer_outer * np.exp(
            -(r / b_outer)**shape_outer
        )

        if obstr_radius == 0:
            return outer

        inner = eta_max_inner * np.exp(
            -(r / b_inner)**shape_inner
        )

        return outer - inner

    def eta_max_deriv(r):
        outer = (
            -eta_maxer_outer * shape_outer / b_outer
            * (r / b_outer)**(shape_outer - 1)
            * np.exp(-(r / b_outer)**shape_outer)
        )

        if obstr_radius == 0:
            return outer

        inner = (
            eta_max_inner * shape_inner / b_inner
            * (r / b_inner)**(shape_inner - 1)
            * np.exp(-(r / b_inner)**shape_inner)
        )

        return outer + inner

    radial_samples = np.unique(np.concatenate([
        np.linspace(0, r_max, n_r // 2),
        np.geomspace(r_max * 1e-12, r_max, n_r // 2)
    ]))

    derivatives = eta_max_deriv(radial_samples)

    valid = np.isfinite(derivatives) & (derivatives != 0)
    radial_samples = radial_samples[valid]
    derivatives = derivatives[valid]

    sign_changes = np.flatnonzero(
        np.signbit(derivatives[:-1]) != np.signbit(derivatives[1:])
    )

    stationary_points = [
        brentq(
            eta_max_deriv,
            radial_samples[i],
            radial_samples[i + 1],
            xtol=1e-12 * r_max
        )
        for i in sign_changes
    ]

    radial_boundaries = np.concatenate([[0], stationary_points, [r_max]])
    eta_max_boundaries = eta_max(radial_boundaries)

    pdf = np.zeros_like(eta)

    for index, efficiency in enumerate(eta):

        if not 0 < efficiency < 1:
            continue

        if np.any(efficiency == eta_max_boundaries[1:-1]):
            pdf[index] = np.inf
            continue

        for i in range(len(radial_boundaries) - 1):

            radius_left = radial_boundaries[i]
            radius_right = radial_boundaries[i + 1]

            efficiency_left = eta_max_boundaries[i]
            efficiency_right = eta_max_boundaries[i + 1]

            if not (
                min(efficiency_left, efficiency_right)
                < efficiency
                < max(efficiency_left, efficiency_right)
            ):
                continue

            radius = brentq(
                lambda r: eta_max(r) - efficiency,
                radius_left,
                radius_right,
                xtol=1e-12 * r_max
            )

            slope = abs(eta_max_deriv(radius))

            pdf[index] += (
                radial_pdf(radius) / slope
                if slope > 0 else np.inf
            )

    return pdf
