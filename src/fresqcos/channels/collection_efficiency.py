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

    # Radial integration grid
    r = np.linspace(0, r_max, n_r)

    # Cumulative radial probability
    density = radial_pdf(r)
    cdf_r = cumulative_trapezoid(density, r, initial=0.0)

    # Transmission associated with each radial point
    eta_r = eta_max(r)

    # Check for nonphysical values before clipping
    if np.any(eta_r < -1e-12) or np.any(eta_r > 1 + 1e-12):
        raise ValueError(
            "Transmission is outside [0, 1]. "
            "Check the annular transmission approximation."
        )

    eta_r = np.clip(eta_r, 0, 1)

    # Construct transmission bin edges
    edges = np.empty(len(eta) + 1)
    edges[1:-1] = (eta[:-1] + eta[1:]) / 2
    edges[0] = max(0, eta[0] - (edges[1] - eta[0]))
    edges[-1] = min(1, eta[-1] + (eta[-1] - edges[-2]))

    # Identify monotonic branches of eta(r)
    direction = np.sign(np.diff(eta_r))

    # Handle flat sections of the numerical transmission
    nonzero = np.flatnonzero(direction)

    if len(nonzero) == 0:
        # Constant transmission: all mass belongs to one bin
        mass, _ = np.histogram(
            [eta_r[0]],
            bins=edges,
            weights=[cdf_r[-1]]
        )
        return mass / np.diff(edges)

    last_nonzero = np.maximum.accumulate(
        np.where(
            direction != 0,
            np.arange(len(direction)),
            -1
        )
    )

    direction = direction[
        np.where(last_nonzero >= 0, last_nonzero, nonzero[0])
    ]

    turning_points = np.where(
        np.diff(direction) != 0
    )[0] + 1

    bounds = np.r_[0, turning_points, len(r) - 1]

    # Accumulate probability mass from every monotonic branch
    mass = np.zeros(len(eta), dtype=float)

    for start, stop in zip(bounds[:-1], bounds[1:]):

        eta_branch = eta_r[start:stop + 1]
        cdf_branch = cdf_r[start:stop + 1]

        if eta_branch[0] < eta_branch[-1]:
            # Increasing transmission
            cdf_edges = np.interp(
                edges,
                eta_branch,
                cdf_branch,
                left=cdf_branch[0],
                right=cdf_branch[-1]
            )

        else:
            # Decreasing transmission
            cdf_edges = np.interp(
                edges,
                eta_branch[::-1],
                cdf_branch[::-1],
                left=cdf_branch[-1],
                right=cdf_branch[0]
            )

        mass += np.abs(np.diff(cdf_edges))

    # Convert probability mass to probability density
    pdf = mass / np.diff(edges)

    return pdf

def compute_collection_efficiency_pdf_annular_weak_turbulence_roots(
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

    if not (0 <= obstr_radius < rx_radius):
        raise ValueError("Require 0 <= obstr_radius < rx_radius.")

    if waist_radius <= 0 or r_max <= 0 or n_r < 100:
        raise ValueError("Invalid waist_radius, r_max or n_r.")

    pdf = np.zeros_like(eta)

    # Outer aperture parameters
    eta_maxer_outer, shape_outer, scale_outer = (
        compute_circular_aperture_parameters(
            rx_radius, waist_radius
        )
    )

    b_outer = rx_radius * scale_outer

    # Inner aperture parameters
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

    def deta_max_dr(r):
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

    # Combine linear and logarithmic grids to resolve
    # both ordinary and very small radial distances.
    r_scan = np.unique(np.r_[
        np.linspace(0, r_max, n_r // 2),
        np.geomspace(r_max * 1e-12, r_max, n_r // 2)
    ])

    # Evaluate derivative only at positive radii
    r_positive = r_scan[r_scan > 0]

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        under="ignore",
        over="ignore"
    ):
        slopes = deta_max_dr(r_positive)

    # Ignore zero/invalid derivative samples
    valid = np.flatnonzero(
        np.isfinite(slopes) & (slopes != 0)
    )

    turning_points = []

    for j, k in zip(valid[:-1], valid[1:]):

        # A sign change indicates a stationary point
        if np.signbit(slopes[j]) != np.signbit(slopes[k]):

            root = brentq(
                deta_max_dr,
                r_positive[j],
                r_positive[k],
                xtol=5e-324,
                rtol=1e-12
            )

            turning_points.append(root)

    # Split radial domain into monotonic intervals
    bounds = np.r_[
        0.0,
        turning_points,
        r_max
    ]

    eta_bounds = eta_max(bounds)


    for i, eta_value in np.ndenumerate(eta):

        if not (0 < eta_value < 1):
            continue

        # At an interior stationary point, the PDF
        # can have an integrable divergence.
        if np.any(eta_value == eta_bounds[1:-1]):
            pdf[i] = np.inf
            continue

        # Search every monotonic interval
        for j in range(len(bounds) - 1):

            r_left = bounds[j]
            r_right = bounds[j + 1]

            eta_left = eta_bounds[j]
            eta_right = eta_bounds[j + 1]

            # Check whether the desired eta lies
            # within this branch's transmission range
            if not (
                min(eta_left, eta_right)
                < eta_value
                < max(eta_left, eta_right)
            ):
                continue

            # Find radial displacement producing eta
            r_root = brentq(
                lambda r: eta_max(r) - eta_value,
                r_left,
                r_right,
                xtol=5e-324,
                rtol=1e-12
            )

            # Jacobian of the transformation
            derivative = deta_max_dr(r_root)

            if derivative == 0:
                pdf[i] = np.nan
                break

            # Add contribution from this root
            pdf[i] += (
                radial_pdf(r_root)
                / abs(derivative)
            )

    return pdf
