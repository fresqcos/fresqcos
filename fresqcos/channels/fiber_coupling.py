"""Module for calculating single-mode fiber coupling efficiency in free-space optical channels
with atmospheric turbulence, based on the model presented in Scriminich et al. 2022.
"""

import math
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import gamma, hyp2f1
from scipy.special import factorial as fac
from numpy.typing import NDArray

IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]


def get_zernikes_index_range(n_max: int) -> list:
    """
    Returns a list of Zernike indexes.

    Parameters
    ----------
    n_max : int
        The maximum Zernike radial index to be returned.

    Returns
    -------
    list
        A list of indexes in the format [[n, m, j]], where
        n is the radial, m is the azimuth and j is the incremental.
    """
    out = []
    j = 0
    for n in range(n_max + 1):
        for m in np.arange(-n, n + 1, 1):
            if (n - np.abs(m)) % 2 == 0:
                out.append([n, m, j])
                j += 1
    return out


def calculate_j_noll(n: IntArray, m: IntArray) -> IntArray:
    """
    Calculates the Noll index j for given Zernike radial (n) and azimuthal (m) indexes.

    Parameters
    ----------
    n : IntArray
        Zernike radial index.
    m : IntArray
        Zernike azimuthal index.

    Returns
    -------
    IntArray
        Noll index j.
    """
    j = n * (n + 1) // 2 + abs(m)

    # Determine the appropriate case for the piecewise function
    case1 = ((m >= 0) & (n % 4 == 2)) | ((m >= 0) & (n % 4 == 3))
    case2 = ((m <= 0) & (n % 4 == 0)) | ((m <= 0) & (n % 4 == 1))

    # Apply conditions
    mask = case1 | case2
    j[mask] += 1

    return j


def eta_smf_max(alpha: float, beta: float) -> float:
    """
    Computes the smf coupling efficiency without turbulence, Eq. 19 Scriminich22.

    Parameters
    ----------
    alpha : float
        Alpha parameter.
    beta : float
        Beta parameter.

    Returns
    -------
    float
        Value of eta_0.
    """
    eta_0 = (
        2
        * (
            (math.exp(-(beta**2)) - math.exp(-(beta**2) * alpha**2))
            / (beta * math.sqrt(1 - alpha**2))
        )
        ** 2
    )
    return eta_0


def beta_param(rx_diameter: float, mfd: float, lmbd: float, f: float) -> float:
    """
    Calculates beta parameter for the computation of eta_0, Eq. 20 Scriminich22.

    Parameters
    ----------
    rx_diameter : float
        Size of the receiver aperture at the lens before the fiber.
    mfd : float
        Mode field diameter of the fiber.
    lmbd : float
        Wavelength of the beam.
    f : float
        Focal length before the fiber.

    Returns
    -------
    float
        Value of beta.
    """

    beta = np.pi * rx_diameter * mfd / 4 / lmbd / f
    return beta


def beta_opt(alpha: float) -> float:
    """
    Compute the optimal beta parameter given alpha.

    Parameters
    ----------
    alpha : float
        Alpha parameter.

    Returns
    -------
    float
        Value of beta
    """

    return 1.22 * math.exp(-0.55 * alpha) - 0.1 * math.exp(-8 * alpha)


def geom_factor(n: int | IntArray, beta: float = 11 / 3) -> FloatArray:
    """
    Compute the geometrical factor to be used for b_n in Eq. 22 Scriminich22.

    Parameters
    ----------
    n : int or IntArray
        Zernike radial index.
    beta : float
        Beta parameter of the turbulence spectrum (default Kolmogorov = 11/3).

    Returns
    -------
    FloatArray
        Value(s) of the geometrical factor.
    """

    n = np.asarray(n)
    g = (
        (n + 1)
        / math.pi
        * gamma((beta + 4) / 2)
        * gamma(beta / 2)
        * gamma((2 * n + 2 - beta) / 2)
        * math.sin(math.pi * (beta - 2) / 2)
        / gamma((2 * n + 4 + beta) / 2)
    )
    return FloatArray(g)


def bn2_zernike(rx_diameter: float, r_0: float, n: int | IntArray) -> FloatArray:
    """
    Compute the Zernike coefficient of order n, Eq. 22 Scriminich22.

    Parameters
    ----------
    rx_diameter : float
        Size of the receiver aperture.
    r_0 : float
        Value of fried parameter.
    n : int or IntArray
        Zernike radial index.

    Returns
    -------
    FloatArray
        Value(s) of bn2.
    """

    zernikes = (rx_diameter / r_0) ** (5 / 3) * geom_factor(n)
    return FloatArray(zernikes)


def bn2(rx_diameter: float, r_0: float, n: int | IntArray, obs: float) -> FloatArray:
    """
    Compute the annular coefficient of order n [Dai and Mahajan 2007, eq. 39].

    Parameters
    ----------
    rx_diameter : float
        Size of the receiver aperture.
    r_0 : float
        Value of fried parameter.
    n : int or IntArray
        Zernike radial index.
    obs : float
        Obstruction ratio of receiver aperture.

    Returns
    -------
    FloatArray
        Value(s) of bn2.
    """

    n = np.asarray(n)
    pi = np.pi
    constant_term = 0.023 * (pi ** (8 / 3)) / (2 ** (5 / 3) * gamma(17 / 6))

    part1 = (n + 1) * gamma(n - 5 / 6) / ((1 - obs**2) * (1 - obs ** (2 * (n + 1))))
    part1 *= (rx_diameter / r_0) ** (5 / 3)

    term1 = (1 + obs ** (2 * n + 17 / 3)) * gamma(14 / 3)
    term1 /= gamma(17 / 6) * gamma(n + 23 / 6)

    term2 = (2 * obs ** (2 * (n + 1))) / fac(n + 1)
    term2 *= hyp2f1(n - 5 / 6, -11 / 6, n + 2, obs**2)

    part2 = term1 - term2

    result = constant_term * part1 * part2

    return FloatArray(result)


def eta_ao(bj2: list) -> float:
    """
    Compute the smf coupling efficiency with turbulence, Eq. 24 Scriminich22.

    Parameters
    ----------
    bj2 : list
        List of Zernike coefficients (without order 0).

    Returns
    -------
    float
        Value of eta_ao.
    """

    eta = np.prod(1 / np.sqrt(1 + 2 * np.array(bj2)))
    return float(eta)


def eta_s(scint_index: float) -> float:
    """
    Compute the impact of scintillation on the smf coupling efficiency.

    Parameters
    ----------
    scint_index : float
        Scintillation index.

    Returns
    -------
    float
        Value of eta_s.
    """
    eta = (1 + scint_index) ** (-1 / 4)
    return float(eta)


def compute_eta_xi_probability_distribution(
    xi: float | FloatArray, bj2: float | FloatArray
) -> FloatArray:
    """
    Compute the probability distribution of xi, Eq. 33 Scriminich22

    Parameters
    ----------
    xi : float or FloatArray
        Input parameter for the probability distribution.
    bj2 : float or FloatArray
        Zernike coefficients squared (without order 0).

    Returns
    -------
    float
        Value(s) of p_xi(xi).
    """

    xi = np.asarray(xi)
    bj2 = np.asarray(bj2)

    def integrand(x):
        """
        Integrand function for the probability distribution of xi.
        """
        expression = np.cos(np.sum((0.5 * np.arctan(2 * bj2 * x))) - xi * x) / (
            np.prod(1 + (4 * (x) ** 2 * bj2**2)) ** 0.25
        )
        return expression

    integral = quad_vec(integrand, 0, np.inf)[0]
    result = integral / np.pi

    return FloatArray(result)


def compute_eta_smf_probability_distribution(
    eta_smf: float | FloatArray, eta_max: float, bj2: float | FloatArray
) -> FloatArray:
    """
    Compute the probability distribution of eta_smf, Eq. 34 Scriminich22.

    Parameters
    ----------
    eta_smf : float or FloatArray
        input parameter for the probability distribution.
    eta_max : float
        Maximum normalized coupled flux computed as eta_0*eta_S.
    bj2 : float or FloatArray
        Zernike coefficients squared (without order 0).

    Returns
    -------
    float
        Value(s) of p_smf(eta_smf).
    """

    eta_smf = np.asarray(eta_smf)
    bj2 = np.asarray(bj2)
    result = compute_eta_xi_probability_distribution(np.log(eta_max / eta_smf), bj2) / eta_smf

    return FloatArray(result)
