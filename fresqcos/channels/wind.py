
"""Module for wind speed profiles."""

import numpy as np


def greenwood_wind(h, v_g=8, v_t=30, zeta=0, h_t=12448, l_t=4800, phi=0, omega_s=0):
    """
    Wind velocity profile based on [Greenwood, 1977], [Hardy, 1998] and [Andrews, 2005].

    Parameters
    ----------
    h : float or array_like
        Height above ground.
    v_g : float
        Wind velocity close to the ground.
    v_t : float
        Wind velocity at the tropopause.
    zeta : float
        Zenith angle (radians).
    h_t : float
        Height of the tropopause.
    l_t : float
        Thickness of the tropopause layer.
    phi : float
        Wind direction relative to telescope azimuth (default = 0).
    omega_s : float, optional
        Satellite slew rate (default = 0).

    Returns
    -------
    v : float or ndarray
        Wind velocity at height h.
    """

    exp_term = np.exp(-(((h - h_t) / l_t) ** 2))

    angular_factor = np.sqrt(np.sin(phi) ** 2 + np.cos(phi) ** 2 * np.cos(zeta) ** 2)

    v = v_g + v_t * exp_term * angular_factor + omega_s * h
    return v
