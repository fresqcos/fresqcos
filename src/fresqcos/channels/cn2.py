"""Module for computing Cn2 profile."""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def hufnagel_valley(altitude: FloatArray, wind_speed_rms: FloatArray, reference_ground: float):
    """
    Computes the refractive index structure constant (Cn2) profile
    based on the Hufnagel-Valley model.

    Parameters
    ----------
    altitude : FloatArray
        Altitude from ground level.
    wind_speed_rms : FloatArray
        RMS of transverse wind speed.
    reference_ground : float
        Reference Cn2 at ground level.

    Returns
    -------
    FloatArray
        An array of Cn2 values corresponding to the input altitudes.
    """
    cn2 = (
        0.00594
        * ((wind_speed_rms / 27) ** 2)
        * ((altitude * 10 ** (-5)) ** 10)
        * np.exp(-altitude / 1000)
        + (2.7 * 10 ** (-16)) * np.exp(-altitude / 1500)
        + reference_ground * np.exp(-altitude / 100)
    )
    return cn2
