"""Module for geometric calculations related to the channel between a ground station and an aerial platform."""

import numpy as np
import math

EARTH_RADIUS_KM = 6371


def slant_range_from_zenith_angle(observer_alt: float, target_alt: float, zenith_angle: float):
    """Compute slant range that corresponds to a particular observer altitude, target
    altitude and zenith angle.

    Parameters
    ----------
    observer_alt : float
        Altitude of the observer [km].
    target_alt : float
        Altitude of the target [km].
    zenith_angle : float
        Zenith angle of the target measured at observer [degrees].

    Returns
    -------
    channel_length : float
        Slant range of the channel [km].
    """
    zenith_angle = np.deg2rad(zenith_angle)
    target_eff_alt = EARTH_RADIUS_KM + target_alt
    observer_eff_alt = EARTH_RADIUS_KM + observer_alt
    channel_length = np.sqrt(
        target_eff_alt**2 + observer_eff_alt**2 * (np.cos(zenith_angle) ** 2 - 1)
    ) - observer_eff_alt * np.cos(zenith_angle)

    return channel_length


def compute_minimum_alt(length: float, height: float):
    """Compute minimum altitude of a horizontal channel between two aerial platforms at the same height.

    Parameters
    ----------
    length : float
        Length of the horizontal channel [km].
    height : float
        Height of the aerial platforms [km].

    Returns
    -------
    h_min : float
        Minimum altitude of the channel [km].
    """
    aerial_platform_eff_alt = EARTH_RADIUS_KM + height
    theta = np.arcsin((length) / (2 * aerial_platform_eff_alt))
    h_min = np.cos(theta) * aerial_platform_eff_alt - EARTH_RADIUS_KM
    return h_min


def compute_sec(theta: float):
    """Compute secant of angle theta.

    Parameters
    ----------
    theta : float
        Angle for which secant will be calculated [degrees].

    Returns
    -------
    sec : float
        Secant result.
    """
    theta = np.deg2rad(theta)
    sec = 1 / np.cos(theta)
    return sec


def compute_central_angle(lat_1: float, lon_1: float, lat_2: float, lon_2: float):
    """Compute central angle between two points on the Earth's surface given their latitudes and longitudes.

    Parameters
    ----------
    lat_1 : float
        Latitude of the first point [degrees].
    lon_1 : float
        Longitude of the first point [degrees].
    lat_2 : float
        Latitude of the second point [degrees].
    lon_2 : float
        Longitude of the second point [degrees].

    Returns
    -------
    angle : float
        Central angle between the two points [degrees].
    """
    phi_1 = np.deg2rad(lat_1)
    phi_2 = np.deg2rad(lat_2)
    Delta_lambda = np.deg2rad(lon_2 - lon_1)

    angle = np.arccos(
        np.sin(phi_1) * np.sin(phi_2) + np.cos(phi_1) * np.cos(phi_2) * np.cos(Delta_lambda)
    )
    return np.rad2deg(angle)


def slant_range_from_central_angle(alt_1: float, alt_2: float, central_angle: float):
    """Compute slant range from two altitudes and a central angle.

    Parameters
    ----------
    alt_1 : float
        Altitude of the first point [km].
    alt_2 : float
        Altitude of the second point [km].
    central_angle : float
        Central angle between the two points [degrees].

    Returns
    -------
    channel_length : float
        Slant range of the channel [km].
    """
    eff_alt_1 = EARTH_RADIUS_KM + alt_1
    eff_alt_2 = EARTH_RADIUS_KM + alt_2
    channel_length = np.sqrt(
        eff_alt_1**2 + eff_alt_2**2 - 2 * eff_alt_1 * eff_alt_2 * np.cos(np.deg2rad(central_angle))
    )
    return channel_length


def slant_range_from_coordinates(
    lat_1: float, lon_1: float, alt_1: float, lat_2: float, lon_2: float, alt_2: float
):
    """Compute slant range between two points given their coordinates.
    Parameters
    ----------
    lat_1 : float
        Latitude of the first point [degrees].
    lon_1 : float
        Longitude of the first point [degrees].
    alt_1 : float
        Altitude of the first point [km].
    lat_2 : float
        Latitude of the second point [degrees].
    lon_2 : float
        Longitude of the second point [degrees].
    alt_2 : float
        Altitude of the second point [km].

    Returns
    -------
    channel_length : float
        Slant range of the channel [km].
    """
    central_angle = compute_central_angle(lat_1, lon_1, lat_2, lon_2)

    eff_alt_1 = EARTH_RADIUS_KM + alt_1
    eff_alt_2 = EARTH_RADIUS_KM + alt_2

    channel_length = math.sqrt(
        eff_alt_1**2 + eff_alt_2**2 - 2 * eff_alt_1 * eff_alt_2 * np.cos(np.deg2rad(central_angle))
    )
    return channel_length


def zenith_angle_from_slant_range(slant_range: float, observer_alt: float, target_alt: float):
    """Compute zenith angle at the observer from slant range with target.

    Parameters
    ----------
    slant_range : float
        Slant range of the channel [km].
    observer_alt : float
        Altitude of the observer [km].
    target_alt : float
        Altitude of the target [km].

    Returns
    -------
    zenith_angle : float
        Zenith angle at the observer's location [degrees].
    """
    observer_eff_alt = EARTH_RADIUS_KM + observer_alt
    target_eff_alt = EARTH_RADIUS_KM + target_alt
    zenith_angle = np.arccos(
        (target_eff_alt**2 - observer_eff_alt**2 - slant_range**2)
        / (2 * slant_range * observer_eff_alt)
    )
    return np.rad2deg(zenith_angle)


def compute_observer_azimuth(
    observer_lat: float, observer_lon: float, target_lat: float, target_lon: float
):
    """Compute azimuth of a ground station pointing at an aerial platform.

    Parameters
    ----------
    observer_lat : float
        Latitude of the observer [degrees].
    observer_lon : float
        Longitude of the observer [degrees].
    target_lat : float
        Latitude of the target [degrees].
    target_lon : float
        Longitude of the target [degrees].

    Returns
    -------
    azimuth : float
        Azimuth of the observer [degrees].
    """
    phi_obs = math.radians(observer_lat)
    phi_target = math.radians(target_lat)
    d_lam = math.radians(target_lon - observer_lon)

    numerator = math.sin(d_lam) * math.cos(phi_target)
    denominator = math.cos(phi_obs) * math.sin(phi_target) - math.sin(phi_obs) * math.cos(
        phi_target
    ) * math.cos(d_lam)

    azimuth = math.degrees(math.atan2(numerator, denominator))
    azimuth_normalized = azimuth % 360
    return azimuth_normalized
