"""Module for geometric calculations related to the channel between a ground station and an aerial platform."""

import numpy as np
import math

EARTH_RADIUS_KM = 6371


def slant_range_from_zenith_angle(ground_station_alt, aerial_platform_alt, zenith_angle):
    """Compute slant range that corresponds to a particular ground station altitude, aerial
    platform altitude and zenith angle.

    Parameters
    ----------
    ground_station_alt : float
        Altitude of the ground station [km].
    aerial_platform_alt : float
        Altitude of the aerial platform [km].
    zenith_angle : float
        Zenith angle of aerial platform [degrees].

    Returns
    -------
    channel_length : float
        Slant range of the channel [km].
    """
    zenith_angle = np.deg2rad(zenith_angle)
    aerial_platform_eff_alt = EARTH_RADIUS_KM + aerial_platform_alt
    ground_station_eff_alt = EARTH_RADIUS_KM + ground_station_alt
    channel_length = np.sqrt(
        aerial_platform_eff_alt**2 + ground_station_eff_alt**2 * (np.cos(zenith_angle) ** 2 - 1)
    ) - ground_station_eff_alt * np.cos(zenith_angle)

    return channel_length


def height_min_horiz(length, height):
    """Compute minimal height of a horizontal channel between two aerial platforms at the same height.

    Parameters
    ----------
    length : float
        Length of the horizontal channel [km].
    height : float
        Height of the aerial platforms [km].

    Returns
    -------
    h_min : float
        Minimal height of the channel [km].
    """
    aerial_platform_eff_alt = EARTH_RADIUS_KM + height
    theta = np.arcsin((length) / (2 * aerial_platform_eff_alt))
    h_min = np.cos(theta) * aerial_platform_eff_alt - EARTH_RADIUS_KM
    return h_min


def sec(theta):
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


def central_angle(latitude_1, longitude_1, latitude_2, longitude_2):
    """Compute central angle between two points on the Earth's surface given their latitudes and longitudes.

    Parameters
    ----------
    latitude_1 : float
        Latitude of the first point [degrees].
    longitude_1 : float
        Longitude of the first point [degrees].
    latitude_2 : float
        Latitude of the second point [degrees].
    longitude_2 : float
        Longitude of the second point [degrees].

    Returns
    -------
    angle : float
        Central angle between the two points [degrees].
    """
    phi_1 = np.deg2rad(latitude_1)
    phi_2 = np.deg2rad(latitude_2)
    Delta_lambda = np.deg2rad(longitude_2 - longitude_1)

    angle = np.arccos(
        np.sin(phi_1) * np.sin(phi_2) + np.cos(phi_1) * np.cos(phi_2) * np.cos(Delta_lambda)
    )
    return np.rad2deg(angle)


def slant_range_from_central_angle(ground_station_alt, aerial_platform_alt, central_angle):
    """Compute slant range that corresponds to a particular ground station altitude, aerial
    platform altitude and central angle.

    Parameters
    ----------
    ground_station_alt : float
        Altitude of the ground station [km].
    aerial_platform_alt : float
        Altitude of the aerial platform [km].
    central_angle : float
        Central angle between the ground station and the aerial platform [degrees].

    Returns
    -------
    channel_length : float
        Slant range of the channel [km].
    """
    ground_station_eff_alt = EARTH_RADIUS_KM + ground_station_alt
    aerial_platform_eff_alt = EARTH_RADIUS_KM + aerial_platform_alt
    channel_length = np.sqrt(
        ground_station_eff_alt**2
        + aerial_platform_eff_alt**2
        - 2 * ground_station_eff_alt * aerial_platform_eff_alt * np.cos(np.deg2rad(central_angle))
    )
    return channel_length


def slant_range_from_coordinates(
    ground_station_lat,
    ground_station_lon,
    ground_station_alt,
    aerial_platform_lat,
    aerial_platform_lon,
    aerial_platform_alt,
):
    """Compute slant range between a ground station and an aerial platform given their coordinates.
    Parameters
    ----------
    ground_station_lat : float
        Latitude of the ground station [degrees].
    ground_station_lon : float
        Longitude of the ground station [degrees].
    ground_station_alt : float
        Altitude of the ground station [km].
    aerial_platform_lat : float
        Latitude of the aerial platform [degrees].
    aerial_platform_lon : float
        Longitude of the aerial platform [degrees].
    aerial_platform_alt : float
        Altitude of the aerial platform [km].

    Returns
    -------
    channel_length : float
        Slant range of the channel [km].
    """
    central_angle = central_angle(
        ground_station_lat, ground_station_lon, aerial_platform_lat, aerial_platform_lon
    )

    ground_station_eff_alt = EARTH_RADIUS_KM + ground_station_alt
    aerial_platform_eff_alt = EARTH_RADIUS_KM + aerial_platform_alt

    channel_length = math.sqrt(
        ground_station_eff_alt**2
        + aerial_platform_eff_alt**2
        - 2 * ground_station_eff_alt * aerial_platform_eff_alt * np.cos(np.deg2rad(central_angle))
    )
    return channel_length


def zenith_angle(slant_range, ground_station_alt, aerial_platform_alt):
    """Compute zenith angle of the channel between a ground station and an aerial platform.

    Parameters
    ----------
    slant_range : float
        Slant range of the channel [km].
    ground_station_alt : float
        Altitude of the ground station [km].
    aerial_platform_alt : float
        Altitude of the aerial platform [km].

    Returns
    -------
    zenith_angle : float
        Zenith angle of the channel [degrees].
    """
    aerial_platform_eff_alt = EARTH_RADIUS_KM + aerial_platform_alt
    ground_station_eff_alt = EARTH_RADIUS_KM + ground_station_alt
    zenith_angle = np.arccos(
        (aerial_platform_eff_alt**2 - ground_station_eff_alt**2 - slant_range**2)
        / (2 * slant_range * ground_station_eff_alt)
    )
    return np.rad2deg(zenith_angle)


def ground_station_azimuth(
    ground_station_lat, ground_station_lon, aerial_platform_lat, aerial_platform_lon
):
    """Compute azimuth of a ground station pointing at an aerial platform.

    Parameters
    ----------
    ground_station_lat : float
        Latitude of the ground station [degrees].
    ground_station_lon : float
        Longitude of the ground station [degrees].
    aerial_platform_lat : float
        Latitude of the aerial platform [degrees].
    aerial_platform_lon : float
        Longitude of the aerial platform [degrees].

    Returns
    -------
    azimuth : float
        Azimuth of the channel [degrees].
    """
    phi_gs = math.radians(ground_station_lat)
    phi_sat = math.radians(aerial_platform_lat)
    d_lam = math.radians(aerial_platform_lon - ground_station_lon)

    numerator = math.sin(d_lam) * math.cos(phi_sat)
    denominator = math.cos(phi_gs) * math.sin(phi_sat) - math.sin(phi_gs) * math.cos(
        phi_sat
    ) * math.cos(d_lam)

    azimuth = math.degrees(math.atan2(numerator, denominator))
    azimuth_normalized = azimuth % 360
    return azimuth_normalized
