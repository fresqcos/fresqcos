import numpy as np
import math

EARTH_RADIUS_KM = 6371


def channel_length_from_zenith_angle(ground_station_alt, aerial_platform_alt, zenith_angle):
    """Compute channel length that corresponds to a particular ground station altitude, aerial
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
    length : float
        Length of the channel [km].
    """
    zenith_angle = np.deg2rad(zenith_angle)
    aerial_platform_eff_alt = EARTH_RADIUS_KM + aerial_platform_alt
    ground_station_eff_elt = EARTH_RADIUS_KM + ground_station_alt
    length = np.sqrt(
        aerial_platform_eff_alt**2 + ground_station_eff_elt**2 * (np.cos(zenith_angle) ** 2 - 1)
    ) - ground_station_eff_elt * np.cos(zenith_angle)

    return length


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
    """Compute central angle between two points on the Earth surface given their latitudes and longitudes.

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


def channel_length_from_central_angle(ground_station_alt, aerial_platform_alt, central_angle):
    """Compute channel length that corresponds to a particular ground station altitude, aerial
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
    length : float
        Length of the channel [km].
    """
    ground_station_eff_elt = EARTH_RADIUS_KM + ground_station_alt
    aerial_platform_eff_alt = EARTH_RADIUS_KM + aerial_platform_alt
    channel_length = np.sqrt(
        ground_station_eff_elt**2
        + aerial_platform_eff_alt**2
        - 2 * ground_station_eff_elt * aerial_platform_eff_alt * np.cos(np.deg2rad(central_angle))
    )
    return channel_length


def zenith_angle(channel_length, ground_station_alt, aerial_platform_alt):
    """Compute zenith angle of the channel between a ground station and an aerial platform.

    Parameters
    ----------
    channel_length : float
        Length of the channel [km].
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
    ground_station_eff_elt = EARTH_RADIUS_KM + ground_station_alt
    zenith_angle = np.arccos(
        (aerial_platform_eff_alt**2 - ground_station_eff_elt**2 - channel_length**2)
        / (2 * channel_length * ground_station_eff_elt)
    )
    return np.rad2deg(zenith_angle)


def ground_station_azimuth(lat_gs, lon_gs, lat_aerial, lon_aerial):
    """Compute azimuth of a ground station pointing at an aerial platform.

    Parameters
    ----------
    lat_gs : float
        Latitude of the ground station [degrees].
    lon_gs : float
        Longitude of the ground station [degrees].
    lat_aerial : float
        Latitude of the aerial platform [degrees].
    lon_aerial : float
        Longitude of the aerial platform [degrees].

    Returns
    -------
    az : float
        Azimuth of the channel [degrees].
    """
    phi_gs = math.radians(lat_gs)
    phi_sat = math.radians(lat_aerial)
    d_lam = math.radians(lon_aerial - lon_gs)

    numerator = math.sin(d_lam) * math.cos(phi_sat)
    denominator = math.cos(phi_gs) * math.sin(phi_sat) - math.sin(phi_gs) * math.cos(
        phi_sat
    ) * math.cos(d_lam)

    az = math.degrees(math.atan2(numerator, denominator))
    az_normalized = az % 360
    return az_normalized
