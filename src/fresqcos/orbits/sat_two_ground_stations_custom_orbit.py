import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from astropy.time import Time
from astropy.coordinates import (
    CartesianRepresentation,
    EarthLocation,
    AltAz,
    GCRS,
    ITRS,
)

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

# --------------------------------------------------------
# 1. Custom orbit, ground stations, and minimum elevation
# --------------------------------------------------------

# Ground station 1 (example: Madrid)
gs1_lat = 40.4168 * u.deg
gs1_lon = -3.7038 * u.deg
gs1_alt = 667 * u.m
gs1 = EarthLocation(lat=gs1_lat, lon=gs1_lon, height=gs1_alt)

# Ground station 2 (example: custom long)
gs2_lat = 1.9838 * u.deg
gs2_lon = -15 * u.deg
gs2_alt = 200 * u.m
gs2 = EarthLocation(lat=gs2_lat, lon=gs2_lon, height=gs2_alt)

# Minimum elevation angle for visibility at both stations
min_elev = 10 * u.deg

# --------------------------------------------------------
# 2. Time grid
# --------------------------------------------------------

start_time = Time("2025-12-01 00:00:00", scale="utc")
end_time = Time("2025-12-02 00:00:00", scale="utc")
step_sec = 120.0  # seconds

step = step_sec * u.s
total_duration = (end_time - start_time).to(u.s)
n_steps = int(np.floor(total_duration / step)) + 1

times = start_time + np.arange(n_steps) * step

# --------------------------------------------------------
# 3. Define custom orbit (edit these elements)
# --------------------------------------------------------
# Example: Molniya-like orbit. Replace with your own parameters.

a = 26600 * u.km  # semi-major axis
ecc = 0.74 * u.one  # eccentricity
inc = 63.4 * u.deg  # inclination
raan = 0.0 * u.deg  # right ascension of ascending node
argp = 270.0 * u.deg  # argument of perigee
nu0 = 0.0 * u.deg  # true anomaly at epoch

orbit = Orbit.from_classical(
    Earth,
    a,
    ecc,
    inc,
    raan,
    argp,
    nu0,
    epoch=start_time,
)

# --------------------------------------------------------
# 4. Propagate orbit (GCRS → ITRS)
# --------------------------------------------------------

r_list = []
for t in times:
    dt = t - orbit.epoch
    orb_t = orbit.propagate(dt)
    r_list.append(orb_t.r.to(u.m))

r = u.Quantity(r_list)  # shape (N,3)

# Positions in GCRS
gcrs = GCRS(
    x=r[:, 0],
    y=r[:, 1],
    z=r[:, 2],
    representation_type=CartesianRepresentation,
    obstime=times,
)

# Transform GCRS → ITRS (Earth-fixed)
itrs = gcrs.transform_to(ITRS(obstime=times))
sat_ecef = itrs.cartesian  # x, y, z with units
sat_xyz_km = sat_ecef.xyz.to(u.km).value.T  # shape (N, 3)

# --------------------------------------------------------
# 5. Elevation & slant range wrt both ground stations
# --------------------------------------------------------

# AltAz for GS1
altaz1_frame = AltAz(obstime=times, location=gs1)
sat_altaz1 = itrs.transform_to(altaz1_frame)
elev1 = sat_altaz1.alt

# AltAz for GS2
altaz2_frame = AltAz(obstime=times, location=gs2)
sat_altaz2 = itrs.transform_to(altaz2_frame)
elev2 = sat_altaz2.alt

# Slant range: satellite–GS distance (in ITRS)
gs1_itrs = gs1.get_itrs(obstime=times).cartesian
gs2_itrs = gs2.get_itrs(obstime=times).cartesian

diff1 = sat_ecef - gs1_itrs
diff2 = sat_ecef - gs2_itrs

slant1 = diff1.norm()  # length quantity
slant2 = diff2.norm()

# Satellite altitude (geodetic height)
sat_geodetic = EarthLocation.from_geocentric(itrs.x, itrs.y, itrs.z)
sat_altitude = sat_geodetic.height  # quantity

# Ground station altitudes (constant, but make arrays for CSV)
gs1_altitude = np.repeat(gs1.height, len(times))
gs2_altitude = np.repeat(gs2.height, len(times))

# Visibility masks
visible1 = elev1 >= min_elev
visible2 = elev2 >= min_elev
visible_both = visible1 & visible2

# --------------------------------------------------------
# 6. Build and save CSV (only times visible from BOTH stations)
# --------------------------------------------------------

df = pd.DataFrame(
    {
        "utc_time": times.utc.iso,
        "elev1_deg": elev1.to(u.deg).value,
        "slant1_km": slant1.to(u.km).value,
        "elev2_deg": elev2.to(u.deg).value,
        "slant2_km": slant2.to(u.km).value,
        "sat_altitude_km": sat_altitude.to(u.km).value,
        "gs1_altitude_km": gs1_altitude.to(u.km).value,
        "gs2_altitude_km": gs2_altitude.to(u.km).value,
        "visible_gs1": visible1,
        "visible_gs2": visible2,
        "visible_both": visible_both,
    }
)

# Keep only simultaneous visibility
df_visible_both = df[df["visible_both"]].copy()
df_visible_both = df_visible_both[
    [
        "utc_time",
        "elev1_deg",
        "slant1_km",
        "elev2_deg",
        "slant2_km",
        "sat_altitude_km",
        "gs1_altitude_km",
        "gs2_altitude_km",
    ]
]

csv_filename = "custom_orbit_two_gs_simultaneous_visibility.csv"
df_visible_both.to_csv(csv_filename, index=False)

print(f"Saved simultaneous-visibility time series to: {csv_filename}")
print(f"Number of samples with visibility from both GS: {len(df_visible_both)}")

# --------------------------------------------------------
# 7. 3D visualization with both cones and animation
# --------------------------------------------------------

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection="3d")

# Earth sphere
R_earth = 6371.0  # km
u_sphere = np.linspace(0, 2 * np.pi, 50)
v_sphere = np.linspace(0, np.pi, 25)
x_earth = R_earth * np.outer(np.cos(u_sphere), np.sin(v_sphere))
y_earth = R_earth * np.outer(np.sin(u_sphere), np.sin(v_sphere))
z_earth = R_earth * np.outer(np.ones_like(u_sphere), np.cos(v_sphere))
ax.plot_surface(x_earth, y_earth, z_earth, rstride=1, cstride=1, alpha=0.3, edgecolor="none")

# Orbit path (ITRS)
ax.plot(
    sat_xyz_km[:, 0],
    sat_xyz_km[:, 1],
    sat_xyz_km[:, 2],
    linewidth=1,
    label="Orbit",
)

# Ground stations in ITRS (at start_time)
gs1_itrs0 = gs1.get_itrs(obstime=start_time).cartesian
gs2_itrs0 = gs2.get_itrs(obstime=start_time).cartesian
gs1_vec_km = gs1_itrs0.xyz.to(u.km).value
gs2_vec_km = gs2_itrs0.xyz.to(u.km).value

x_gs1, y_gs1, z_gs1 = gs1_vec_km
x_gs2, y_gs2, z_gs2 = gs2_vec_km

ax.scatter(x_gs1, y_gs1, z_gs1, s=60, label="GS1")
ax.scatter(x_gs2, y_gs2, z_gs2, s=60, label="GS2")


def add_visibility_cone(ax, gs_vec_km, min_elev, color=None, alpha=0.15):
    """
    Draw a visibility cone anchored at a ground station in ITRS.
    """
    half_angle_rad = np.deg2rad(90.0 - min_elev.to(u.deg).value)

    # Zenith unit vector at GS
    u_z = gs_vec_km / np.linalg.norm(gs_vec_km)

    # Two orthonormal vectors perpendicular to u_z
    ref = np.array([0.0, 0.0, 1.0])
    if np.allclose(np.cross(u_z, ref), 0):
        ref = np.array([0.0, 1.0, 0.0])

    u_x = np.cross(ref, u_z)
    u_x /= np.linalg.norm(u_x)
    u_y = np.cross(u_z, u_x)

    # Cone length (bigger than orbit radius)
    max_orbit_radius = np.max(np.linalg.norm(sat_xyz_km, axis=1))
    h_max = 1.2 * max_orbit_radius  # km

    h = np.linspace(0, h_max, 40)
    phi = np.linspace(0, 2 * np.pi, 50)
    H, PHI = np.meshgrid(h, phi)

    A = np.cos(half_angle_rad)
    Bc = np.sin(half_angle_rad) * np.cos(PHI)
    Bs = np.sin(half_angle_rad) * np.sin(PHI)

    x_gs, y_gs, z_gs = gs_vec_km

    X = x_gs + H * (A * u_z[0] + Bc * u_x[0] + Bs * u_y[0])
    Y = y_gs + H * (A * u_z[1] + Bc * u_x[1] + Bs * u_y[1])
    Z = z_gs + H * (A * u_z[2] + Bc * u_x[2] + Bs * u_y[2])

    ax.plot_surface(X, Y, Z, alpha=alpha, edgecolor="none")


# Add cones for both GS
add_visibility_cone(ax, gs1_vec_km, min_elev)
add_visibility_cone(ax, gs2_vec_km, min_elev)

# Animated satellite: point + trail (in ITRS)
(sat_point,) = ax.plot([], [], [], marker="o", markersize=6, linestyle="")
(trail_line,) = ax.plot([], [], [], linewidth=2, label="Satellite")

# Symmetric limits
max_orbit_radius = np.max(np.linalg.norm(sat_xyz_km, axis=1))
max_val = max(max_orbit_radius, R_earth * 1.1)
lim = 1.2 * max_val
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.set_aspect("equal")

ax.set_xlabel("X [km]")
ax.set_ylabel("Y [km]")
ax.set_zlabel("Z [km]")
ax.set_title(f"Custom-orbit satellite with two GS (min elev = {min_elev.to(u.deg).value:.1f}°)")
ax.legend()


def init():
    sat_point.set_data([], [])
    sat_point.set_3d_properties([])
    trail_line.set_data([], [])
    trail_line.set_3d_properties([])
    return sat_point, trail_line


def update(frame):
    x, y, z = sat_xyz_km[frame]

    # Set satellite position
    sat_point.set_data([x], [y])
    sat_point.set_3d_properties([z])

    # Color by visibility:
    #   red    -> visible from both
    #   orange -> visible from one
    #   blue   -> visible from none
    if visible_both[frame]:
        sat_point.set_color("red")
    elif visible1[frame] or visible2[frame]:
        sat_point.set_color("orange")
    else:
        sat_point.set_color("blue")

    # Trail
    trail_line.set_data(sat_xyz_km[: frame + 1, 0], sat_xyz_km[: frame + 1, 1])
    trail_line.set_3d_properties(sat_xyz_km[: frame + 1, 2])

    return sat_point, trail_line


ani = FuncAnimation(
    fig,
    update,
    frames=len(sat_xyz_km),
    init_func=init,
    interval=50,  # ms per frame
    blit=True,
)

plt.show()
