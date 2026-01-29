import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from astropy.time import Time
from astropy.coordinates import (
    EarthLocation,
    GCRS,
    ITRS,
    AltAz,
    CartesianRepresentation,
)

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

# -------------------------------------------------------------------
# 1. Define ground station
# -------------------------------------------------------------------

# Example ground station: Madrid, Spain
gs_lat = 40.4168 * u.deg  # latitude
gs_lon = -3.7038 * u.deg  # longitude
gs_alt = 667 * u.m  # height above sea level

ground_station = EarthLocation(lat=gs_lat, lon=gs_lon, height=gs_alt)

# Minimum elevation threshold for considering a pass "visible"
min_elev = 10 * u.deg

# Time window for simulation (UTC)
start_time = Time("2025-12-01 00:00:00", scale="utc")
end_time = Time("2025-12-02 00:00:00", scale="utc")  # 1 day later
step = 120.0 * u.s  # propagation step

# -------------------------------------------------------------------
# 2. Define the orbit from classical orbital elements
# -------------------------------------------------------------------
# You can change these parameters for Molniya, sun-sync, etc.

# EXAMPLE A: Molniya-like orbit
a = 26600 * u.km  # semi-major axis
ecc = 0.74 * u.one  # eccentricity
inc = 63.4 * u.deg  # inclination (critical Molniya inclination)
raan = 0.0 * u.deg  # right ascension of ascending node
argp = 270.0 * u.deg  # argument of perigee
nu0 = 0.0 * u.deg  # true anomaly at epoch

# --- OR ---

# EXAMPLE B: Sun-synchronous-like orbit (~600 km altitude)
# a   = (Earth.R + 600 * u.km)  # semi-major axis
# ecc = 0.0 * u.one
# inc = 97.8 * u.deg            # typical SSO inclination
# raan = 0.0 * u.deg
# argp = 0.0 * u.deg
# nu0  = 0.0 * u.deg

# Create orbit object at epoch = start_time
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

# -------------------------------------------------------------------
# 3. Build time grid and propagate the orbit
# -------------------------------------------------------------------

total_duration = (end_time - start_time).to(u.s)
n_steps = int(np.floor(total_duration / step)) + 1

# Time array
times = start_time + np.arange(n_steps) * step

# Propagate and collect position vectors in GCRS (Earth-centered inertial)
r_list = []
for t in times:
    dt = t - orbit.epoch
    orb_t = orbit.propagate(dt)
    r_list.append(orb_t.r.to(u.m))  # position vector in meters

r = u.Quantity(r_list)  # shape (N, 3)

# -------------------------------------------------------------------
# 4. Convert from inertial (GCRS) to Earth-fixed (ITRS)
# -------------------------------------------------------------------

gcrs = GCRS(
    x=r[:, 0],
    y=r[:, 1],
    z=r[:, 2],
    representation_type=CartesianRepresentation,
    obstime=times,
)

itrs = gcrs.transform_to(ITRS(obstime=times))

sat_ecef = itrs.cartesian  # satellite ECEF coordinates

# Ground station ECEF coordinates at each time
gs_itrs = ground_station.get_itrs(obstime=times).cartesian

# -------------------------------------------------------------------
# 5. Slant range (satellite–ground station distance)
# -------------------------------------------------------------------

diff = sat_ecef - gs_itrs
slant_range = diff.norm()  # Quantity in meters

# -------------------------------------------------------------------
# 6. Topocentric AltAz: elevation, azimuth
# -------------------------------------------------------------------

altaz_frame = AltAz(obstime=times, location=ground_station)
sat_altaz = itrs.transform_to(altaz_frame)

elev = sat_altaz.alt  # elevation
az = sat_altaz.az  # azimuth (not used but available)

visible = elev > min_elev  # bool mask for elevation above threshold

# -------------------------------------------------------------------
# 7. Satellite and ground station altitudes
# -------------------------------------------------------------------

# Satellite geodetic altitude above reference ellipsoid
sat_geodetic = EarthLocation.from_geocentric(itrs.x, itrs.y, itrs.z)
sat_altitude = sat_geodetic.height  # e.g. meters

# Ground station altitude is constant, but make an array for convenience
gs_altitude = np.repeat(ground_station.height, len(times))

# -------------------------------------------------------------------
# 8. Find individual passes above min elevation
# -------------------------------------------------------------------


def find_passes(visible_mask):
    """
    Breaks a boolean 'visible' array into a list of (start_idx, end_idx) passes
    where visible is True between start_idx and end_idx (inclusive).
    """
    passes = []
    if not np.any(visible_mask):
        return passes

    vis_int = visible_mask.astype(int)
    changes = np.where(np.diff(vis_int) != 0)[0]

    start_idx = None
    for idx in changes:
        if not visible_mask[idx] and visible_mask[idx + 1]:
            # False -> True : pass starts
            start_idx = idx + 1
        elif visible_mask[idx] and not visible_mask[idx + 1]:
            # True -> False : pass ends
            if start_idx is not None:
                passes.append((start_idx, idx))
                start_idx = None

    # If still visible at the end, close last pass
    if visible_mask[-1] and start_idx is not None:
        passes.append((start_idx, len(visible_mask) - 1))

    return passes


passes = find_passes(visible)

if not passes:
    print(f"No passes above {min_elev.to(u.deg).value:.1f} deg in this interval.")
else:
    print(f"Passes above {min_elev.to(u.deg).value:.1f} deg:")
    for i, (i_start, i_end) in enumerate(passes, start=1):
        t_start = times[i_start]
        t_end = times[i_end]

        elev_pass = elev[i_start : i_end + 1]
        range_pass = slant_range[i_start : i_end + 1]

        i_max_elev = np.argmax(elev_pass)
        i_min_range = np.argmin(range_pass)

        t_max_elev = times[i_start + i_max_elev]
        t_min_range = times[i_start + i_min_range]

        print(f"\nPass {i}:")
        print(f"  AOS (rise above {min_elev.to(u.deg):.1f}): {t_start.iso}")
        print(f"  LOS (fall below {min_elev.to(u.deg):.1f}): {t_end.iso}")
        print(f"  Max elevation: {elev_pass.max().to(u.deg):.2f} at {t_max_elev.iso}")
        print(f"  Min slant range: {range_pass.min().to(u.km):.2f} at {t_min_range.iso}")

# -------------------------------------------------------------------
# 9. Save ALL time steps to CSV
# -------------------------------------------------------------------

df = pd.DataFrame(
    {
        "utc_time": times.utc.iso,  # timestamps
        "elevation_deg": elev.to(u.deg).value,  # elevation
        "slant_range_km": slant_range.to(u.km).value,  # slant range
        "sat_altitude_km": sat_altitude.to(u.km).value,  # satellite altitude
        "gs_altitude_km": gs_altitude.to(u.km).value,  # ground station altitude
    }
)

# If you want ONLY samples where elevation is above min_elev, uncomment:
df = df[df["elevation_deg"] > min_elev.to(u.deg).value]

csv_filename = "custom_orbit_geometry_timeseries.csv"
df.to_csv(csv_filename, index=False)

print(f"\nSaved time series to: {csv_filename}")

# Transform GCRS → ITRS for Earth-fixed coordinates (same frame as GS)
itrs = gcrs.transform_to(ITRS(obstime=times))
sat_ecef = itrs.cartesian
sat_xyz_km = sat_ecef.xyz.to(u.km).value.T  # shape (N, 3)

# --------------------------------------------------------
# 5. Elevation vs ground station (for visibility mask)
# --------------------------------------------------------

altaz_frame = AltAz(obstime=times, location=ground_station)
sat_altaz = itrs.transform_to(altaz_frame)
elev = sat_altaz.alt
visible = elev >= min_elev  # True when sat is above min elevation

# --------------------------------------------------------
# 6. 3D figure setup (everything in ITRS)
# --------------------------------------------------------

fig = plt.figure(figsize=(8, 8))
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
    label="Orbit (ITRS)",
)

# Ground station in ITRS (at start_time)
gs_itrs0 = ground_station.get_itrs(obstime=start_time).cartesian
gs_vec_km = gs_itrs0.xyz.to(u.km).value  # (3,)
x_gs, y_gs, z_gs = gs_vec_km
ax.scatter(x_gs, y_gs, z_gs, s=50, label="Ground station")


# --------------------------------------------------------
# 7. Visibility cone in ITRS anchored at GS
# --------------------------------------------------------

# Half-angle of cone (from zenith)
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

# Cone length (slightly larger than orbit radius)
max_orbit_radius = np.max(np.linalg.norm(sat_xyz_km, axis=1))
h_max = 1.2 * max_orbit_radius  # km

h = np.linspace(0, h_max, 50)
phi = np.linspace(0, 2 * np.pi, 60)
H, PHI = np.meshgrid(h, phi)

A = np.cos(half_angle_rad)
Bc = np.sin(half_angle_rad) * np.cos(PHI)
Bs = np.sin(half_angle_rad) * np.sin(PHI)

X_cone = x_gs + H * (A * u_z[0] + Bc * u_x[0] + Bs * u_y[0])
Y_cone = y_gs + H * (A * u_z[1] + Bc * u_x[1] + Bs * u_y[1])
Z_cone = z_gs + H * (A * u_z[2] + Bc * u_x[2] + Bs * u_y[2])

ax.plot_surface(X_cone, Y_cone, Z_cone, alpha=0.15, edgecolor="none")

# --------------------------------------------------------
# 8. Animated satellite: point + trail (in ITRS)
# --------------------------------------------------------

(sat_point,) = ax.plot([], [], [], marker="o", markersize=6, linestyle="")
(trail_line,) = ax.plot([], [], [], linewidth=2, label="Satellite")

# Symmetric limits
max_val = max(max_orbit_radius, R_earth * 1.1)
lim = 1.2 * max_val
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.set_aspect("equal")

ax.set_xlabel("X [km]")
ax.set_ylabel("Y [km]")
ax.set_zlabel("Z [km]")
ax.set_title(f"Custom orbit & visibility cone (min elev = {min_elev.to(u.deg).value:.1f}°)")
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

    # Color by visibility
    if visible[frame]:
        sat_point.set_color("red")
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
