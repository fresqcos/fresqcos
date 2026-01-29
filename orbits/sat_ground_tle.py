import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import ITRS
from matplotlib.animation import FuncAnimation

from sgp4.api import Satrec
from astropy.time import Time
from astropy.coordinates import (
    CartesianRepresentation,
    AltAz,
    EarthLocation,
)
from astropy.coordinates.builtin_frames import TEME, ITRS

# -------------------------------------------------------------------
# 1. Define TLE and ground station
# -------------------------------------------------------------------

# Example TLE (ISS) – replace with your satellite
tle1 = "1 41731U 16051A   25346.20108513  .00208160  00000-0  96011-3 0  9996"
tle2 = "2 41731  97.2963 272.1897 0007599  74.8788 285.3322 15.82964751520877"
sat = Satrec.twoline2rv(tle1, tle2)

# Ground station
gs_lat = 41.3874 * u.deg
gs_lon = 2.1686 * u.deg
gs_alt = 500 * u.m

ground_station = EarthLocation(lat=gs_lat, lon=gs_lon, height=gs_alt)

# Minimum elevation for considering a "visible" pass
min_elev = 10 * u.deg  # only passes above 30 degrees

# Time window for search (UTC)
start_time = Time("2025-12-01 00:00:00", scale="utc")
end_time = Time("2025-12-31 00:00:00", scale="utc")
step_sec = 30.0  # time step in seconds

# -------------------------------------------------------------------
# 2. Build time grid and propagate TLE with sgp4
# -------------------------------------------------------------------

mjd_start = start_time.mjd
mjd_end = end_time.mjd
mjd_step = step_sec / 86400.0

times = Time(np.arange(mjd_start, mjd_end, mjd_step), format="mjd", scale="utc")

# Propagate using sgp4; returns TEME position (km) and velocity (km/s)
e, r, v = sat.sgp4_array(times.jd1, times.jd2)
ok = e == 0

if not np.any(ok):
    raise RuntimeError("No successful sgp4 points; check TLE/time range")

times_ok = times[ok]
r_ok = r[ok] * u.km

# -------------------------------------------------------------------
# 3. Transform TEME coordinates to ITRS (ECEF)
# -------------------------------------------------------------------

teme = TEME(
    x=r_ok[:, 0],
    y=r_ok[:, 1],
    z=r_ok[:, 2],
    representation_type=CartesianRepresentation,
    obstime=times_ok,
)

itrs = teme.transform_to(ITRS(obstime=times_ok))

# Satellite ECEF coordinates
sat_ecef = itrs.cartesian

# Ground station ECEF coordinates at each time
gs_itrs = ground_station.get_itrs(obstime=times_ok).cartesian

# -------------------------------------------------------------------
# 4. Slant range (channel length)
# -------------------------------------------------------------------

diff = sat_ecef - gs_itrs
slant_range = diff.norm()  # quantity in meters

# -------------------------------------------------------------------
# 5. Topocentric AltAz for elevation/azimuth
# -------------------------------------------------------------------

altaz_frame = AltAz(obstime=times_ok, location=ground_station)
sat_altaz = itrs.transform_to(altaz_frame)

elev = sat_altaz.alt  # elevation angle
az = sat_altaz.az  # (not used here but available)

# -------------------------------------------------------------------
# 6. Satellite and ground station altitudes
# -------------------------------------------------------------------

# Satellite altitude above reference ellipsoid
sat_geodetic = EarthLocation.from_geocentric(itrs.x, itrs.y, itrs.z)
sat_altitude = sat_geodetic.height  # quantity, e.g. in meters

# Ground station altitude (same at all timesteps, but keep as array for csv)
# use np.size so this works even if times_ok is a scalar Time (returns 1)
gs_altitude = np.repeat(ground_station.height, np.size(times_ok))

# Boolean mask for visibility above min elevation (still used for passes)
visible = elev > min_elev

# -------------------------------------------------------------------
# 7. Pass detection
# -------------------------------------------------------------------


def find_passes(visible_mask):
    passes = []
    if not np.any(visible_mask):
        return passes

    vis_int = visible_mask.astype(int)
    changes = np.where(np.diff(vis_int) != 0)[0]

    start_idx = None
    for idx in changes:
        if not visible_mask[idx] and visible_mask[idx + 1]:
            start_idx = idx + 1
        elif visible_mask[idx] and not visible_mask[idx + 1]:
            if start_idx is not None:
                passes.append((start_idx, idx))
                start_idx = None

    if visible_mask[-1] and start_idx is not None:
        passes.append((start_idx, len(visible_mask) - 1))

    return passes


passes = find_passes(visible)

if not passes:
    print(f"No passes above {min_elev.to(u.deg).value:.1f} deg in this interval.")
else:
    print(f"Passes above {min_elev.to(u.deg).value:.1f} deg:")
    for i, (i_start, i_end) in enumerate(passes, start=1):
        t_start = times_ok[i_start]
        t_end = times_ok[i_end]

        elev_pass = elev[i_start : i_end + 1]
        range_pass = slant_range[i_start : i_end + 1]

        i_max_elev = np.argmax(elev_pass)
        i_min_range = np.argmin(range_pass)

        t_max_elev = times_ok[i_start + i_max_elev]
        t_min_range = times_ok[i_start + i_min_range]

        print(f"\nPass {i}:")
        print(f"  AOS (rise above {min_elev.to(u.deg):.1f}): {t_start.iso}")
        print(f"  LOS (fall below {min_elev.to(u.deg):.1f}): {t_end.iso}")
        print(f"  Max elevation: {elev_pass.max().to(u.deg):.2f} at {t_max_elev.iso}")
        print(f"  Min slant range: {range_pass.min().to(u.km):.2f} at {t_min_range.iso}")

# -------------------------------------------------------------------
# 8. Save ALL time steps to CSV
# -------------------------------------------------------------------

# Build a DataFrame with one row per time step
df = pd.DataFrame(
    {
        "utc_time": times_ok.utc.iso,  # string timestamps
        "elevation_deg": elev.to(u.deg).value,  # elevation angle
        "slant_range_km": slant_range.to(u.km).value,  # distance satellite–GS
        "sat_altitude_km": sat_altitude.to(u.km).value,  # satellite altitude
        "gs_altitude_km": gs_altitude.to(u.km).value,  # ground station altitude
    }
)

# If you want ONLY samples where elevation is above min_elev, uncomment:
df = df[df["elevation_deg"] > min_elev.to(u.deg).value]

csv_filename = "satellite_geometry_timeseries.csv"
df.to_csv(csv_filename, index=False)

print(f"\nSaved time series to: {csv_filename}")

# --------------------------------------------------------
# 9. Compute elevation mask (for coloring in animation)
# --------------------------------------------------------

altaz_frame = AltAz(obstime=times_ok, location=ground_station)
sat_altaz = itrs.transform_to(altaz_frame)
elev = sat_altaz.alt
visible = elev >= min_elev  # boolean mask

# --------------------------------------------------------
# 10. Set up 3D figure (all in ITRS coordinates)
# --------------------------------------------------------

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

# Earth sphere (centered at ITRS origin)
R_earth = 6371.0  # km
u_sphere = np.linspace(0, 2 * np.pi, 50)
v_sphere = np.linspace(0, np.pi, 25)
x_earth = R_earth * np.outer(np.cos(u_sphere), np.sin(v_sphere))
y_earth = R_earth * np.outer(np.sin(u_sphere), np.sin(v_sphere))
z_earth = R_earth * np.outer(np.ones_like(u_sphere), np.cos(v_sphere))
ax.plot_surface(x_earth, y_earth, z_earth, rstride=1, cstride=1, alpha=0.3, edgecolor="none")

sat_ecef = itrs.cartesian  # x, y, z with units (meters by default)
sat_xyz_km = sat_ecef.xyz.to(u.km).value.T  # shape (N, 3)

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
# 11. Visibility cone (ITRS, anchored at GS)
# --------------------------------------------------------

# Half-angle of cone (angle from zenith)
half_angle_rad = np.deg2rad(90.0 - min_elev.to(u.deg).value)

# Zenith unit vector at GS = normalized GS position in ITRS
u_z = gs_vec_km / np.linalg.norm(gs_vec_km)

# Build two orthonormal vectors perpendicular to u_z
ref = np.array([0.0, 0.0, 1.0])
if np.allclose(np.cross(u_z, ref), 0):
    ref = np.array([0.0, 1.0, 0.0])

u_x = np.cross(ref, u_z)
u_x /= np.linalg.norm(u_x)
u_y = np.cross(u_z, u_x)

# Cone length (just for visualization; make it slightly larger than orbit radius)
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
# 12. Animated satellite: point + trail (in ITRS)
# --------------------------------------------------------

(sat_point,) = ax.plot([], [], [], marker="o", markersize=6, linestyle="")
(trail_line,) = ax.plot([], [], [], linewidth=2, label="Satellite")

# Set symmetric limits
max_val = max(max_orbit_radius, R_earth * 1.1)
lim = 1.2 * max_val
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.set_aspect("equal")

ax.set_xlabel("X [km]")
ax.set_ylabel("Y [km]")
ax.set_zlabel("Z [km]")
ax.set_title(f"Orbit & visibility cone (min elev = {min_elev.to(u.deg).value:.1f}°)")
ax.legend()


def init():
    sat_point.set_data([], [])
    sat_point.set_3d_properties([])
    trail_line.set_data([], [])
    trail_line.set_3d_properties([])
    return sat_point, trail_line


def update(frame):
    x, y, z = sat_xyz_km[frame]

    # Satellite point
    sat_point.set_data([x], [y])
    sat_point.set_3d_properties([z])

    # Color the satellite based on visibility
    if visible[frame]:
        sat_point.set_color("red")
    else:
        sat_point.set_color("blue")

    # Trail from start up to current frame
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
