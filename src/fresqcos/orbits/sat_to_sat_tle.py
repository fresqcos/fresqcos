import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sgp4.api import Satrec
from astropy.time import Time
from astropy.coordinates import CartesianRepresentation
from astropy.coordinates.builtin_frames import TEME

# -------------------------------------------------------------
# 1. Define TLEs for the two satellites
# -------------------------------------------------------------

# Satellite 1 TLE
tle1_a = "1 25544U 98067A   23329.51782528  .00014351  00000+0  25818-3 0  9990"
tle2_a = "2 25544  51.6418 201.9003 0006931  30.1013  76.2109 15.49984292428327"

# Satellite 2 TLE (example; replace with your real one)
tle1_b = "1 43013U 17073A   23329.41496606  .00000088  00000+0  00000+0 0  9990"
tle2_b = "2 43013  97.5594 101.0294 0014671  86.5668 273.6819 14.95186308365736"

sat_a = Satrec.twoline2rv(tle1_a, tle2_a)
sat_b = Satrec.twoline2rv(tle1_b, tle2_b)

# Distance threshold
distance_limit_km = 1000.0  # km

# -------------------------------------------------------------
# 2. Define time span and step
# -------------------------------------------------------------

start_time = Time("2025-12-01 08:00:00", scale="utc")
end_time = Time("2025-12-01 10:0:00", scale="utc")
step_sec = 30  # time step in seconds

mjd_start = start_time.mjd
mjd_end = end_time.mjd
mjd_step = step_sec / 86400.0

times = Time(np.arange(mjd_start, mjd_end, mjd_step), format="mjd", scale="utc")

# -------------------------------------------------------------
# 3. Propagate both satellites with SGP4 (TEME frame)
# -------------------------------------------------------------

# sat_a
e_a, r_a, v_a = sat_a.sgp4_array(times.jd1, times.jd2)
# sat_b
e_b, r_b, v_b = sat_b.sgp4_array(times.jd1, times.jd2)

ok = (e_a == 0) & (e_b == 0)

if not np.any(ok):
    raise RuntimeError("No successful SGP4 points for both satellites in this interval.")

times_ok = times[ok]
r_a_ok = r_a[ok] * u.km  # shape (N, 3)
r_b_ok = r_b[ok] * u.km  # shape (N, 3)

# -------------------------------------------------------------
# 4. Compute separation distance
# -------------------------------------------------------------

teme_a = TEME(
    x=r_a_ok[:, 0],
    y=r_a_ok[:, 1],
    z=r_a_ok[:, 2],
    representation_type=CartesianRepresentation,
    obstime=times_ok,
)

teme_b = TEME(
    x=r_b_ok[:, 0],
    y=r_b_ok[:, 1],
    z=r_b_ok[:, 2],
    representation_type=CartesianRepresentation,
    obstime=times_ok,
)

diff = teme_a.cartesian - teme_b.cartesian
separation = diff.norm()  # Quantity in km

# -------------------------------------------------------------
# 5. Find times where distance < threshold
# -------------------------------------------------------------

close_mask = separation < (distance_limit_km * u.km)


def find_encounters(mask):
    encounters = []
    if not np.any(mask):
        return encounters

    m_int = mask.astype(int)
    changes = np.where(np.diff(m_int) != 0)[0]

    start_idx = None
    for idx in changes:
        if not mask[idx] and mask[idx + 1]:
            start_idx = idx + 1
        elif mask[idx] and not mask[idx + 1]:
            if start_idx is not None:
                encounters.append((start_idx, idx))
                start_idx = None

    if mask[-1] and start_idx is not None:
        encounters.append((start_idx, len(mask) - 1))

    return encounters


encounters = find_encounters(close_mask)

if not encounters:
    print(f"No encounters with separation < {distance_limit_km} km in this interval.")
else:
    print(f"Encounters with separation < {distance_limit_km} km:")
    for i, (i_start, i_end) in enumerate(encounters, start=1):
        t_start = times_ok[i_start]
        t_end = times_ok[i_end]

        sep_segment = separation[i_start : i_end + 1]
        min_sep = sep_segment.min()
        i_min = np.argmin(sep_segment)
        t_min = times_ok[i_start + i_min]

        print(f"\nEncounter {i}:")
        print(f"  Start: {t_start.iso}")
        print(f"  End:   {t_end.iso}")
        print(f"  Min separation: {min_sep.to(u.km).value:.3f} km at {t_min.iso}")

# -------------------------------------------------------------
# 6. Save time steps with separation < limit to CSV
# -------------------------------------------------------------

df = pd.DataFrame(
    {
        "utc_time": times_ok.utc.iso,
        "separation_km": separation.to(u.km).value,
    }
)

df = df[df["separation_km"] < distance_limit_km]

csv_filename = "tle_two_satellite_separation.csv"
df.to_csv(csv_filename, index=False)

print(f"\nSaved time series to: {csv_filename}")

# -------------------------------------------------------------
# 7. 3D animation of both satellites and their contact
# -------------------------------------------------------------

sat1_xyz = r_a_ok.value  # (N, 3)
sat2_xyz = r_b_ok.value  # (N, 3)
sep_km = separation.to(u.km).value  # (N,)

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection="3d")

# Earth sphere (TEME-centered)
R_earth = 6371.0  # km
u_sphere = np.linspace(0, 2 * np.pi, 50)
v_sphere = np.linspace(0, np.pi, 25)
x_earth = R_earth * np.outer(np.cos(u_sphere), np.sin(v_sphere))
y_earth = R_earth * np.outer(np.sin(u_sphere), np.sin(v_sphere))
z_earth = R_earth * np.outer(np.ones_like(u_sphere), np.cos(v_sphere))
ax.plot_surface(x_earth, y_earth, z_earth, rstride=1, cstride=1, alpha=0.3, edgecolor="none")

# Orbit paths
ax.plot(sat1_xyz[:, 0], sat1_xyz[:, 1], sat1_xyz[:, 2], linewidth=1, label="Sat 1 orbit")
ax.plot(sat2_xyz[:, 0], sat2_xyz[:, 1], sat2_xyz[:, 2], linewidth=1, label="Sat 2 orbit")

# Animated objects
(sat1_point,) = ax.plot(
    [], [], [], color="C4", marker="o", markersize=6, linestyle="", label="Sat 1"
)
(sat2_point,) = ax.plot(
    [], [], [], color="C5", marker="o", markersize=6, linestyle="", label="Sat 2"
)
(sat1_trail,) = ax.plot([], [], [], color="C6", linewidth=2)
(sat2_trail,) = ax.plot([], [], [], color="C7", linewidth=2)
(link_line,) = ax.plot([], [], [], color="C3", linewidth=2, linestyle="--", label="Link")

# Axis limits
max_radius1 = np.max(np.linalg.norm(sat1_xyz, axis=1))
max_radius2 = np.max(np.linalg.norm(sat2_xyz, axis=1))
max_val = max(max_radius1, max_radius2, R_earth * 1.1)
lim = 1.2 * max_val
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.set_aspect("equal")

ax.set_xlabel("X [km]")
ax.set_ylabel("Y [km]")
ax.set_zlabel("Z [km]")
ax.set_title(f"Two satellites – contact if separation < {distance_limit_km} km")
ax.legend()


def init():
    sat1_point.set_data([], [])
    sat1_point.set_3d_properties([])
    sat2_point.set_data([], [])
    sat2_point.set_3d_properties([])
    sat1_trail.set_data([], [])
    sat1_trail.set_3d_properties([])
    sat2_trail.set_data([], [])
    sat2_trail.set_3d_properties([])
    link_line.set_data([], [])
    link_line.set_3d_properties([])
    return sat1_point, sat2_point, sat1_trail, sat2_trail, link_line


def update(frame):
    x1, y1, z1 = sat1_xyz[frame]
    x2, y2, z2 = sat2_xyz[frame]

    # Satellite positions
    sat1_point.set_data([x1], [y1])
    sat1_point.set_3d_properties([z1])

    sat2_point.set_data([x2], [y2])
    sat2_point.set_3d_properties([z2])

    # Trails
    sat1_trail.set_data(sat1_xyz[: frame + 1, 0], sat1_xyz[: frame + 1, 1])
    sat1_trail.set_3d_properties(sat1_xyz[: frame + 1, 2])

    sat2_trail.set_data(sat2_xyz[: frame + 1, 0], sat2_xyz[: frame + 1, 1])
    sat2_trail.set_3d_properties(sat2_xyz[: frame + 1, 2])

    # Check contact condition
    in_contact = sep_km[frame] < distance_limit_km

    # Link only when in contact
    if in_contact:
        link_line.set_data([x1, x2], [y1, y2])
        link_line.set_3d_properties([z1, z2])
        color = "C3"
        # Color satellites depending on contact
        sat1_point.set_color(color)
        sat2_point.set_color(color)
    else:
        # Hide the link by setting empty data
        link_line.set_data([], [])
        link_line.set_3d_properties([])
        color1 = "C4"
        color2 = "C5"
        # Color satellites depending on contact
        sat1_point.set_color(color1)
        sat2_point.set_color(color2)

    return sat1_point, sat2_point, sat1_trail, sat2_trail, link_line


ani = FuncAnimation(
    fig,
    update,
    frames=len(times_ok),
    init_func=init,
    interval=50,  # ms per frame
    blit=False,
)

plt.show()

# To save the animation (requires ffmpeg installed):
# ani.save("two_satellites_contact.mp4", fps=20)
