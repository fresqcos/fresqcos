# pip install astropy poliastro numpy matplotlib pandas

import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from astropy.time import Time
from astropy.coordinates import CartesianRepresentation
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

# -------------------------------------------------------------
# 1. Define custom orbits for the two satellites
# -------------------------------------------------------------
# Edit these orbital elements to match your case

# Satellite 1: example Molniya-like
a1 = 26600 * u.km  # semi-major axis
ecc1 = 0.74 * u.one  # eccentricity
inc1 = 63.4 * u.deg  # inclination
raan1 = 0.0 * u.deg  # RAAN
argp1 = 270.0 * u.deg  # argument of perigee
nu1 = 0.0 * u.deg  # true anomaly at epoch

start_time = Time("2025-11-15 08:00:00", scale="utc")

sat1 = Orbit.from_classical(
    Earth,
    a1,
    ecc1,
    inc1,
    raan1,
    argp1,
    nu1,
    epoch=start_time,
)

# Satellite 2: example near-circular LEO
a2 = Earth.R + 700 * u.km  # ~700 km circular
ecc2 = 0.001 * u.one
inc2 = 98.0 * u.deg
raan2 = 20.0 * u.deg
argp2 = 0.0 * u.deg
nu2 = 0.0 * u.deg

sat2 = Orbit.from_classical(
    Earth,
    a2,
    ecc2,
    inc2,
    raan2,
    argp2,
    nu2,
    epoch=start_time,
)

# Distance threshold
distance_limit_km = 1000.0  # km

# -------------------------------------------------------------
# 2. Define time span and step
# -------------------------------------------------------------

end_time = Time("2025-12-01 10:00:00", scale="utc")
step_sec = 240.0  # time step in seconds

step = step_sec * u.s
total_duration = (end_time - start_time).to(u.s)
n_steps = int(np.floor(total_duration / step)) + 1

times = start_time + np.arange(n_steps) * step

# -------------------------------------------------------------
# 3. Propagate both satellites (inertial frame, Earth-centered)
# -------------------------------------------------------------

r1_list = []
r2_list = []

for t in times:
    dt1 = t - sat1.epoch
    dt2 = t - sat2.epoch
    sat1_t = sat1.propagate(dt1)
    sat2_t = sat2.propagate(dt2)
    r1_list.append(sat1_t.r.to(u.km))  # (3,) position vector
    r2_list.append(sat2_t.r.to(u.km))

r1 = u.Quantity(r1_list)  # shape (N, 3)
r2 = u.Quantity(r2_list)  # shape (N, 3)

# -------------------------------------------------------------
# 4. Compute separation distance
# -------------------------------------------------------------

diff = r1 - r2
separation = np.linalg.norm(diff.to(u.km).value, axis=1) * u.km  # (N,)

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
        t_start = times[i_start]
        t_end = times[i_end]

        sep_segment = separation[i_start : i_end + 1]
        min_sep = sep_segment.min()
        i_min = np.argmin(sep_segment)
        t_min = times[i_start + i_min]

        print(f"\nEncounter {i}:")
        print(f"  Start: {t_start.iso}")
        print(f"  End:   {t_end.iso}")
        print(f"  Min separation: {min_sep.to(u.km).value:.3f} km at {t_min.iso}")

# -------------------------------------------------------------
# 6. Save time steps with separation < limit to CSV
# -------------------------------------------------------------

df = pd.DataFrame(
    {
        "utc_time": times.utc.iso,
        "separation_km": separation.to(u.km).value,
    }
)

df = df[df["separation_km"] < distance_limit_km]

csv_filename = "two_custom_satellite_separation.csv"
df.to_csv(csv_filename, index=False)

print(f"\nSaved time series to: {csv_filename}")

# -------------------------------------------------------------
# 7. 3D animation of both satellites and their contact
# -------------------------------------------------------------

sat1_xyz = r1.to(u.km).value  # (N, 3)
sat2_xyz = r2.to(u.km).value  # (N, 3)
sep_km = separation.to(u.km).value  # (N,)

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection="3d")

# Earth sphere (geocentric) – FIXED FORMULA
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

# Equal aspect in 3D (this is what keeps the sphere looking like a sphere)
ax.set_box_aspect([1, 1, 1])

ax.set_xlabel("X [km]")
ax.set_ylabel("Y [km]")
ax.set_zlabel("Z [km]")
ax.set_title(f"Two custom-orbit satellites – contact if separation < {distance_limit_km} km")
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
        sat1_point.set_color(color)
        sat2_point.set_color(color)
    else:
        link_line.set_data([], [])
        link_line.set_3d_properties([])
        sat1_point.set_color("C4")
        sat2_point.set_color("C5")

    return sat1_point, sat2_point, sat1_trail, sat2_trail, link_line


ani = FuncAnimation(
    fig,
    update,
    frames=len(times),
    init_func=init,
    interval=50,  # ms per frame
    blit=False,
)

plt.show()

# To save the animation (requires ffmpeg installed):
# ani.save("two_custom_satellites_contact.mp4", fps=20)
