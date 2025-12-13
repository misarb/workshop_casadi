#!/usr/bin/env python3
"""
Run Example 4: EV Minimum-Time Trajectory Optimization
Generates images for workshop presentation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from casadi import *
import os
import json

os.makedirs('../public/slides/intro-optimization-mpc-workshop', exist_ok=True)

# BelkX colors
BELKX_BLUE = '#0079C1'
BELKX_GRAY = '#6B7280'
BELKX_ORANGE = '#F59E0B'
BELKX_GREEN = '#10B981'
BELKX_RED = '#EF4444'

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16, 'lines.linewidth': 2.5})

print("=" * 60)
print("Example 4: EV Minimum-Time Trajectory Optimization")
print("=" * 60)

# ===== CREATE SIMPLE TRACK =====
print("\nCreating race track...")

# Create a simple oval track
theta = np.linspace(0, 2*np.pi, 100)
# Oval shape
centerline_x = 100 * np.cos(theta)
centerline_y = 50 * np.sin(theta)
track_width = 6.0  # meters

print(f"  Track: {len(centerline_x)} points, width: {track_width}m")

# ===== VISUALIZE TRACK =====
print("\nGenerating track layout...")
fig, ax = plt.subplots(figsize=(12, 8))

# Draw track boundaries
inner_x = centerline_x - track_width/2 * np.sin(theta)
inner_y = centerline_y + track_width/2 * np.cos(theta)
outer_x = centerline_x + track_width/2 * np.sin(theta)
outer_y = centerline_y - track_width/2 * np.cos(theta)

ax.fill(outer_x, outer_y, color=BELKX_GRAY, alpha=0.2, label='Track')
ax.fill(inner_x, inner_y, color='white')

# Draw centerline
ax.plot(centerline_x, centerline_y, 'k--', linewidth=2, alpha=0.5, label='Centerline')

# Mark start/finish
ax.plot(centerline_x[0], centerline_y[0], 'go', markersize=15, label='Start', zorder=5)
ax.plot(centerline_x[-1], centerline_y[-1], 'ro', markersize=15, label='Finish', zorder=5)

ax.set_xlabel('X Position (m)', fontsize=14)
ax.set_ylabel('Y Position (m)', fontsize=14)
ax.set_title('Race Track Layout (Oval Circuit)', fontsize=16, pad=15)
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.set_xlim(-140, 140)
ax.set_ylim(-80, 80)

plt.tight_layout()
plt.savefig('../public/slides/intro-optimization-mpc-workshop/track-layout.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Saved: track-layout.png")

# ===== SIMPLIFIED TRAJECTORY OPTIMIZATION =====
print("\nSetting up trajectory optimization...")

# Simplified problem: optimize path around track
N = 50  # Number of waypoints
dt_var = 0.5  # Fixed time step for simplicity

opti = Opti()

# State variables (simplified point mass)
x = opti.variable(N+1)  # X position
y = opti.variable(N+1)  # Y position
v = opti.variable(N+1)  # Velocity

# Control variables
a = opti.variable(N)    # Acceleration

# Objective: minimize time (or maximize average velocity)
avg_velocity = sum1(v) / (N+1)
opti.minimize(-avg_velocity)  # Maximize average velocity

# Dynamics (simplified point mass)
opti.subject_to(x[0] == centerline_x[0])
opti.subject_to(y[0] == centerline_y[0])
opti.subject_to(v[0] == 15.0)  # Start at 15 m/s

for i in range(N):
    # Position update (assume straight line segments)
    direction_x = (centerline_x[min((i+1)*2, len(centerline_x)-1)] - centerline_x[i*2])
    direction_y = (centerline_y[min((i+1)*2, len(centerline_y)-1)] - centerline_y[i*2])
    norm = np.sqrt(direction_x**2 + direction_y**2) + 1e-6

    opti.subject_to(x[i+1] == x[i] + v[i] * dt_var * direction_x / norm)
    opti.subject_to(y[i+1] == y[i] + v[i] * dt_var * direction_y / norm)

    # Velocity update
    opti.subject_to(v[i+1] == v[i] + a[i] * dt_var)

# Constraints
v_max = 25.0  # m/s
v_min = 5.0
a_max = 3.0
a_min = -4.0

opti.subject_to(opti.bounded(v_min, v, v_max))
opti.subject_to(opti.bounded(a_min, a, a_max))

# Stay near track (simplified)
for i in range(N+1):
    # Distance from origin constraint (rough approximation)
    dist_sq = x[i]**2 / 100**2 + y[i]**2 / 50**2
    opti.subject_to(dist_sq <= 1.2)  # Stay within track bounds

# Initial guess
track_indices = np.linspace(0, len(centerline_x)-1, N+1).astype(int)
opti.set_initial(x, centerline_x[track_indices])
opti.set_initial(y, centerline_y[track_indices])
opti.set_initial(v, 15.0)
opti.set_initial(a, 0)

# Solve
print("\nSolving trajectory optimization...")
p_opts = {"expand": True}
s_opts = {"max_iter": 500, "print_level": 0, "tol": 1e-3, "acceptable_tol": 1e-2}
opti.solver('ipopt', p_opts, s_opts)

try:
    sol = opti.solve()
    print("  ✓ Optimization successful!")
    solved = True
except RuntimeError:
    print("  ⚠ Using best available solution")
    sol = opti.debug
    solved = False

# Extract solution
x_opt = sol.value(x)
y_opt = sol.value(y)
v_opt = sol.value(v)
a_opt = sol.value(a)

print(f"\nTrajectory results:")
print(f"  Waypoints: {N+1}")
print(f"  Average velocity: {np.mean(v_opt):.1f} m/s ({np.mean(v_opt)*3.6:.1f} km/h)")
print(f"  Max velocity: {np.max(v_opt):.1f} m/s ({np.max(v_opt)*3.6:.1f} km/h)")
print(f"  Min velocity: {np.min(v_opt):.1f} m/s ({np.min(v_opt)*3.6:.1f} km/h)")

# ===== OPTIMAL TRAJECTORY PLOT =====
print("\nGenerating optimal trajectory plot...")
fig, ax = plt.subplots(figsize=(12, 8))

# Draw track
ax.fill(outer_x, outer_y, color=BELKX_GRAY, alpha=0.2)
ax.fill(inner_x, inner_y, color='white')
ax.plot(centerline_x, centerline_y, 'k--', linewidth=1.5, alpha=0.4, label='Centerline')

# Draw optimal path
path_line = ax.plot(x_opt, y_opt, color=BELKX_BLUE, linewidth=4, label='Optimal Path', zorder=4)

# Mark start/finish
ax.plot(x_opt[0], y_opt[0], 'go', markersize=15, label='Start', zorder=5)
ax.plot(x_opt[-1], y_opt[-1], 'ro', markersize=15, label='Finish', zorder=5)

ax.set_xlabel('X Position (m)', fontsize=14)
ax.set_ylabel('Y Position (m)', fontsize=14)
ax.set_title(f'Optimal Racing Line (Avg: {np.mean(v_opt)*3.6:.1f} km/h)', fontsize=16, pad=15)
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.set_xlim(-140, 140)
ax.set_ylim(-80, 80)

plt.tight_layout()
plt.savefig('/home/altenlabs/phd/article_submission/workShop/optimization-mpc/notebooks/figures/trajectory-optimal.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Saved: trajectory-optimal.png")

# ===== VELOCITY HEATMAP =====
print("\nGenerating velocity heatmap...")
fig, ax = plt.subplots(figsize=(12, 8))

# Draw track
ax.fill(outer_x, outer_y, color=BELKX_GRAY, alpha=0.1)
ax.fill(inner_x, inner_y, color='white')

# Plot path colored by velocity
scatter = ax.scatter(x_opt, y_opt, c=v_opt*3.6, s=200, cmap='RdYlGn_r',
                    vmin=v_min*3.6, vmax=v_max*3.6, zorder=4, edgecolors='black', linewidth=1.5)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Velocity (km/h)', fontsize=14)

# Mark start/finish
ax.plot(x_opt[0], y_opt[0], 'ko', markersize=15, markerfacecolor='green',
        markeredgewidth=2, label='Start', zorder=5)
ax.plot(x_opt[-1], y_opt[-1], 'ko', markersize=15, markerfacecolor='red',
        markeredgewidth=2, label='Finish', zorder=5)

ax.set_xlabel('X Position (m)', fontsize=14)
ax.set_ylabel('Y Position (m)', fontsize=14)
ax.set_title('Velocity Distribution Along Optimal Path', fontsize=16, pad=15)
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.set_xlim(-140, 140)
ax.set_ylim(-80, 80)

plt.tight_layout()
plt.savefig('/home/altenlabs/phd/article_submission/workShop/optimization-mpc/notebooks/figures/trajectory-velocity.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Saved: trajectory-velocity.png")

print("\n" + "=" * 60)
print("✓ Example 4 Complete!")
print("  Generated 3 images in public/slides/intro-optimization-mpc-workshop/")
print("=" * 60)
