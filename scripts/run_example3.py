#!/usr/bin/env python3
"""
Run Example 3: EV Energy-Optimal Driving
Generates images for workshop presentation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from casadi import *
import os

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
print("Example 3: EV Energy-Optimal Driving")
print("=" * 60)

# ===== PARAMETERS =====
N = 100             # Number of segments
dt = 1.0            # Time step (s)
total_distance = 10000  # 10 km in meters

# Vehicle parameters
m = 1500.0          # Mass (kg)
g = 9.81            # Gravity
Crr = 0.01          # Rolling resistance
rho = 1.225         # Air density
Cd = 0.3            # Drag coefficient
A = 2.5             # Frontal area (m^2)

# Limits
v_max = 25.0        # Max speed (m/s) ~90 km/h
v_min = 10.0        # Min speed (m/s) ~36 km/h
a_max = 2.0         # Max acceleration (m/s^2)
a_min = -3.0        # Max deceleration (m/s^2)

print(f"\nVehicle: {m} kg, Cd={Cd}, A={A} m²")
print(f"Speed range: {v_min*3.6:.0f}-{v_max*3.6:.0f} km/h")

# ===== CREATE ROUTE WITH ELEVATION =====
print("\nCreating route profile...")
distance = np.linspace(0, total_distance, N+1)

# Create elevation profile with hills
elevation = np.zeros(N+1)
# Uphill: 0-3km (gain 80m)
mask1 = (distance >= 0) & (distance < 3000)
elevation[mask1] = 80 * (distance[mask1] / 3000)
# Flat: 3-5km
mask2 = (distance >= 3000) & (distance < 5000)
elevation[mask2] = 80
# Downhill: 5-7km (lose 60m)
mask3 = (distance >= 5000) & (distance < 7000)
elevation[mask3] = 80 - 60 * ((distance[mask3] - 5000) / 2000)
# Flat: 7-10km
mask4 = distance >= 7000
elevation[mask4] = 20

# Calculate grade (slope)
grade = np.zeros(N)
for i in range(N):
    grade[i] = np.arctan((elevation[i+1] - elevation[i]) / (distance[i+1] - distance[i] + 1e-6))

# Variable speed limits
v_limit = np.ones(N) * v_max
v_limit[0:30] = 60/3.6  # Urban area
v_limit[70:100] = 50/3.6  # Suburban

#===== VISUALIZE ROUTE =====
print("\nGenerating route elevation plot...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Elevation
ax1.fill_between(distance/1000, 0, elevation, alpha=0.3, color=BELKX_GRAY)
ax1.plot(distance/1000, elevation, color=BELKX_GRAY, linewidth=3)
ax1.set_ylabel('Elevation (m)', fontsize=14)
ax1.set_title('10km Route: Elevation Profile and Speed Limits', fontsize=16, pad=15)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-5, 100)

# Speed limits
ax2.fill_between(distance[:N]/1000, 0, v_limit*3.6, alpha=0.3, color=BELKX_ORANGE, step='mid')
ax2.step(distance[:N]/1000, v_limit*3.6, color=BELKX_ORANGE, linewidth=2.5, where='mid', label='Speed Limit')
ax2.set_ylabel('Speed Limit (km/h)', fontsize=14)
ax2.set_xlabel('Distance (km)', fontsize=14)
ax2.legend(loc='upper right', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('../public/slides/intro-optimization-mpc-workshop/ev-route-elevation.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Saved: ev-route-elevation.png")

# ===== OPTIMIZATION =====
print("\nSetting up optimization problem...")
opti = Opti()

# Decision variables (simplified: velocity at each segment)
v = opti.variable(N)        # Velocity (m/s)
s = opti.variable(N)        # Traveled distance

# Energy consumption model (simplified)
energy = 0
for i in range(N-1):
    # Approximate acceleration
    a = (v[i+1] - v[i]) / dt

    # Power components
    P_grade = m * g * sin(grade[i]) * v[i]
    P_rolling = Crr * m * g * v[i]
    P_drag = 0.5 * rho * Cd * A * v[i]**3
    P_accel = m * a * v[i]

    # Total power (only count positive = consuming)
    P_total = P_accel + P_grade + P_rolling + P_drag
    energy += fmax(0, P_total) * dt

opti.minimize(energy)

# Dynamics (simplified kinematics)
opti.subject_to(s[0] == 0)
for i in range(N-1):
    opti.subject_to(s[i+1] == s[i] + v[i] * dt)

# Constraints
opti.subject_to(opti.bounded(v_min, v, v_max))  # Speed limits
for i in range(N):
    opti.subject_to(v[i] <= v_limit[i])  # Position-dependent limits

# Reach destination
opti.subject_to(s[N-1] >= total_distance * 0.95)  # Relax slightly for feasibility

# Initial guess (constant speed)
v_init = total_distance / (N * dt)
opti.set_initial(v, v_init)
opti.set_initial(s, np.linspace(0, total_distance, N))

# Solve
print("\nSolving optimization...")
p_opts = {"expand": True}
s_opts = {"max_iter": 1000, "print_level": 0, "tol": 1e-4, "acceptable_tol": 1e-3}
opti.solver('ipopt', p_opts, s_opts)

try:
    sol = opti.solve()
    print("  ✓ Optimization successful!")
except RuntimeError as e:
    print(f"  ⚠ Using best available solution")
    sol = opti.debug

# Extract solution
v_opt = sol.value(v)
s_opt = sol.value(s)
energy_opt = sol.value(energy)

print(f"\nOptimal solution:")
print(f"  Energy consumption: {energy_opt/1e6:.2f} MJ = {energy_opt/3.6e6:.2f} kWh")
print(f"  Average speed: {np.mean(v_opt)*3.6:.1f} km/h")
print(f"  Total time: {N*dt/60:.1f} min")

# Comparison: Constant speed
v_constant = 60 / 3.6  # 60 km/h
energy_constant = 0
for i in range(N-1):
    P_grade = m * g * np.sin(grade[i]) * v_constant
    P_rolling = Crr * m * g * v_constant
    P_drag = 0.5 * rho * Cd * A * v_constant**3
    P_total = P_grade + P_rolling + P_drag
    energy_constant += max(0, P_total) * dt

print(f"\nComparison:")
print(f"  Constant 60 km/h: {energy_constant/3.6e6:.2f} kWh")
print(f"  Optimal: {energy_opt/3.6e6:.2f} kWh")
print(f"  Savings: {(1 - energy_opt/energy_constant)*100:.1f}%")

# ===== SPEED PROFILE PLOT =====
print("\nGenerating speed profile plot...")
fig, ax = plt.subplots(figsize=(14, 6))

# Plot optimal speed
ax.plot(s_opt/1000, v_opt*3.6, color=BELKX_BLUE, linewidth=3, label='Optimal Speed')

# Plot constant speed for comparison
ax.axhline(v_constant*3.6, color=BELKX_RED, linestyle='--', linewidth=2, alpha=0.7, label=f'Constant {v_constant*3.6:.0f} km/h')

# Show speed limit
ax.fill_between(distance[:N]/1000, 0, v_limit*3.6, alpha=0.1, color=BELKX_ORANGE, step='mid', label='Speed Limit Zone')

ax.set_xlabel('Distance (km)', fontsize=14)
ax.set_ylabel('Speed (km/h)', fontsize=14)
ax.set_title(f'Optimal vs Constant Speed (Save {(1-energy_opt/energy_constant)*100:.1f}% Energy)', fontsize=16, pad=15)
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('../public/slides/intro-optimization-mpc-workshop/ev-speed-profile.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Saved: ev-speed-profile.png")

# ===== ENERGY BREAKDOWN =====
print("\nGenerating energy breakdown plot...")

# Calculate energy components for optimal profile
E_accel_opt = 0
E_grade_opt = 0
E_rolling_opt = 0
E_drag_opt = 0

for i in range(N-1):
    a = (v_opt[i+1] - v_opt[i]) / dt

    P_a = max(0, m * a * v_opt[i]) * dt
    P_g = max(0, m * g * np.sin(grade[i]) * v_opt[i]) * dt
    P_r = Crr * m * g * v_opt[i] * dt
    P_d = 0.5 * rho * Cd * A * v_opt[i]**3 * dt

    E_accel_opt += P_a
    E_grade_opt += P_g
    E_rolling_opt += P_r
    E_drag_opt += P_d

# Same for constant
E_grade_const = 0
E_rolling_const = 0
E_drag_const = 0

for i in range(N-1):
    E_grade_const += max(0, m * g * np.sin(grade[i]) * v_constant) * dt
    E_rolling_const += Crr * m * g * v_constant * dt
    E_drag_const += 0.5 * rho * Cd * A * v_constant**3 * dt

fig, ax = plt.subplots(figsize=(10, 6))

strategies = ['Constant\n60 km/h', 'Optimal']
components = ['Acceleration', 'Grade', 'Rolling', 'Drag']
colors_stack = [BELKX_ORANGE, BELKX_RED, BELKX_GRAY, BELKX_BLUE]

data_const = np.array([0, E_grade_const, E_rolling_const, E_drag_const]) / 3.6e6
data_opt = np.array([E_accel_opt, E_grade_opt, E_rolling_opt, E_drag_opt]) / 3.6e6

x = np.arange(len(strategies))
width = 0.5

bottom_const = 0
bottom_opt = 0

for i, (component, color) in enumerate(zip(components, colors_stack)):
    if i == 0:  # Acceleration
        ax.bar(x[1], data_opt[i], width, bottom=bottom_opt, label=component, color=color, alpha=0.8)
        bottom_opt += data_opt[i]
    else:
        ax.bar(x[0], data_const[i], width, bottom=bottom_const, color=color, alpha=0.8)
        ax.bar(x[1], data_opt[i], width, bottom=bottom_opt, label=component, color=color, alpha=0.8)
        bottom_const += data_const[i]
        bottom_opt += data_opt[i]

# Add totals
ax.text(x[0], bottom_const + 0.05, f'{bottom_const:.2f} kWh', ha='center', fontsize=13, fontweight='bold')
ax.text(x[1], bottom_opt + 0.05, f'{bottom_opt:.2f} kWh', ha='center', fontsize=13, fontweight='bold')

ax.set_ylabel('Energy Consumption (kWh)', fontsize=14)
ax.set_title('Energy Consumption Breakdown', fontsize=16, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(strategies)
ax.legend(loc='upper right', fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../public/slides/intro-optimization-mpc-workshop/ev-energy-breakdown.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Saved: ev-energy-breakdown.png")

print("\n" + "=" * 60)
print("✓ Example 3 Complete!")
print("  Generated 3 images in public/slides/intro-optimization-mpc-workshop/")
print("=" * 60)
