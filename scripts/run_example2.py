#!/usr/bin/env python3
"""
Run Example 2: Solar + Battery Energy Management
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
BELKX_YELLOW = '#FBBF24'

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'lines.linewidth': 2.5,
})

print("=" * 60)
print("Example 2: Solar + Battery Energy Management")
print("=" * 60)

# ===== PARAMETERS =====
N = 24              # 24 hours
dt = 1.0            # 1 hour timesteps
time_hours = np.arange(N)

# Battery parameters
E_bat_cap = 13.5    # Battery capacity (kWh) - Tesla Powerwall size
P_bat_max = 5.0     # Max battery power (kW)
eta_charge = 0.95   # Charging efficiency
eta_discharge = 0.95
SoC_init = 0.5      # Initial 50%

# Grid parameters
P_grid_max = 20.0   # Grid connection limit

print(f"\nSystem specifications:")
print(f"  Battery: {E_bat_cap} kWh, ±{P_bat_max} kW")
print(f"  Grid limit: {P_grid_max} kW")
print(f"  Initial SoC: {SoC_init*100}%")

# ===== GENERATE REALISTIC DATA =====
print("\nGenerating 24h profiles...")

# Solar generation (kW) - typical 5kW system
solar_profile = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 12AM-6AM: Night
    0.2, 0.8, 1.8, 2.9, 4.0, 4.8,  # 6AM-12PM: Morning rise
    5.0, 4.9, 4.5, 3.8, 2.8, 1.5,  # 12PM-6PM: Afternoon decline
    0.5, 0.1, 0.0, 0.0, 0.0, 0.0   # 6PM-12AM: Evening/night
])

# Load demand (kW) - typical household
load_profile = np.array([
    0.5, 0.4, 0.4, 0.4, 0.4, 0.6,  # 12AM-6AM: Night (minimal)
    1.2, 2.0, 2.5, 2.0, 1.5, 1.8,  # 6AM-12PM: Morning peak
    2.0, 2.2, 2.0, 2.5, 3.5, 4.0,  # 12PM-6PM: Afternoon rise
    3.8, 3.0, 2.0, 1.5, 1.0, 0.7   # 6PM-12AM: Evening peak, bedtime
])

# Electricity prices ($/kWh)
price_profile = np.array([
    0.10, 0.10, 0.10, 0.10, 0.10, 0.12,  # 12AM-6AM: Off-peak
    0.15, 0.18, 0.22, 0.20, 0.18, 0.20,  # 6AM-12PM: Morning rise
    0.25, 0.25, 0.25, 0.28, 0.30, 0.35,  # 12PM-6PM: Afternoon peak
    0.38, 0.35, 0.28, 0.22, 0.15, 0.12   # 6PM-12AM: Evening peak, decline
])

# ===== VISUALIZE INPUT DATA =====
print("\nGenerating input data visualization...")
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Solar generation
axes[0].fill_between(time_hours, 0, solar_profile, alpha=0.3, color=BELKX_YELLOW)
axes[0].plot(time_hours, solar_profile, color=BELKX_ORANGE, linewidth=3, marker='o', markersize=6)
axes[0].set_ylabel('Solar Generation (kW)', fontsize=14)
axes[0].set_title('24-Hour Profiles: Solar, Load, and Prices', fontsize=16, pad=15)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 6)

# Load demand
axes[1].fill_between(time_hours, 0, load_profile, alpha=0.3, color=BELKX_BLUE)
axes[1].plot(time_hours, load_profile, color=BELKX_BLUE, linewidth=3, marker='s', markersize=6)
axes[1].set_ylabel('Load Demand (kW)', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 5)

# Electricity prices
axes[2].fill_between(time_hours, 0, price_profile, alpha=0.3, color=BELKX_RED)
axes[2].plot(time_hours, price_profile, color=BELKX_RED, linewidth=3, marker='^', markersize=6)
axes[2].set_ylabel('Price ($/kWh)', fontsize=14)
axes[2].set_xlabel('Hour of Day', fontsize=14)
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(0, 0.42)

for ax in axes:
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f'{h}:00' for h in range(0, 24, 2)], rotation=45, ha='right')

plt.tight_layout()
plt.savefig('../public/slides/intro-optimization-mpc-workshop/solar-battery-inputs.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Saved: solar-battery-inputs.png")

# ===== OPTIMIZATION =====
print("\nSetting up optimization problem...")
opti = Opti()

# Decision variables
P_bat = opti.variable(N)        # Battery power (positive = charge)
P_grid = opti.variable(N)       # Grid power (positive = import)
E_bat = opti.variable(N+1)      # Battery energy

# Objective: minimize grid cost
grid_cost = 0
for t in range(N):
    grid_cost += price_profile[t] * P_grid[t] * dt
opti.minimize(grid_cost)

# Power balance: Solar + Grid = Load + Battery
for t in range(N):
    opti.subject_to(
        solar_profile[t] + P_grid[t] == load_profile[t] + P_bat[t]
    )

# Grid constraints (only import, no export for simplicity)
opti.subject_to(opti.bounded(0, P_grid, P_grid_max))

# Battery power constraints
opti.subject_to(opti.bounded(-P_bat_max, P_bat, P_bat_max))

# Battery dynamics
opti.subject_to(E_bat[0] == SoC_init * E_bat_cap)

for t in range(N):
    # Separate charge and discharge with efficiency
    E_charge = fmax(0, P_bat[t]) * dt * eta_charge
    E_discharge = fmax(0, -P_bat[t]) * dt / eta_discharge

    opti.subject_to(
        E_bat[t+1] == E_bat[t] + E_charge - E_discharge
    )

# Battery capacity constraints
opti.subject_to(opti.bounded(0, E_bat, E_bat_cap))

# Cyclic constraint (end at same SoC)
opti.subject_to(E_bat[N] >= SoC_init * E_bat_cap)

# Initial guess
opti.set_initial(P_bat, 0)
opti.set_initial(P_grid, load_profile - solar_profile)
opti.set_initial(E_bat, SoC_init * E_bat_cap)

# Solve
print("\nSolving optimization...")
p_opts = {"expand": True}
s_opts = {"max_iter": 2000, "print_level": 0, "tol": 1e-6}
opti.solver('ipopt', p_opts, s_opts)

try:
    sol = opti.solve()
    print("  ✓ Optimization successful!")
except RuntimeError as e:
    print(f"  ✗ Solver failed, using debug values")
    sol = opti.debug

# Extract solution
P_bat_opt = sol.value(P_bat)
P_grid_opt = sol.value(P_grid)
E_bat_opt = sol.value(E_bat)
cost_opt = sol.value(grid_cost)

# Calculate derived quantities
P_solar_to_load = np.minimum(solar_profile, load_profile)
P_solar_to_bat = np.maximum(0, P_bat_opt) * (solar_profile > load_profile)

print(f"\nOptimal solution:")
print(f"  Grid cost: ${cost_opt:.2f}")
print(f"  Total grid import: {np.sum(P_grid_opt) * dt:.2f} kWh")
print(f"  Final battery SoC: {E_bat_opt[-1]/E_bat_cap*100:.1f}%")
print(f"  Self-consumption: {(1 - np.sum(P_grid_opt)/np.sum(load_profile))*100:.1f}%")

# Comparison: No battery
P_grid_no_battery = np.maximum(0, load_profile - solar_profile)
cost_no_battery = np.sum(price_profile * P_grid_no_battery * dt)

# Comparison: No solar, no battery
cost_no_solar_battery = np.sum(price_profile * load_profile * dt)

print(f"\nCost comparison:")
print(f"  No solar, no battery: ${cost_no_solar_battery:.2f}")
print(f"  Solar only: ${cost_no_battery:.2f} (save ${cost_no_solar_battery-cost_no_battery:.2f}, {(1-cost_no_battery/cost_no_solar_battery)*100:.0f}%)")
print(f"  Solar + Battery (optimal): ${cost_opt:.2f} (save ${cost_no_solar_battery-cost_opt:.2f}, {(1-cost_opt/cost_no_solar_battery)*100:.0f}%)")

# ===== GENERATE POWER SCHEDULE PLOT =====
print("\nGenerating power schedule plot...")
fig, ax = plt.subplots(figsize=(14, 7))

# Create stacked area plot
bottom = np.zeros(N)

# Solar to load (green)
ax.fill_between(time_hours, bottom, bottom + P_solar_to_load,
                alpha=0.7, color=BELKX_GREEN, label='Solar → Load', edgecolor='black', linewidth=0.5)
bottom += P_solar_to_load

# Grid import (red)
ax.fill_between(time_hours, bottom, bottom + P_grid_opt,
                alpha=0.7, color=BELKX_RED, label='Grid → Load', edgecolor='black', linewidth=0.5)

# Battery discharge (blue, below zero)
P_bat_discharge = np.minimum(0, P_bat_opt)
ax.fill_between(time_hours, 0, P_bat_discharge,
                alpha=0.7, color=BELKX_BLUE, label='Battery → Load', edgecolor='black', linewidth=0.5)

# Battery charge (orange, below solar)
P_bat_charge = np.maximum(0, P_bat_opt)
ax.fill_between(time_hours, -P_bat_charge, 0,
                alpha=0.7, color=BELKX_ORANGE, label='Solar → Battery', edgecolor='black', linewidth=0.5)

# Load demand line
ax.plot(time_hours, load_profile, color='black', linewidth=3, linestyle='--', marker='o', markersize=5, label='Load Demand')

ax.set_xlabel('Hour of Day', fontsize=14)
ax.set_ylabel('Power (kW)', fontsize=14)
ax.set_title('Optimal 24h Power Schedule', fontsize=16, pad=15)
ax.legend(loc='upper left', fontsize=12, ncol=2)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', linewidth=1)
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([f'{h}:00' for h in range(0, 24, 2)], rotation=45, ha='right')

plt.tight_layout()
plt.savefig('../public/slides/intro-optimization-mpc-workshop/solar-battery-schedule.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Saved: solar-battery-schedule.png")

# ===== GENERATE BATTERY SOC PLOT =====
print("\nGenerating battery SoC plot...")
fig, ax = plt.subplots(figsize=(14, 6))

SoC_pct = E_bat_opt / E_bat_cap * 100

ax.fill_between(np.arange(N+1), 0, SoC_pct, alpha=0.3, color=BELKX_BLUE)
ax.plot(np.arange(N+1), SoC_pct, color=BELKX_BLUE, linewidth=3, marker='o', markersize=7)
ax.axhline(SoC_init*100, color=BELKX_GRAY, linestyle='--', linewidth=2, alpha=0.7, label=f'Initial/Final ({SoC_init*100:.0f}%)')

ax.set_xlabel('Hour of Day', fontsize=14)
ax.set_ylabel('State of Charge (%)', fontsize=14)
ax.set_title('Battery State of Charge Over 24 Hours', fontsize=16, pad=15)
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)
ax.set_xticks(range(0, 25, 2))
ax.set_xticklabels([f'{h%24}:00' for h in range(0, 25, 2)], rotation=45, ha='right')

# Annotate key events
max_soc_idx = np.argmax(SoC_pct)
min_soc_idx = np.argmin(SoC_pct)
ax.annotate(f'Peak: {SoC_pct[max_soc_idx]:.1f}%',
            xy=(max_soc_idx, SoC_pct[max_soc_idx]),
            xytext=(max_soc_idx-2, SoC_pct[max_soc_idx]+10),
            arrowprops=dict(arrowstyle='->', color=BELKX_GREEN, lw=2),
            fontsize=12, fontweight='bold', color=BELKX_GREEN)

plt.tight_layout()
plt.savefig('../public/slides/intro-optimization-mpc-workshop/solar-battery-soc.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Saved: solar-battery-soc.png")

print("\n" + "=" * 60)
print("✓ Example 2 Complete!")
print("  Generated 3 images in public/slides/intro-optimization-mpc-workshop/")
print("=" * 60)
