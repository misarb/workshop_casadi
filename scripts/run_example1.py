#!/usr/bin/env python3
"""
Run Example 1: EV Battery Charging Optimization
Generates images for workshop presentation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from casadi import *
import os

# Create output directory if needed
os.makedirs('../public/slides/intro-optimization-mpc-workshop', exist_ok=True)

# BelkX color palette
BELKX_BLUE = '#0079C1'
BELKX_GRAY = '#6B7280'
BELKX_ORANGE = '#F59E0B'
BELKX_GREEN = '#10B981'
BELKX_RED = '#EF4444'

# Configure matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.figsize': (12, 6),
    'figure.dpi': 100,
    'lines.linewidth': 2.5,
})

print("=" * 60)
print("Example 1: EV Battery Charging Optimization")
print("=" * 60)

# ===== PARAMETERS =====
N = 13              # Number of hours
dt = 1.0            # Time step (hours)
time_hours = np.arange(N)
time_labels = ['6PM', '7PM', '8PM', '9PM', '10PM', '11PM', '12AM',
               '1AM', '2AM', '3AM', '4AM', '5AM', '6AM']

# Battery parameters
E_cap = 75.0        # Battery capacity (kWh)
P_max = 11.0        # Max charging power (kW)
eta = 0.95          # Charging efficiency
SoC_init = 0.2      # Initial SoC (20%)
SoC_target = 0.8    # Target SoC (80%)

print(f"\nBattery: {E_cap} kWh, Max power: {P_max} kW")
print(f"Initial SoC: {SoC_init*100}%, Target: {SoC_target*100}%")

# Time-of-use pricing
prices = np.array([
    0.30, 0.30, 0.30,  # 6-9 PM: Peak
    0.15, 0.15, 0.15, 0.15,  # 9 PM - 1 AM: Mid
    0.10, 0.10, 0.10, 0.10, 0.10, 0.10,  # 1-7 AM: Off-peak
])

print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}/kWh")

# ===== GENERATE PRICE VISUALIZATION =====
print("\nGenerating electricity price chart...")
fig, ax = plt.subplots(figsize=(12, 5))
colors = [BELKX_RED if p >= 0.25 else BELKX_ORANGE if p >= 0.14 else BELKX_GREEN for p in prices]
ax.bar(time_hours, prices, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Electricity Price ($/kWh)', fontsize=14)
ax.set_title('Time-of-Use Electricity Pricing', fontsize=16, pad=15)
ax.set_xticks(time_hours)
ax.set_xticklabels(time_labels, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 0.35)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=BELKX_RED, alpha=0.7, label='Peak (6-9 PM)'),
    Patch(facecolor=BELKX_ORANGE, alpha=0.7, label='Mid-peak (9 PM-1 AM)'),
    Patch(facecolor=BELKX_GREEN, alpha=0.7, label='Off-peak (1-7 AM)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

plt.tight_layout()
plt.savefig('../public/slides/intro-optimization-mpc-workshop/electricity-prices.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Saved: electricity-prices.png")

# ===== OPTIMIZATION =====
print("\nSetting up optimization problem...")
opti = Opti()

# Decision variables
P = opti.variable(N)        # Charging power
SoC = opti.variable(N+1)    # State of charge

# Objective: minimize cost
cost = 0
for t in range(N):
    cost += prices[t] * P[t] * dt

opti.minimize(cost)

# Constraints
opti.subject_to(SoC[0] == SoC_init)

for t in range(N):
    opti.subject_to(SoC[t+1] == SoC[t] + eta * P[t] * dt / E_cap)

opti.subject_to(opti.bounded(0, P, P_max))
opti.subject_to(SoC[N] >= SoC_target)

# Initial guess
energy_needed = (SoC_target - SoC_init) * E_cap
P_constant = energy_needed / (eta * N * dt)
opti.set_initial(P, P_constant)
opti.set_initial(SoC, np.linspace(SoC_init, SoC_target, N+1))

# Solve
print("\nSolving optimization...")
p_opts = {"expand": True}
s_opts = {"max_iter": 1000, "print_level": 0, "tol": 1e-6}
opti.solver('ipopt', p_opts, s_opts)

try:
    sol = opti.solve()
    print("  ✓ Optimization successful!")
except RuntimeError as e:
    print(f"  ✗ Solver failed, using debug values")
    sol = opti.debug

# Extract solution
P_opt = sol.value(P)
SoC_opt = sol.value(SoC)
cost_opt = sol.value(cost)

print(f"\nOptimal solution:")
print(f"  Total cost: ${cost_opt:.2f}")
print(f"  Energy charged: {np.sum(P_opt) * dt:.2f} kWh")
print(f"  Final SoC: {SoC_opt[-1]*100:.1f}%")

# ===== COMPARISON STRATEGIES =====
print("\nComputing comparison strategies...")

# Immediate charging
P_immediate = np.zeros(N)
SoC_immediate = np.zeros(N+1)
SoC_immediate[0] = SoC_init
for t in range(N):
    if SoC_immediate[t] < SoC_target:
        P_immediate[t] = P_max
        SoC_immediate[t+1] = SoC_immediate[t] + eta * P_immediate[t] * dt / E_cap
    else:
        P_immediate[t] = 0
        SoC_immediate[t+1] = SoC_immediate[t]
cost_immediate = np.sum(prices * P_immediate * dt)

# Constant charging
P_constant_arr = np.ones(N) * P_constant
SoC_constant = np.zeros(N+1)
SoC_constant[0] = SoC_init
for t in range(N):
    SoC_constant[t+1] = SoC_constant[t] + eta * P_constant_arr[t] * dt / E_cap
cost_constant = np.sum(prices * P_constant_arr * dt)

print(f"\nCost comparison:")
print(f"  Immediate: ${cost_immediate:.2f}")
print(f"  Constant:  ${cost_constant:.2f} (save ${cost_immediate-cost_constant:.2f})")
print(f"  Optimal:   ${cost_opt:.2f} (save ${cost_immediate-cost_opt:.2f}, {(1-cost_opt/cost_immediate)*100:.0f}%)")

# ===== GENERATE CHARGING SCHEDULE PLOT =====
print("\nGenerating charging schedule plot...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

ax1.fill_between(time_hours, 0, P_opt, alpha=0.3, color=BELKX_BLUE, label='Optimal charging')
ax1.plot(time_hours, P_opt, color=BELKX_BLUE, linewidth=3, marker='o', markersize=8)
ax1.axhline(P_max, color=BELKX_RED, linestyle='--', linewidth=2, alpha=0.7, label=f'Max power ({P_max} kW)')
ax1.set_ylabel('Charging Power (kW)', fontsize=14)
ax1.set_title('Optimal EV Charging Schedule', fontsize=16, pad=15)
ax1.legend(loc='upper right', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.5, P_max + 1)

ax2.plot(np.arange(N+1), SoC_opt*100, color=BELKX_GREEN, linewidth=3, marker='o', markersize=8, label='Optimal')
ax2.axhline(SoC_target*100, color=BELKX_RED, linestyle='--', linewidth=2, alpha=0.7, label=f'Target ({SoC_target*100:.0f}%)')
ax2.axhline(SoC_init*100, color=BELKX_GRAY, linestyle='--', linewidth=2, alpha=0.7, label=f'Initial ({SoC_init*100:.0f}%)')
ax2.set_ylabel('State of Charge (%)', fontsize=14)
ax2.set_xlabel('Time', fontsize=14)
ax2.legend(loc='lower right', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(15, 85)

ax2.set_xticks(time_hours)
ax2.set_xticklabels(time_labels, rotation=45, ha='right')

plt.tight_layout()
plt.savefig('../public/slides/intro-optimization-mpc-workshop/battery-charging-schedule.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Saved: battery-charging-schedule.png")

# ===== GENERATE COST COMPARISON PLOT =====
print("\nGenerating cost comparison plot...")
fig, ax = plt.subplots(figsize=(10, 6))

strategies = ['Immediate\n(Naive)', 'Constant\nCharging', 'Optimal\n(Optimized)']
costs = [cost_immediate, cost_constant, cost_opt]
colors_bar = [BELKX_RED, BELKX_ORANGE, BELKX_GREEN]
savings = [0, cost_immediate - cost_constant, cost_immediate - cost_opt]
savings_pct = [0, (1-cost_constant/cost_immediate)*100, (1-cost_opt/cost_immediate)*100]

bars = ax.bar(strategies, costs, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)

for i, (bar, cost, saving, pct) in enumerate(zip(bars, costs, savings, savings_pct)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
            f'${cost:.2f}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')
    if saving > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'Save ${saving:.2f}\n({pct:.0f}%)',
                ha='center', va='center', fontsize=11, color='white', fontweight='bold')

ax.set_ylabel('Total Cost ($)', fontsize=14)
ax.set_title('Charging Strategy Cost Comparison', fontsize=16, pad=15)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(costs) * 1.15)

plt.tight_layout()
plt.savefig('../public/slides/intro-optimization-mpc-workshop/battery-cost-comparison.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Saved: battery-cost-comparison.png")

print("\n" + "=" * 60)
print("✓ Example 1 Complete!")
print("  Generated 3 images in public/slides/intro-optimization-mpc-workshop/")
print("=" * 60)
