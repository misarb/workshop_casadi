"""
Vehicle and energy system models for optimization workshop.

This module provides pre-built models for:
- Battery dynamics
- Vehicle energy consumption
- Vehicle kinematics
"""

import numpy as np


def battery_soc_update(soc_current, power, dt, capacity, efficiency=0.95):
    """
    Update battery state of charge.

    Parameters:
    -----------
    soc_current : float
        Current state of charge (0-1)
    power : float
        Charging power in kW (positive = charge, negative = discharge)
    dt : float
        Time step in hours
    capacity : float
        Battery capacity in kWh
    efficiency : float, default 0.95
        Charging/discharging efficiency (0-1)

    Returns:
    --------
    soc_next : float
        Updated state of charge
    """
    # Apply efficiency based on charge/discharge
    if power >= 0:
        # Charging
        energy_change = efficiency * power * dt
    else:
        # Discharging
        energy_change = power * dt / efficiency

    soc_next = soc_current + energy_change / capacity

    return soc_next


def vehicle_power_consumption(velocity, acceleration, grade=0,
                              mass=1500, Crr=0.01, Cd=0.3, A=2.5):
    """
    Calculate vehicle power consumption.

    Parameters:
    -----------
    velocity : float
        Vehicle velocity in m/s
    acceleration : float
        Vehicle acceleration in m/s^2
    grade : float, default 0
        Road grade in radians (positive = uphill)
    mass : float, default 1500
        Vehicle mass in kg
    Crr : float, default 0.01
        Rolling resistance coefficient
    Cd : float, default 0.3
        Drag coefficient
    A : float, default 2.5
        Frontal area in m^2

    Returns:
    --------
    power : float
        Power consumption in Watts
    """
    g = 9.81  # Gravity
    rho = 1.225  # Air density

    # Power components
    P_acceleration = mass * acceleration * velocity
    P_grade = mass * g * np.sin(grade) * velocity
    P_rolling = Crr * mass * g * velocity
    P_drag = 0.5 * rho * Cd * A * velocity**3

    total_power = P_acceleration + P_grade + P_rolling + P_drag

    return total_power


def bicycle_model_continuous(state, control, wheelbase=2.7):
    """
    Continuous-time bicycle model dynamics.

    Parameters:
    -----------
    state : array-like, shape (4,)
        Current state [x, y, psi, v]
        x, y: position (m)
        psi: heading angle (rad)
        v: velocity (m/s)
    control : array-like, shape (2,)
        Control inputs [delta, a]
        delta: steering angle (rad)
        a: acceleration (m/s^2)
    wheelbase : float, default 2.7
        Vehicle wheelbase in meters

    Returns:
    --------
    state_dot : array-like, shape (4,)
        State derivatives [x_dot, y_dot, psi_dot, v_dot]
    """
    x, y, psi, v = state
    delta, a = control

    x_dot = v * np.cos(psi)
    y_dot = v * np.sin(psi)
    psi_dot = (v / wheelbase) * np.tan(delta)
    v_dot = a

    return np.array([x_dot, y_dot, psi_dot, v_dot])


# TODO: Add more models:
# - dynamic_bicycle_model() - with tire forces
# - battery_thermal_model()
# - solar_pv_model()
# - wind_turbine_model()
