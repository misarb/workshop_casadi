# Workshop Jupyter Notebooks

This directory contains hands-on Jupyter notebooks for the **Optimization & Model Predictive Control** workshop.

## 📚 Notebooks

### 01_ev_battery_charging.ipynb
**Topic**: Smart EV charging optimization

**You will learn**:
- Formulate time-varying cost minimization
- Model battery state of charge dynamics
- Implement in CasADi Opti framework
- Compare optimal vs. naive strategies

**Key concepts**: Decision variables, objective function, dynamics constraints

**Difficulty**: ⭐ Beginner

---

### 02_solar_battery_management.ipynb
**Topic**: Solar + battery energy management system

**You will learn**:
- Model multi-component energy system
- Handle bidirectional power flows
- Implement power balance constraints
- Consider charging/discharging efficiency

**Key concepts**: Energy balance, battery dynamics, multi-objective optimization

**Difficulty**: ⭐⭐ Intermediate

---

### 03_ev_energy_management.ipynb
**Topic**: EV energy-optimal speed planning

**You will learn**:
- Build physics-based energy consumption model
- Optimize over spatial trajectory
- Handle position-dependent constraints (speed limits)
- Apply MPC-style receding horizon

**Key concepts**: Vehicle dynamics, energy modeling, preview optimization

**Difficulty**: ⭐⭐ Intermediate

---

### 04_ev_trajectory_optimization.ipynb
**Topic**: Minimum-time trajectory planning

**You will learn**:
- Implement vehicle kinematic model
- Optimize in 2D space
- Handle track boundary constraints
- Balance time vs. energy objectives

**Key concepts**: Trajectory optimization, nonlinear dynamics, non-convex problems

**Difficulty**: ⭐⭐⭐ Advanced

---

## 🚀 Getting Started

### Prerequisites

```bash
# Install required packages
pip install casadi numpy matplotlib jupyter pandas

# Verify CasADi installation
python -c "import casadi; print(casadi.__version__)"
```

**Recommended**: Python 3.8+, CasADi 3.6+

### Running Notebooks

```bash
# Start Jupyter notebook server
jupyter notebook

# Or use JupyterLab
jupyter lab
```

Navigate to the notebook you want to run and execute cells in order.

### Notebook Structure

Each notebook follows this pattern:

1. **Introduction**: Problem description and motivation
2. **Setup**: Import libraries, define parameters
3. **Formulation**: Decision variables, objective, constraints
4. **Implementation**: CasADi code walkthrough
5. **Solve**: Run optimization
6. **Results**: Visualization and analysis
7. **Exercises**: Optional challenges to extend the example

---

## 📁 Directory Structure

```
notebooks/
├── README.md                           # This file
├── 01_ev_battery_charging.ipynb        # Example 1
├── 02_solar_battery_management.ipynb   # Example 2
├── 03_ev_energy_management.ipynb       # Example 3
├── 04_ev_trajectory_optimization.ipynb # Example 4
└── utils/
    ├── plotting.py                     # Plotting helper functions
    ├── vehicle_models.py               # Vehicle dynamics models
    └── data/
        ├── price_profiles.csv          # Electricity price data
        ├── solar_profiles.csv          # Solar generation profiles
        ├── load_profiles.csv           # Load demand profiles
        └── track_layouts.json          # Race track definitions
```

---

## 🛠️ Utilities

### plotting.py

Helper functions for consistent visualization:

```python
from utils.plotting import (
    plot_time_series,      # Time series with multiple variables
    plot_battery_schedule, # Battery charging/discharging
    plot_trajectory,       # 2D trajectory on track
    plot_comparison,       # Before/after comparison
    setup_belkx_style,     # Apply BelkX plot style
)

# Use in notebook
setup_belkx_style()
plot_time_series(time, power, xlabel='Time (h)', ylabel='Power (kW)')
```

### vehicle_models.py

Pre-built vehicle dynamics models:

```python
from utils.vehicle_models import (
    battery_dynamics,       # SoC update equation
    vehicle_energy_model,   # Power consumption
    bicycle_model,          # Kinematic bicycle model
)
```

---

## 📊 Data Files

### price_profiles.csv

Time-of-use electricity pricing:
- `hour`: 0-23
- `price_peak`: Peak rate ($/kWh)
- `price_mid`: Mid-peak rate
- `price_offpeak`: Off-peak rate

### solar_profiles.csv

Typical solar PV generation:
- `hour`: 0-23
- `generation_winter`: Winter profile (kW)
- `generation_summer`: Summer profile (kW)
- `generation_avg`: Average profile (kW)

### load_profiles.csv

Household electricity demand:
- `hour`: 0-23
- `load_weekday`: Weekday profile (kW)
- `load_weekend`: Weekend profile (kW)

### track_layouts.json

Race track definitions:
```json
{
  "track_name": {
    "centerline": [[x1, y1], [x2, y2], ...],
    "width": 6.0,
    "start": [x, y],
    "finish": [x, y]
  }
}
```

---

## 💡 Tips for Workshop

1. **Run cells in order**: Each cell builds on previous ones
2. **Experiment**: Modify parameters and re-run
3. **Check solver output**: Look for "Optimal solution found"
4. **Visualize**: Plot intermediate results to debug
5. **Ask questions**: No question is too basic!

### Common Issues

**Import error**: Make sure CasADi is installed
```bash
pip install casadi
```

**Solver fails**: Check constraints are feasible
```python
# Add this to debug
opti.debug.show_infeasibilities()
```

**Plots don't show**: Use magic command in Jupyter
```python
%matplotlib inline
```

---

## 🎯 Learning Path

**Beginner**:
1. Start with notebook 01 (battery charging)
2. Understand basic optimization setup
3. Modify parameters (prices, battery size)

**Intermediate**:
1. Work through notebooks 01-02
2. Try exercises at end of each notebook
3. Combine concepts (e.g., EV with solar charging)

**Advanced**:
1. Complete all notebooks
2. Extend examples (add uncertainty, multiple vehicles)
3. Apply to your own problems

---

## 📝 Exercises

Each notebook includes optional exercises:

- **Easy**: Modify parameters, observe changes
- **Medium**: Add new constraints or objectives
- **Hard**: Extend problem (e.g., uncertainty, multi-stage)

Solutions are provided in `solutions/` folder (created after workshop).

---

## 🔗 Additional Resources

**CasADi**:
- [Official Documentation](https://web.casadi.org/docs/)
- [CasADi Examples](https://github.com/casadi/casadi/tree/main/docs/examples/python)

**Optimization Theory**:
- Convex Optimization - Boyd & Vandenberghe (free PDF)
- Numerical Optimization - Nocedal & Wright

**MPC**:
- Model Predictive Control - Camacho & Bordons
- [MPC Toolbox Examples](https://www.mathworks.com/help/mpc/examples.html)

---

## 🤝 Contributing

Found a bug or have an improvement?

1. Test your change in a notebook
2. Document what you changed
3. Share with instructor or submit PR

---

## 📧 Support

**During workshop**: Ask instructor or teaching assistant

**After workshop**:
- Email: training@belkx.com
- Forum: community.belkx.com/workshops

---

## 📄 License

Workshop materials © 2025 BelkX Training

Licensed for educational use. Please retain attribution when sharing.
