# Sample Data Files

This directory contains sample data files for the optimization workshop examples.

## Files

### `price_profiles.csv`
Time-of-use electricity pricing profiles for 24 hours.

### `solar_profiles.csv`
Solar PV generation profiles for different weather conditions (5 kW system).

### `load_profiles.csv`
Household and commercial load demand profiles.

### `track_layouts.json`
Pre-defined race track layouts for trajectory optimization (oval, figure-8, circuit, parking).

## Usage

```python
import pandas as pd
import json

# Load pricing
prices = pd.read_csv('utils/data/price_profiles.csv')
price_profile = prices['price_tou_standard'].values

# Load solar
solar = pd.read_csv('utils/data/solar_profiles.csv')
solar_profile = solar['solar_sunny'].values

# Load demand
loads = pd.read_csv('utils/data/load_profiles.csv')
load_profile = loads['load_residential'].values

# Load tracks
with open('utils/data/track_layouts.json') as f:
    tracks = json.load(f)
```

All data files use standard SI units (kW, m, m/s).
