#!/usr/bin/env python3
"""
Plot vertical profiles of temperature, salinity, and density for 5 experiments
South of 50°S, annual mean
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import gsw  # For density calculation

# Experiment names and labels
exps = ['pi', 'mh', 'lig', 'lgm', 'mis']
exp_labels = ['PI', 'MH', 'LIG', 'LGM', 'MIS3']

# Initialize storage
profiles = {}

print("Loading data for all experiments...")
for exp in exps:
    print(f"Processing {exp.upper()}...")

    # Load temperature and salinity
    ds_temp = xr.open_dataset(f'./{exp}/temp_reg.nc')
    ds_salt = xr.open_dataset(f'./{exp}/salt_reg.nc')

    # Extract variables
    temp = ds_temp['temp']  # [time, depth, lat, lon]
    salt = ds_salt['salt']  # [time, depth, lat, lon]
    lat = ds_temp['lat'].values
    depth_raw = ds_temp['depth_coord'].values  # 负值: 0 to -6000

    # Convert depth to positive values (for plotting with 0 at top)
    depth = np.abs(depth_raw)

    # Annual mean (average over time dimension)
    temp_annual = temp.mean(dim='time')
    salt_annual = salt.mean(dim='time')

    # Mask for 50°S and south
    lat_mask = lat < -65

    # Select southern ocean data
    temp_south = temp_annual.sel(lat=lat[lat_mask])
    salt_south = salt_annual.sel(lat=lat[lat_mask])

    # Compute area-weighted mean over lat/lon
    temp_profile = temp_south.mean(dim=['lat', 'lon']).values
    salt_profile = salt_south.mean(dim=['lat', 'lon']).values

    # Remove NaN values (deep layers without data)
    valid_mask = ~np.isnan(temp_profile) & ~np.isnan(salt_profile)
    depth_valid = depth[valid_mask]
    temp_valid = temp_profile[valid_mask]
    salt_valid = salt_profile[valid_mask]

    # Calculate potential density (sigma0, referenced to 0 dbar)
    # Using GSW for proper density calculation
    pressure = gsw.p_from_z(-depth_valid, -60)  # Approximate pressure at 60°S
    SA = gsw.SA_from_SP(salt_valid, pressure, 0, -60)
    CT = gsw.CT_from_t(SA, temp_valid, pressure)
    sigma0 = gsw.sigma0(SA, CT)

    profiles[exp] = {
        'depth': depth_valid,
        'temp': temp_valid,
        'salt': salt_valid,
        'density': sigma0
    }

    print(f"  Valid depth levels: {len(depth_valid)} (max depth: {depth_valid.max():.0f} m)")
    print(f"  Temperature range: {temp_valid.min():.2f} to {temp_valid.max():.2f} °C")
    print(f"  Salinity range: {salt_valid.min():.2f} to {salt_valid.max():.2f} PSU")
    print(f"  Density (σ₀) range: {sigma0.min():.2f} to {sigma0.max():.2f} kg/m³")

    ds_temp.close()
    ds_salt.close()

print("\nCreating figure...")

# Create figure with 5 subplots (one per experiment)
fig, axes = plt.subplots(1, 5, figsize=(18, 8), sharey=True)
fig.subplots_adjust(wspace=0.4, left=0.06, right=0.98, top=0.85, bottom=0.10)

for idx, (exp, label) in enumerate(zip(exps, exp_labels)):
    ax = axes[idx]

    depth = profiles[exp]['depth']
    temp = profiles[exp]['temp']
    salt = profiles[exp]['salt']
    dens = profiles[exp]['density']

    # Create three x-axes for T, S, and density
    ax2 = ax.twiny()
    ax3 = ax.twiny()

    # Offset the third axis
    ax3.spines['top'].set_position(('outward', 50))

    # Plot profiles with different line styles
    p1, = ax.plot(temp, depth, color='red', linewidth=2, label='Temperature')
    p2, = ax2.plot(salt, depth, color='blue', linewidth=2, label='Salinity')
    p3, = ax3.plot(dens, depth, color='darkgreen', linewidth=2, label='Density')

    # Set depth axis (inverted, 0 at top)
    ax.set_ylim(depth.max() + 100, 0)
    if idx == 0:
        ax.set_ylabel('Depth (m)', fontsize=12)

    # Set x-axes labels
    ax.set_xlabel('T (°C)', fontsize=10, color='red')
    ax2.set_xlabel('S (PSU)', fontsize=10, color='blue')
    ax3.set_xlabel('σ₀ (kg/m³)', fontsize=10, color='darkgreen')

    # Color x-axis ticks and spines
    ax.tick_params(axis='x', labelcolor='red', colors='red', labelsize=9)
    ax.spines['bottom'].set_color('red')
    ax2.tick_params(axis='x', labelcolor='blue', colors='blue', labelsize=9)
    ax2.spines['top'].set_color('blue')
    ax3.tick_params(axis='x', labelcolor='darkgreen', colors='darkgreen', labelsize=9)
    ax3.spines['top'].set_color('darkgreen')

    # Set x-axis limits based on data range (with some padding)
    t_min, t_max = temp.min(), temp.max()
    s_min, s_max = salt.min(), salt.max()
    d_min, d_max = dens.min(), dens.max()

    ax.set_xlim(t_min - 0.3, t_max + 0.3)
    ax2.set_xlim(s_min - 0.05, s_max + 0.05)
    ax3.set_xlim(d_min - 0.1, d_max + 0.1)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Title for each subplot
    ax.set_title(f'{label}', fontsize=14, fontweight='bold', pad=70)

    # Add legend only to first subplot
    if idx == 0:
        lines = [p1, p2, p3]
        labels_leg = ['Temperature', 'Salinity', 'Density (σ₀)']
        ax.legend(lines, labels_leg, loc='lower left', fontsize=9, framealpha=0.9)

# Save figure
output_file = 'plot_vertical_profiles_5exps.pdf'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nFigure saved: {output_file}")
