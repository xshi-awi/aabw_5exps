#!/usr/bin/env python
"""
Plot SAM-Freshwater Flux Composite Analysis
- 3 rows x 5 columns
- Row 1: Total FW flux (-fw)
- Row 2: Sea ice FW flux (-fw - evap - prec - snow - runoff)
- Row 3: Other FW flux (evap + prec + snow + runoff)
- Columns: PI, MH, LIG, LGM, MIS
- Data: JJA mean anomaly (High SAM - Low SAM)
- Units: mm/day
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

print('='*80)
print('Plotting SAM-Freshwater Flux Composite (3 rows x 5 cols)')
print('='*80)
print()

# Configuration
DATA_PATH = Path(__file__).parent  # fw_data folder
OUTPUT_PATH = DATA_PATH.parent  # composite_sam folder

EXPERIMENTS = ['pi', 'mh', 'lig', 'lgm', 'mis']
EXP_TITLES = ['PI', 'MH', 'LIG', 'LGM', 'MIS3']

ROW_VARS = ['total_fw', 'seaice_fw', 'other_fw']
ROW_TITLES = ['Total FW Flux', 'Sea Ice FW Flux', 'Other FW Flux']

# JJA months (June=6, July=7, August=8) -> indices 5, 6, 7 in 0-based
JJA_INDICES = [5, 6, 7]

# Unit conversion: m/s -> mm/day
MS_TO_MMDAY = 86400.0 * 1000.0

# Color scale range (mm/day)
VMAX = 3.0  # Adjusted based on data range (SO range ~±10, but most values smaller)

# Southern Ocean region
LAT_MIN = -90
LAT_MAX = -50


def add_cyclic_point(data, lon):
    """Add cyclic point to avoid longitude gap"""
    if len(data.shape) == 2:
        cyclic_data = np.concatenate([data, data[:, 0:1]], axis=1)
    else:
        cyclic_data = np.concatenate([data, data[0:1]], axis=0)
    cyclic_lon = np.concatenate([lon, [lon[0] + 360]])
    return cyclic_data, cyclic_lon


def load_jja_anomaly(var_name, exp_key):
    """Load high and low SAM data, compute JJA anomaly (High - Low)"""

    high_file = DATA_PATH / f'{var_name}_highsam_{exp_key}_reg.nc'
    low_file = DATA_PATH / f'{var_name}_lowsam_{exp_key}_reg.nc'

    if not high_file.exists() or not low_file.exists():
        print(f'  WARNING: Missing files for {var_name}_{exp_key}')
        return None, None, None

    # Load data
    ds_high = xr.open_dataset(high_file)
    ds_low = xr.open_dataset(low_file)

    # Get data - shape is (time=12, depth=1, lat, lon)
    high_data = ds_high[var_name].values  # (12, 1, lat, lon)
    low_data = ds_low[var_name].values

    lon = ds_high['lon'].values
    lat = ds_high['lat'].values

    # Extract JJA months and compute mean
    high_jja = np.nanmean(high_data[JJA_INDICES, 0, :, :], axis=0)  # (lat, lon)
    low_jja = np.nanmean(low_data[JJA_INDICES, 0, :, :], axis=0)

    # Compute anomaly (High - Low) and convert to mm/day
    anomaly = (high_jja - low_jja) * MS_TO_MMDAY

    ds_high.close()
    ds_low.close()

    return anomaly, lon, lat


def plot_composite():
    """Create 3x5 composite plot"""

    fig = plt.figure(figsize=(20, 12))

    # Projection for Southern Ocean polar stereographic
    proj = ccrs.SouthPolarStereo()
    data_proj = ccrs.PlateCarree()

    # Create axes
    axes = []
    for row in range(3):
        row_axes = []
        for col in range(5):
            ax = fig.add_subplot(3, 5, row * 5 + col + 1, projection=proj)
            ax.set_extent([-180, 180, LAT_MIN, LAT_MAX], crs=data_proj)
            row_axes.append(ax)
        axes.append(row_axes)

    # Color normalization (symmetric around 0)
    norm = TwoSlopeNorm(vmin=-VMAX, vcenter=0, vmax=VMAX)
    cmap = plt.cm.RdBu_r

    # Store one mappable for colorbar
    cf_last = None

    for col, exp_key in enumerate(EXPERIMENTS):
        print(f'\nProcessing {exp_key.upper()}...')

        for row, (var_name, row_title) in enumerate(zip(ROW_VARS, ROW_TITLES)):
            ax = axes[row][col]

            # Load JJA anomaly
            anomaly, lon, lat = load_jja_anomaly(var_name, exp_key)

            if anomaly is None:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
                ax.coastlines(resolution='50m', linewidth=0.5, color='black')
                continue

            # Print statistics
            so_mask = lat < -50
            so_data = anomaly[so_mask, :]
            print(f'  {var_name}: SO mean={np.nanmean(so_data):.4f}, range=[{np.nanmin(so_data):.4f}, {np.nanmax(so_data):.4f}] mm/day')

            # Add cyclic point to avoid longitude gap
            data_cyclic, lon_cyclic = add_cyclic_point(anomaly, lon)

            # Define contour levels for smooth plotting
            levels = np.linspace(-VMAX, VMAX, 31)

            # Plot using contourf for smooth appearance
            cf = ax.contourf(lon_cyclic, lat, data_cyclic,
                            levels=levels,
                            cmap=cmap, norm=norm,
                            transform=data_proj,
                            extend='both')
            cf_last = cf

            # Add land (gray) on top of data
            ax.add_feature(cfeature.LAND, facecolor='gray', edgecolor='black', linewidth=0.5, zorder=10)
            ax.coastlines(resolution='50m', linewidth=0.5, color='black', zorder=11)

            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                             alpha=0.5, linestyle='--',
                             x_inline=False, y_inline=True)
            gl.bottom_labels = False
            gl.xlabel_style = {'size': 10, 'color': '#333333'}
            gl.ylabel_style = {'size': 9, 'color': '#333333'}
            gl.top_labels = False
            gl.right_labels = False

            # Add panel label (a), (b), (c)... in top-left corner
            panel_idx = row * 5 + col
            panel_label = f'({chr(97 + panel_idx)})'
            ax.text(0.05, 0.95, panel_label, transform=ax.transAxes,
                    fontsize=20, fontweight='bold', va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9),
                    zorder=100)

            # Row title (leftmost column)
            if col == 0:
                ax.text(-0.15, 0.5, row_title, transform=ax.transAxes,
                       fontsize=18, fontweight='bold', rotation=90,
                       ha='center', va='center')

            # Column title (top row)
            if row == 0:
                ax.set_title(EXP_TITLES[col], fontsize=22, fontweight='bold', pad=10)

    # Add colorbar
    cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.025])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=cbar_ax, orientation='horizontal', extend='both')
    cbar.set_label('Freshwater Flux Anomaly (High SAM - Low SAM) [mm/day]', fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.14,
                       wspace=0.05, hspace=0.08)

    # Save
    output_file = OUTPUT_PATH / 'sam_fwflux_composite_3rows_5cols.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'\nSaved: {output_file}')

    output_png = OUTPUT_PATH / 'sam_fwflux_composite_3rows_5cols.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_png}')

    plt.close()


if __name__ == '__main__':
    plot_composite()
    print('\nDone!')
