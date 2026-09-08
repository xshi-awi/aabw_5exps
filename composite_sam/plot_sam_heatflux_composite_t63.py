#!/usr/bin/env python
"""
Plot SAM-Heat Flux Composite Analysis on T63 Grid
- 3 rows x 5 columns
- Row 1: Total heat flux (SW+LW+SH+LH)
- Row 2: Radiative flux (SW+LW)
- Row 3: Turbulent flux (SH+LH)
- Columns: PI, MH, LIG, LGM, MIS
- Plot on ECHAM T63 Gaussian grid (no interpolation)
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

print('='*80)
print('Plotting SAM-Heat Flux Composite on T63 Grid (3 rows x 5 cols)')
print('='*80)
print()

# Configuration
BASE_PATH = Path('/work/ba1066/a270064/cc_projects/aabw_5exps')
OUTPUT_PATH = BASE_PATH / 'composite_sam'

EXPERIMENTS = ['pi', 'mh', 'lig', 'lgm', 'mis']
EXP_TITLES = ['PI', 'MH', 'LIG', 'LGM', 'MIS3']

ROW_VARS = ['total_heat_diff', 'radiative_diff', 'turbulent_diff']
ROW_TITLES = ['Total Heat Flux\n(SW+LW+SH+LH)', 'Radiative Flux\n(SW+LW)', 'Turbulent Flux\n(SH+LH)']

# Color scale range (W/m2)
VMAX = 15.0

# Southern Ocean region
LAT_MIN = -90
LAT_MAX = -50


def add_cyclic_point(data, lon):
    """添加循环点避免经度缝隙"""
    if len(data.shape) == 2:
        cyclic_data = np.concatenate([data, data[:, 0:1]], axis=1)
    else:
        cyclic_data = np.concatenate([data, data[0:1]], axis=0)
    cyclic_lon = np.concatenate([lon, [lon[0] + 360]])
    return cyclic_data, cyclic_lon


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

    for col, exp_key in enumerate(EXPERIMENTS):
        print(f'\nProcessing {exp_key.upper()}...')

        # Load composite data from T63 grid file
        input_file = OUTPUT_PATH / f'heatflux_sam_composite_t63_{exp_key}.nc'

        if not input_file.exists():
            print(f'  File not found: {input_file}')
            for row in range(3):
                ax = axes[row][col]
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
                ax.coastlines(resolution='50m', linewidth=0.5, color='black')
            continue

        ds = xr.open_dataset(input_file)

        lon = ds['lon'].values
        lat = ds['lat'].values

        for row, (var_name, row_title) in enumerate(zip(ROW_VARS, ROW_TITLES)):
            ax = axes[row][col]

            # Get data
            data = ds[var_name].values  # (lat, lon)

            # Add cyclic point to avoid longitude gap
            data_cyclic, lon_cyclic = add_cyclic_point(data, lon)

            # Define contour levels for smooth plotting
            levels = np.linspace(-VMAX, VMAX, 31)

            # Plot using contourf for smooth appearance
            cf = ax.contourf(lon_cyclic, lat, data_cyclic,
                            levels=levels,
                            cmap=cmap, norm=norm,
                            transform=data_proj,
                            extend='both')

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
            panel_label = f'({chr(97 + panel_idx)})'  # (a), (b), (c)...
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

        ds.close()

    # Add colorbar
    cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.025])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=cbar_ax, orientation='horizontal', extend='both')
    cbar.set_label('Heat Flux Anomaly (High SAM - Low SAM) [W/m$^2$]', fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.14,
                       wspace=0.05, hspace=0.08)

    # Save
    output_file = OUTPUT_PATH / 'sam_heatflux_composite_t63_3rows_5cols.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'\nSaved: {output_file}')

    output_png = OUTPUT_PATH / 'sam_heatflux_composite_t63_3rows_5cols.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_png}')

    plt.close()


if __name__ == '__main__':
    plot_composite()
    print('\nDone!')
