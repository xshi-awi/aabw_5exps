#!/usr/bin/env python3
"""
Plot publication-quality climate pattern figures for AABW paper
Figure 1: SST/SSS/Density anomalies (JJA) - 3 rows x 4 columns
Figure 2: MLD/Sea Ice absolute values (JJA) - 2 rows x 5 columns
Figure 3: U10 wind anomalies (JJA) - 1 row x 4 columns

Uses interpolated regular grid data (*_reg.nc files)

Author: Claude Code
Date: 2026-01-06
Updated: 2026-01-08 - Improved colorbars, panel labels, smoothing
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
import gsw
import warnings
warnings.filterwarnings('ignore')

# Experiment names and labels
exps = ['pi', 'mh', 'lig', 'lgm', 'mis']
exp_labels = {'pi': 'PI', 'mh': 'MH', 'lig': 'LIG', 'lgm': 'LGM', 'mis': 'MIS3'}
paleo_exps = ['mh', 'lig', 'lgm', 'mis']

# Base directory
base_dir = '/work/ba0989/a270064/bb1029/wmt/exps'

# JJA months (austral winter): June(5), July(6), August(7) - 0-indexed for Python
jja_months = [5, 6, 7]

def load_variable_reg(exp, var_name, jja_only=True):
    """Load variable from regular grid and optionally compute JJA mean"""
    if var_name in ['u10', 'v10']:
        ds = xr.open_dataset(f'{base_dir}/{exp}/{var_name}_reg.nc')
        if var_name == 'u10':
            data = ds['var165']
        else:
            data = ds['var166']
    else:
        ds = xr.open_dataset(f'{base_dir}/{exp}/{var_name}_reg.nc')
        data = ds[var_name]

    # Handle depth_coord dimension if present (squeeze it out)
    if 'depth_coord' in data.dims:
        data = data.isel(depth_coord=0)

    if jja_only:
        # Select JJA months (time index 5,6,7 for June,July,August)
        data_jja = data.isel(time=jja_months).mean(dim='time')
    else:
        data_jja = data

    lon = ds['lon'].values
    lat = ds['lat'].values

    ds.close()

    return data_jja.values, lon, lat

def calculate_density(sst, sss, lon_grid, lat_grid):
    """Calculate potential density (sigma2) from SST and SSS on regular grid"""
    p_ref = 2000  # Reference pressure for sigma2

    # Broadcast lon/lat to 2D if needed
    if lon_grid.ndim == 1:
        lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
    else:
        lon_2d, lat_2d = lon_grid, lat_grid

    # Calculate absolute salinity and conservative temperature
    SA = gsw.SA_from_SP(sss, p=0, lon=lon_2d, lat=lat_2d)
    CT = gsw.CT_from_t(SA, sst, p=0)

    # Potential density
    sigma2 = gsw.sigma2(SA, CT)

    return sigma2

def smooth_data(data, sigma=1.5):
    """Apply Gaussian smoothing to data, handling NaN values"""
    # Create a copy and replace NaN with 0 for smoothing
    data_filled = np.nan_to_num(data, nan=0.0)

    # Create a mask for valid data
    mask = ~np.isnan(data)
    mask_float = mask.astype(float)

    # Smooth both data and mask
    smoothed = gaussian_filter(data_filled, sigma=sigma)
    smoothed_mask = gaussian_filter(mask_float, sigma=sigma)

    # Normalize by mask to handle edges properly
    with np.errstate(divide='ignore', invalid='ignore'):
        result = smoothed / smoothed_mask
        result[smoothed_mask < 0.1] = np.nan

    return result

def add_cyclic_point(data, lon):
    """
    Add cyclic (wraparound) point in longitude to avoid discontinuity
    Returns extended data and longitude arrays
    """
    if lon.ndim == 1:
        # 1D longitude array
        if data.ndim == 2:
            # 2D data array (lat, lon)
            cyclic_data = np.concatenate([data, data[:, 0:1]], axis=1)
        else:
            # 1D data array
            cyclic_data = np.concatenate([data, [data[0]]])

        cyclic_lon = np.concatenate([lon, [lon[0] + 360]])
    else:
        # 2D longitude array (lat, lon)
        if data.ndim == 2:
            cyclic_data = np.concatenate([data, data[:, 0:1]], axis=1)
            cyclic_lon = np.concatenate([lon, lon[:, 0:1] + 360], axis=1)
        else:
            raise ValueError("Dimension mismatch between data and lon")

    return cyclic_data, cyclic_lon

def plot_figure1():
    """
    Figure 1: SST/SSS/Density anomalies (JJA)
    Layout: 3 rows (SST, SSS, Density) x 4 columns (MH, LIG, LGM, MIS3)
    MH/LIG share one colorbar at bottom, LGM/MIS share another colorbar at bottom
    """
    print("Creating Figure 1: SST/SSS/Density anomalies...")

    fig = plt.figure(figsize=(16, 14))

    # Load PI reference data
    print("  Loading PI reference data...")
    sst_pi, lon, lat = load_variable_reg('pi', 'sst')
    sss_pi, _, _ = load_variable_reg('pi', 'sss')
    lon_2d, lat_2d = np.meshgrid(lon, lat)
    dens_pi = calculate_density(sst_pi, sss_pi, lon, lat)

    # Smooth PI data for contours
    sst_pi_smooth = smooth_data(sst_pi, sigma=2)
    sss_pi_smooth = smooth_data(sss_pi, sigma=2)
    dens_pi_smooth = smooth_data(dens_pi, sigma=2)

    # Plot parameters
    vars_info = [
        {'name': 'SST', 'pi_data': sst_pi, 'pi_smooth': sst_pi_smooth, 'unit': '°C',
         'anom_levels_warm': np.linspace(-2, 2, 21),
         'anom_levels_cold': np.linspace(-10, 10, 21),
         'pi_levels': np.arange(-2, 14, 2),
         'cmap': 'RdBu_r'},
        {'name': 'SSS', 'pi_data': sss_pi, 'pi_smooth': sss_pi_smooth, 'unit': 'psu',
         'anom_levels_warm': np.linspace(-0.5, 0.5, 21),
         'anom_levels_cold': np.linspace(-2, 2, 21),
         'pi_levels': np.arange(33, 35.5, 0.5),
         'cmap': 'RdBu_r'},
        {'name': 'Density', 'pi_data': dens_pi, 'pi_smooth': dens_pi_smooth, 'unit': 'kg/m³',
         'anom_levels_warm': np.linspace(-0.2, 0.2, 21),
         'anom_levels_cold': np.linspace(-0.8, 0.8, 21),
         'pi_levels': np.arange(35.5, 38, 0.5),
         'cmap': 'RdBu_r'}
    ]

    # Create GridSpec for better layout control
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.12,
                  left=0.08, right=0.95, top=0.92, bottom=0.08)

    # Loop over rows (variables) and columns (experiments)
    panel_label = 0
    axes_dict = {}  # Store axes for colorbar positioning

    for row, var_info in enumerate(vars_info):
        print(f"  Plotting {var_info['name']}...")

        # Store images for colorbars
        im_warm = None  # For MH/LIG
        im_cold = None  # For LGM/MIS
        axes_warm = []
        axes_cold = []

        for col, exp in enumerate(paleo_exps):
            ax = fig.add_subplot(gs[row, col], projection=ccrs.SouthPolarStereo())
            axes_dict[(row, col)] = ax

            if exp in ['mh', 'lig']:
                axes_warm.append(ax)
            else:
                axes_cold.append(ax)

            # Load paleo experiment data
            if var_info['name'] == 'SST':
                data_exp, _, _ = load_variable_reg(exp, 'sst')
            elif var_info['name'] == 'SSS':
                data_exp, _, _ = load_variable_reg(exp, 'sss')
            else:  # Density
                sst_exp, _, _ = load_variable_reg(exp, 'sst')
                sss_exp, _, _ = load_variable_reg(exp, 'sss')
                data_exp = calculate_density(sst_exp, sss_exp, lon, lat)

            # Calculate anomaly and smooth
            anomaly = data_exp - var_info['pi_data']
            anomaly_smooth = smooth_data(anomaly, sigma=1.5)

            # Add cyclic point to avoid longitude discontinuity
            anomaly_cyclic, lon_cyclic = add_cyclic_point(anomaly_smooth, lon)
            if lon_2d.ndim == 1:
                lon_2d_cyclic, lat_2d_cyclic = np.meshgrid(lon_cyclic, lat)
            else:
                lon_2d_cyclic, lat_2d_cyclic = lon_cyclic, lat_2d

            # Also create cyclic version of PI data for contours
            pi_smooth_cyclic, _ = add_cyclic_point(var_info['pi_smooth'], lon)

            # Choose colorbar range based on experiment
            if exp in ['mh', 'lig']:
                levels = var_info['anom_levels_warm']
            else:  # lgm, mis
                levels = var_info['anom_levels_cold']

            # Plot anomaly with contourf for smooth colors
            im = ax.contourf(lon_2d_cyclic, lat_2d_cyclic, anomaly_cyclic,
                           levels=levels,
                           cmap=var_info['cmap'],
                           transform=ccrs.PlateCarree(),
                           extend='both')

            # Store colorbar handles
            if exp in ['mh', 'lig']:
                im_warm = im
            else:
                im_cold = im

            # Overlay PI contours (smoothed)
            cs = ax.contour(lon_2d_cyclic, lat_2d_cyclic, pi_smooth_cyclic,
                           levels=var_info['pi_levels'],
                           colors='black', linewidths=0.7,
                           transform=ccrs.PlateCarree(),
                           alpha=0.5)
            ax.clabel(cs, inline=True, fontsize=7, fmt='%.1f')

            # Map settings
            ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
            ax.coastlines(linewidth=0.5, zorder=3)
            ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')

            # Panel labels (a, b, c, ...)
            ax.text(0.02, 0.98, f'({chr(97+panel_label)})',
                   transform=ax.transAxes, fontsize=11, fontweight='bold',
                   va='top', ha='left', bbox=dict(boxstyle='round',
                   facecolor='white', alpha=0.8, edgecolor='none'))
            panel_label += 1

            # Titles
            if row == 0:
                ax.set_title(f'{exp_labels[exp]}', fontsize=12, fontweight='bold')
            if col == 0:
                ax.text(-0.15, 0.5, var_info['name'],
                       transform=ax.transAxes, fontsize=12, fontweight='bold',
                       rotation=90, va='center', ha='right')

        # Add colorbars at bottom of each row - centered under MH/LIG and LGM/MIS
        # MH/LIG colorbar
        pos0 = axes_warm[0].get_position()
        pos1 = axes_warm[1].get_position()
        cbar_ax_warm = fig.add_axes([pos0.x0, pos0.y0 - 0.035,
                                     pos1.x1 - pos0.x0, 0.012])
        cbar_warm = fig.colorbar(im_warm, cax=cbar_ax_warm, orientation='horizontal')
        cbar_warm.set_label(f'Δ{var_info["name"]} ({var_info["unit"]})', fontsize=9)
        cbar_warm.ax.tick_params(labelsize=8)

        # LGM/MIS colorbar
        pos2 = axes_cold[0].get_position()
        pos3 = axes_cold[1].get_position()
        cbar_ax_cold = fig.add_axes([pos2.x0, pos2.y0 - 0.035,
                                     pos3.x1 - pos2.x0, 0.012])
        cbar_cold = fig.colorbar(im_cold, cax=cbar_ax_cold, orientation='horizontal')
        cbar_cold.set_label(f'Δ{var_info["name"]} ({var_info["unit"]})', fontsize=9)
        cbar_cold.ax.tick_params(labelsize=8)

    output_file = f'{base_dir}/fig01_climate_sst_sss_density_jja.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

def plot_figure2():
    """
    Figure 2: MLD/Sea Ice absolute values (JJA)
    Layout: 2 rows (MLD, Sea Ice) x 5 columns (all experiments)
    """
    print("Creating Figure 2: MLD/Sea Ice absolute values...")

    fig = plt.figure(figsize=(20, 9))

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 5, figure=fig, hspace=0.25, wspace=0.08,
                  left=0.06, right=0.94, top=0.92, bottom=0.1)

    vars_info = [
        {'name': 'MLD', 'var': 'MLD1', 'unit': 'm',
         'levels': np.linspace(0, 500, 21), 'cmap': 'viridis_r'},
        {'name': 'Sea Ice', 'var': 'a_ice', 'unit': 'fraction',
         'levels': np.linspace(0, 1, 21), 'cmap': 'Blues'}
    ]

    panel_label = 0
    for row, var_info in enumerate(vars_info):
        print(f"  Plotting {var_info['name']}...")

        axes_row = []
        im_row = None

        for col, exp in enumerate(exps):
            ax = fig.add_subplot(gs[row, col], projection=ccrs.SouthPolarStereo())
            axes_row.append(ax)

            # Load data
            data, lon, lat = load_variable_reg(exp, var_info['var'])
            lon_2d, lat_2d = np.meshgrid(lon, lat)

            # Fix MLD: convert negative to positive
            if var_info['var'] == 'MLD1':
                data = np.abs(data)

            # Smooth data
            data_smooth = smooth_data(data, sigma=1.5)

            # Add cyclic point to avoid longitude discontinuity
            data_cyclic, lon_cyclic = add_cyclic_point(data_smooth, lon)
            lon_2d_cyclic, lat_2d_cyclic = np.meshgrid(lon_cyclic, lat)

            # Plot with contourf for smooth colors
            im = ax.contourf(lon_2d_cyclic, lat_2d_cyclic, data_cyclic,
                            levels=var_info['levels'],
                            cmap=var_info['cmap'],
                            transform=ccrs.PlateCarree(),
                            extend='max')
            im_row = im

            # Map settings
            ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
            ax.coastlines(linewidth=0.5, zorder=3)
            ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')

            # Panel labels
            ax.text(0.02, 0.98, f'({chr(97+panel_label)})',
                   transform=ax.transAxes, fontsize=11, fontweight='bold',
                   va='top', ha='left', bbox=dict(boxstyle='round',
                   facecolor='white', alpha=0.8, edgecolor='none'))
            panel_label += 1

            # Titles
            if row == 0:
                ax.set_title(f'{exp_labels[exp]}', fontsize=12, fontweight='bold')
            if col == 0:
                ax.text(-0.12, 0.5, var_info['name'],
                       transform=ax.transAxes, fontsize=12, fontweight='bold',
                       rotation=90, va='center', ha='right')

        # Colorbar at bottom center of each row
        pos_first = axes_row[0].get_position()
        pos_last = axes_row[-1].get_position()
        cbar_width = (pos_last.x1 - pos_first.x0) * 0.6
        cbar_x0 = (pos_first.x0 + pos_last.x1 - cbar_width) / 2
        cbar_ax = fig.add_axes([cbar_x0, pos_first.y0 - 0.045, cbar_width, 0.015])
        cbar = fig.colorbar(im_row, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(f'{var_info["name"]} ({var_info["unit"]})', fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    output_file = f'{base_dir}/fig02_climate_mld_seaice_jja.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

def plot_figure3():
    """
    Figure 3: U10 wind anomalies (JJA)
    Layout: 1 row x 4 columns (paleo experiments)
    Anomalies in shading, PI wind vectors as arrows
    """
    print("Creating Figure 3: U10 wind anomalies...")

    fig = plt.figure(figsize=(16, 5.5))

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 4, figure=fig, hspace=0.1, wspace=0.08,
                  left=0.05, right=0.95, top=0.88, bottom=0.18)

    # Load PI reference
    print("  Loading PI wind data...")
    u10_pi, lon, lat = load_variable_reg('pi', 'u10')
    v10_pi, _, _ = load_variable_reg('pi', 'v10')
    windspeed_pi = np.sqrt(u10_pi**2 + v10_pi**2)

    lon_2d, lat_2d = np.meshgrid(lon, lat)

    # Subsample for wind vectors - denser sampling
    subsample = 6  # Changed from 10 to 6 for denser vectors
    lon_sub = lon[::subsample]
    lat_sub = lat[::subsample]
    lon_2d_sub, lat_2d_sub = np.meshgrid(lon_sub, lat_sub)
    u_sub = u10_pi[::subsample, ::subsample]
    v_sub = v10_pi[::subsample, ::subsample]

    axes_row = []
    im_row = None

    for col, exp in enumerate(paleo_exps):
        print(f"  Plotting {exp_labels[exp]}...")

        ax = fig.add_subplot(gs[0, col], projection=ccrs.SouthPolarStereo())
        axes_row.append(ax)

        # Load paleo wind data
        u10_exp, _, _ = load_variable_reg(exp, 'u10')
        v10_exp, _, _ = load_variable_reg(exp, 'v10')
        windspeed_exp = np.sqrt(u10_exp**2 + v10_exp**2)

        # Calculate anomaly and smooth
        windspeed_anom = windspeed_exp - windspeed_pi
        windspeed_anom_smooth = smooth_data(windspeed_anom, sigma=1.5)

        # Add cyclic point to avoid longitude discontinuity
        windspeed_anom_cyclic, lon_cyclic = add_cyclic_point(windspeed_anom_smooth, lon)
        lon_2d_cyclic, lat_2d_cyclic = np.meshgrid(lon_cyclic, lat)

        # Plot wind speed anomaly with contourf
        levels = np.linspace(-4, 4, 21)
        im = ax.contourf(lon_2d_cyclic, lat_2d_cyclic, windspeed_anom_cyclic,
                        levels=levels,
                        cmap='RdBu_r',
                        transform=ccrs.PlateCarree(),
                        extend='both')
        im_row = im

        # Overlay PI wind vectors - larger and denser
        q = ax.quiver(lon_2d_sub, lat_2d_sub, u_sub, v_sub,
                     transform=ccrs.PlateCarree(),
                     scale=100, width=0.004, alpha=0.7, color='black',
                     headwidth=4, headlength=5)

        # Map settings
        ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
        ax.coastlines(linewidth=0.5, zorder=3)
        ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')

        # Panel labels
        ax.text(0.02, 0.98, f'({chr(97+col)})',
               transform=ax.transAxes, fontsize=11, fontweight='bold',
               va='top', ha='left', bbox=dict(boxstyle='round',
               facecolor='white', alpha=0.8, edgecolor='none'))

        # Title
        ax.set_title(f'{exp_labels[exp]}', fontsize=12, fontweight='bold')

    # Add quiver key
    ax_last = axes_row[-1]
    ax_last.quiverkey(q, 0.95, 0.02, 10, '10 m/s', labelpos='W',
                      coordinates='axes', fontproperties={'size': 9})

    # Colorbar at bottom center of row
    pos_first = axes_row[0].get_position()
    pos_last = axes_row[-1].get_position()
    cbar_width = (pos_last.x1 - pos_first.x0) * 0.6
    cbar_x0 = (pos_first.x0 + pos_last.x1 - cbar_width) / 2
    cbar_ax = fig.add_axes([cbar_x0, 0.08, cbar_width, 0.025])
    cbar = fig.colorbar(im_row, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('ΔWind Speed (m/s)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    output_file = f'{base_dir}/fig03_climate_winds_jja.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

if __name__ == '__main__':
    print("=" * 60)
    print("Plotting Climate Patterns (JJA) for AABW Paper")
    print("=" * 60)

    plot_figure1()
    plot_figure2()
    plot_figure3()

    print("=" * 60)
    print("All figures completed successfully!")
    print("=" * 60)
