#!/usr/bin/env python
"""
Plot SAM-WMT composite analysis results
- SAM time series for all 5 experiments
- WMT composites: 4 regions x 5 experiments (similar to plot_wmt_4regions_5exps.py)
- Components: Total WMT, Heat, Sea Ice FW, Other FW
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ============================================================================
# Nature-style publication aesthetics (matches plot_wmt_4regions_5exps_from_100years.py)
# ============================================================================
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 16,
    'axes.linewidth': 1.2,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#222222',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Refined Nature-style color palette
COL_TOTAL   = '#111111'   # near-black
COL_HEAT    = '#D7263D'   # crimson
COL_SEAICE  = '#1B998B'   # teal
COL_OTHERFW = '#2E86AB'   # ocean blue

print('='*80)
print('Plotting SAM-WMT Composite Analysis')
print('='*80)
print()

# Configuration
DATA_PATH = Path('/work/ba1066/a270064/cc_projects/aabw_5exps/composite_sam')
OUTPUT_PATH = DATA_PATH

EXPERIMENTS = [
    ('pi', 'PI'),
    ('mh', 'MH'),
    ('lig', 'LIG'),
    ('lgm', 'LGM'),
    ('mis', 'MIS')
]

REGIONS = [
    ('southern_ocean', 'Southern Ocean'),
    ('ross_sea', 'Ross Sea'),
    ('weddell_sea', 'Weddell Sea'),
    ('adelie', 'Adélie Land')
]


def plot_sam_timeseries():
    """
    Plot SAM JJA time series for all 5 experiments
    """

    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)

    print('\n' + '='*80)
    print('Plotting SAM time series')
    print('='*80)

    for idx, (exp_key, exp_title) in enumerate(EXPERIMENTS):
        ax = axes[idx]

        sam_file = DATA_PATH / f'sam_index_{exp_key}.nc'

        if not sam_file.exists():
            print(f'\n⚠ SAM file not found: {sam_file.name}')
            ax.text(0.5, 0.5, 'Data Not Available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel(exp_title, fontsize=12, fontweight='bold')
            continue

        ds = xr.open_dataset(sam_file)
        sam_jja = ds['sam_jja']

        n_high = ds.attrs['n_high_sam_years']
        n_low = ds.attrs['n_low_sam_years']
        threshold_std = ds.attrs['threshold_std']

        print(f'\n{exp_title}:')
        print(f'  Mean: {sam_jja.mean().values:.6f}')
        print(f'  Std:  {sam_jja.std().values:.2f}')
        print(f'  High SAM events: {n_high}')
        print(f'  Low SAM events:  {n_low}')

        # Plot time series
        years = sam_jja.year.values
        sam_values = sam_jja.values

        # Calculate actual mean and std
        sam_mean = sam_jja.mean().values
        sam_std = sam_jja.std().values

        ax.plot(years, sam_values, 'k-', linewidth=1.5, label='SAM JJA')
        ax.axhline(y=sam_mean, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        # Threshold lines (correct: mean ± std)
        threshold_high = sam_mean + threshold_std * sam_std
        threshold_low = sam_mean - threshold_std * sam_std

        ax.axhline(y=threshold_high, color='red', linestyle='--',
                  linewidth=1.5, alpha=0.7, label=f'Mean+{threshold_std}σ')
        ax.axhline(y=threshold_low, color='blue', linestyle='--',
                  linewidth=1.5, alpha=0.7, label=f'Mean-{threshold_std}σ')

        # Highlight high/low SAM years
        high_mask = sam_values > threshold_high
        low_mask = sam_values < threshold_low

        ax.scatter(years[high_mask], sam_values[high_mask],
                  color='red', s=50, zorder=5, label=f'High SAM (n={n_high})')
        ax.scatter(years[low_mask], sam_values[low_mask],
                  color='blue', s=50, zorder=5, label=f'Low SAM (n={n_low})')

        # Labels
        ax.set_ylabel(f'{exp_title}\nSAM Index', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, ncol=2)

        ds.close()

    axes[-1].set_xlabel('Year', fontsize=12)
    plt.suptitle('SAM Index (JJA) - 100-year Time Series\n' +
                 'Southern Annular Mode: Normalized [SLP(40°S) - SLP(65°S)]',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_file = OUTPUT_PATH / 'sam_timeseries_all_exps.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'\n✓ Saved: {output_file}')
    return output_file


def plot_wmt_composite_4regions():
    """
    Plot WMT composite (High SAM - Low SAM) for 4 regions x 5 experiments
    (Supplementary Fig. A6).

    2026-11-27: WMT in 0.1 sigma_2 bins (user 12:04; old 0.2-bin figure in
    versions/*_v20261127bin0p2.*). Drawn at the manuscript text width (5.15 in), so
    every size below is the printed size in pt (A4 readability standard). Data, colours,
    4 x 5 layout and axis limits as before; one y-axis title for the whole figure, y tick
    labels on the left column only (each row shares its limits), region names in two
    lines, x tick labels at whole sigma_2 values, one x-axis title under the bottom row.
    """
    FW = 5.15                                   # in, printed width
    FS_TICK, FS_LAB, FS_TITLE, FS_LETTER = 7.5, 8, 9, 9
    K = 'k'
    left_in, right_in, wgap_in = 0.80, 0.05, 0.10
    top_in, hgap_in, bot_in, ph_in = 0.24, 0.10, 0.70, 0.72
    pw_in = (FW - left_in - right_in - 4 * wgap_in) / 5
    FH = top_in + 4 * ph_in + 3 * hgap_in + bot_in
    fig = plt.figure(figsize=(FW, FH))
    axes = np.empty((4, 5), dtype=object)
    for r in range(4):
        for c in range(5):
            axes[r, c] = fig.add_axes([(left_in + c * (pw_in + wgap_in)) / FW,
                                       (bot_in + (3 - r) * (ph_in + hgap_in)) / FH,
                                       pw_in / FW, ph_in / FH])

    panel_labels = [chr(ord('a') + i) for i in range(20)]
    legend_handles = None

    print('\n' + '='*80)
    print('Plotting WMT Composite: High SAM - Low SAM')
    print('='*80)

    # First pass: collect all data to determine per-row y-axis limits
    all_data = {}  # (row_idx, col_idx) -> dict of arrays
    for row_idx, (region_key, region_title) in enumerate(REGIONS):
        for col_idx, (exp_key, exp_title) in enumerate(EXPERIMENTS):
            composite_file = DATA_PATH / f'wmt_sam_composite_{region_key}_{exp_key}.nc'
            if not composite_file.exists():
                continue
            ds = xr.open_dataset(composite_file)
            total_heat_diff = -ds['total_heat_surface_exchange_flux_nonadvective_heat_diff'] / 1e9
            evap_salt_diff = -ds['surface_ocean_flux_advective_negative_rhs_evaporation_salt_diff'] / 1e9
            snow_salt_diff = -ds['surface_ocean_flux_advective_negative_rhs_snow_salt_diff'] / 1e9
            seaice_salt_diff = -ds['surface_ocean_flux_advective_negative_rhs_sea_ice_melt_salt_diff'] / 1e9
            rain_salt_diff = -ds['surface_ocean_flux_advective_negative_rhs_rain_and_ice_salt_diff'] / 1e9
            river_salt_diff = -ds['surface_ocean_flux_advective_negative_rhs_rivers_salt_diff'] / 1e9
            seaice_fw_diff = seaice_salt_diff
            other_fw_diff = evap_salt_diff + snow_salt_diff + rain_salt_diff + river_salt_diff
            total_wmt_diff = total_heat_diff + seaice_fw_diff + other_fw_diff
            all_data[(row_idx, col_idx)] = {
                'sigma2': ds['sigma2'].values,
                'total_wmt': total_wmt_diff.values,
                'heat': total_heat_diff.values,
                'seaice_fw': seaice_fw_diff.values,
                'other_fw': other_fw_diff.values,
            }
            ds.close()

    # Compute per-row y limits
    row_ylims = {}
    for row_idx in range(4):
        all_vals = []
        for col_idx in range(5):
            if (row_idx, col_idx) in all_data:
                d = all_data[(row_idx, col_idx)]
                for k in ['total_wmt', 'heat', 'seaice_fw', 'other_fw']:
                    all_vals.extend(d[k].tolist())
        if all_vals:
            vmax = max(abs(v) for v in all_vals if np.isfinite(v))
            vmax = vmax * 1.15  # 15% margin
            row_ylims[row_idx] = (-vmax, vmax)
        else:
            row_ylims[row_idx] = (-1, 1)

    # Second pass: plot
    for row_idx, (region_key, region_title) in enumerate(REGIONS):
        for col_idx, (exp_key, exp_title) in enumerate(EXPERIMENTS):
            ax = axes[row_idx, col_idx]
            for sp in ('left', 'bottom'):
                ax.spines[sp].set_linewidth(0.6)
                ax.spines[sp].set_color(K)

            if (row_idx, col_idx) not in all_data:
                print(f'\n⚠ Composite file not found for {region_key} {exp_key}')
                ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center',
                        fontsize=FS_LAB, color=K, transform=ax.transAxes)
                continue

            d = all_data[(row_idx, col_idx)]
            sigma2 = d['sigma2']

            print(f'\n{region_title} - {exp_title}:')

            l_total, = ax.plot(sigma2, d['total_wmt'], color=COL_TOTAL, linestyle='-',
                               linewidth=1.3, label='Total WMT', solid_capstyle='round',
                               zorder=10)
            l_heat, = ax.plot(sigma2, d['heat'], color=COL_HEAT, linestyle='-',
                              linewidth=1.1, label='Heat', solid_capstyle='round', zorder=8)
            l_si, = ax.plot(sigma2, d['seaice_fw'], color=COL_SEAICE, linestyle='-',
                            linewidth=1.0, label='Sea-ice FW', solid_capstyle='round',
                            zorder=6)
            l_oth, = ax.plot(sigma2, d['other_fw'], color=COL_OTHERFW, linestyle='-',
                             linewidth=1.0, label='Other FW', solid_capstyle='round',
                             zorder=6)
            if legend_handles is None:
                legend_handles = [l_total, l_heat, l_si, l_oth]

            # X axis: limits as before, labels at whole sigma_2 values, bottom row only
            if exp_key in ['pi', 'mh', 'lig']:
                ax.set_xlim(35.5, 37.2)
                ax.set_xticks([36.0, 37.0])
                ax.set_xticks([35.5, 36.5], minor=True)
            else:  # lgm, mis
                ax.set_xlim(36.0, 38.5)
                ax.set_xticks([37.0, 38.0])          # 36 at the left edge would touch the
                ax.set_xticks([36.0, 36.5, 37.5], minor=True)   # 37 of the LIG panel
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%g'))
            if row_idx != len(REGIONS) - 1:
                ax.tick_params(axis='x', labelbottom=False)

            # Y axis: unified per row, tick labels on the left column only
            ax.set_ylim(row_ylims[row_idx])
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10],
                                                              symmetric=True))
            if col_idx != 0:
                ax.tick_params(axis='y', labelleft=False)

            if row_idx == 0:
                ax.set_title(f'{exp_title}', fontsize=FS_TITLE, fontweight='bold',
                             pad=4, color=K)

            # Region name left of the tick labels and of the shared y-axis title
            if col_idx == 0:
                ax.annotate(region_title.replace(' ', '\n'),
                            xy=(0, 0.5), xycoords='axes fraction',
                            xytext=(-38, 0), textcoords='offset points',
                            rotation=90, ha='right', va='center',
                            fontsize=FS_LAB + 0.5, fontweight='bold', color=K,
                            linespacing=1.05, annotation_clip=False)

            panel_idx = row_idx * 5 + col_idx
            ax.text(0.04, 0.96, f'({panel_labels[panel_idx]})',
                    transform=ax.transAxes, fontsize=FS_LETTER, fontweight='bold',
                    ha='left', va='top', color=K)

            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.4, color='#999999')
            ax.set_axisbelow(True)
            ax.axhline(y=0, color='#444444', linestyle='-', linewidth=0.5, zorder=1)
            ax.tick_params(axis='both', which='major', labelsize=FS_TICK, pad=2,
                           length=2.5, width=0.6, color=K, labelcolor=K)
            ax.tick_params(axis='both', which='minor', length=1.5, width=0.5, color=K)

    # One x-axis title under the bottom row and one y-axis title for all rows
    fig.text((left_in + (FW - left_in - right_in) / 2) / FW, (bot_in - 0.20) / FH,
             r'$\sigma_2$  (kg m$^{-3}$)', ha='center', va='top', fontsize=FS_LAB,
             fontweight='bold', color=K)
    y_mid = (bot_in + (4 * ph_in + 3 * hgap_in) / 2) / FH
    fig.text((left_in - 0.36) / FW, y_mid, 'WMT anomaly  (Sv)', rotation=90,
             ha='center', va='center', fontsize=FS_LAB, fontweight='bold', color=K)

    if legend_handles is not None:
        leg = fig.legend(handles=legend_handles, loc='lower center',
                         bbox_to_anchor=((left_in + (FW - left_in - right_in) / 2) / FW, 0.03 / FH),
                         ncol=4, frameon=True, fancybox=False, edgecolor=K,
                         facecolor='white', framealpha=1.0, fontsize=FS_LAB,
                         handlelength=2.2, handletextpad=0.6, columnspacing=1.6,
                         borderpad=0.5, labelcolor=K)
        leg.get_frame().set_linewidth(0.6)

    output_file = OUTPUT_PATH / 'wmt_sam_composite_4regions_5exps.pdf'
    plt.savefig(output_file, dpi=600)
    png_file = output_file.with_suffix('.png')
    plt.savefig(png_file, dpi=600)
    plt.close()

    print(f'\n✓ Saved: {output_file}')
    print(f'✓ Saved: {png_file}')
    return output_file


def plot_wmt_composite_5exps_per_variable():
    """
    Plot WMT composite for each variable: 5 experiments x 4 regions
    Generate 3 figures: Total WMT, Heat, Sea Ice FW
    """

    variables = [
        ('total', 'Total WMT'),
        ('heat', 'Heat Flux'),
        ('seaice_fw', 'Sea Ice Freshwater')
    ]

    output_files = []

    for var_key, var_title in variables:

        fig, axes = plt.subplots(4, 5, figsize=(18, 14))

        print(f'\n{"="*80}')
        print(f'Plotting {var_title} composite')
        print(f'{"="*80}')

        for row_idx, (region_key, region_title) in enumerate(REGIONS):
            for col_idx, (exp_key, exp_title) in enumerate(EXPERIMENTS):
                ax = axes[row_idx, col_idx]

                composite_file = DATA_PATH / f'wmt_sam_composite_{region_key}_{exp_key}.nc'

                if not composite_file.exists():
                    ax.text(0.5, 0.5, 'Data Not Available',
                           ha='center', va='center', fontsize=10, transform=ax.transAxes)
                    ax.set_xlabel(r'$\sigma_2$ [kg/m$^3$]', fontsize=10)
                    ax.set_ylabel('WMT Anomaly [Sv]', fontsize=10)
                    if row_idx == 0:
                        ax.set_title(f'{exp_title}', fontsize=12, fontweight='bold')
                    if col_idx == 0:
                        ax.text(-0.15, 0.5, region_title, rotation=90, va='center',
                               fontsize=13, fontweight='bold', transform=ax.transAxes)
                    ax.grid(True, alpha=0.3)
                    continue

                ds = xr.open_dataset(composite_file)
                sigma2 = ds['sigma2'].values

                # Extract data based on variable
                if var_key == 'total':
                    # Calculate total WMT
                    total_heat_diff = -ds['total_heat_surface_exchange_flux_nonadvective_heat_diff'] / 1e9
                    evap_salt_diff = -ds['surface_ocean_flux_advective_negative_rhs_evaporation_salt_diff'] / 1e9
                    snow_salt_diff = -ds['surface_ocean_flux_advective_negative_rhs_snow_salt_diff'] / 1e9
                    seaice_salt_diff = -ds['surface_ocean_flux_advective_negative_rhs_sea_ice_melt_salt_diff'] / 1e9
                    rain_salt_diff = -ds['surface_ocean_flux_advective_negative_rhs_rain_and_ice_salt_diff'] / 1e9
                    river_salt_diff = -ds['surface_ocean_flux_advective_negative_rhs_rivers_salt_diff'] / 1e9
                    data_diff = total_heat_diff + evap_salt_diff + snow_salt_diff + seaice_salt_diff + rain_salt_diff + river_salt_diff

                elif var_key == 'heat':
                    data_diff = -ds['total_heat_surface_exchange_flux_nonadvective_heat_diff'] / 1e9

                elif var_key == 'seaice_fw':
                    data_diff = -ds['surface_ocean_flux_advective_negative_rhs_sea_ice_melt_salt_diff'] / 1e9

                # Plot
                ax.plot(sigma2, data_diff.values,
                       color='k', linestyle='-', linewidth=2,
                       label=f'High - Low SAM')

                ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

                # Axes settings
                ax.set_xlabel(r'$\sigma_2$ [kg/m$^3$]', fontsize=10)
                ax.set_ylabel('WMT Anomaly [Sv]', fontsize=10)

                if row_idx == 0:
                    ax.set_title(f'{exp_title}', fontsize=12, fontweight='bold')

                if col_idx == 0:
                    ax.text(-0.15, 0.5, region_title, rotation=90, va='center',
                           fontsize=13, fontweight='bold', transform=ax.transAxes)

                ax.grid(True, alpha=0.3)

                # X-axis limits
                if exp_key in ['pi', 'mh', 'lig']:
                    ax.set_xlim(35.5, 37.2)
                else:
                    ax.set_xlim(36.0, 38.5)

                ds.close()

        plt.suptitle(f'{var_title} Composite: High SAM - Low SAM (JJA)\n'
                     f'4 Regions × 5 Experiments',
                     fontsize=18, fontweight='bold', y=0.998)

        plt.tight_layout(rect=[0, 0, 1, 0.995])

        output_file = OUTPUT_PATH / f'wmt_sam_composite_5exps_{var_key}.pdf'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f'\n✓ Saved: {output_file}')
        output_files.append(output_file)

    return output_files


def main():
    """Main plotting routine"""

    print('='*80)
    print('SAM-WMT Composite Plotting')
    print('='*80)
    print()

    # 1. SAM time series
    sam_file = plot_sam_timeseries()

    # 2. WMT composite: 4 regions x 5 experiments
    composite_file = plot_wmt_composite_4regions()

    # 3. WMT composite by variable
    variable_files = plot_wmt_composite_5exps_per_variable()

    print('\n' + '='*80)
    print('All plots complete!')
    print('='*80)
    print(f'\nGenerated files:')
    print(f'  1. SAM time series: {sam_file.name}')
    print(f'  2. WMT composite (4 regions): {composite_file.name}')
    print(f'  3. Variable composites:')
    for vf in variable_files:
        print(f'     - {vf.name}')
    print()


if __name__ == '__main__':
    import sys
    if '--a6' in sys.argv:          # only Supplementary Fig. A6; main() also redraws three others
        plot_wmt_composite_4regions()
    else:
        main()
