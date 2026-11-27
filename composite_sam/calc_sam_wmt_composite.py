#!/usr/bin/env python
"""
Calculate SAM index and perform composite analysis with WMT data
- SAM: Southern Annular Mode from SLP data (Marshall index method)
- Composite: High SAM vs Low SAM (threshold = 1.2 * std)
- Season: JJA (austral winter, AABW formation period)
"""

import numpy as np
import xarray as xr
from pathlib import Path
import sys

print('='*80)
print('SAM Index Calculation and WMT Composite Analysis')
print('='*80)
print()

# Configuration
BASE_PATH = Path('/work/ba1066/a270064/cc_projects/aabw_5exps')
OUTPUT_PATH = Path('/work/ba1066/a270064/cc_projects/aabw_5exps/composite_sam')

EXPERIMENTS = ['pi', 'mh', 'lig', 'lgm', 'mis']
REGIONS = ['southern_ocean', 'ross_sea', 'weddell_sea', 'adelie']

# SAM definition (Marshall, 2003)
# SAM = normalized [SLP(40°S) - SLP(65°S)]
SAM_LAT_40S = -40.0
SAM_LAT_65S = -65.0

# Composite threshold
THRESHOLD_STD = 1.2

# JJA months (austral winter)
JJA_MONTHS = [6, 7, 8]


def calculate_sam_index(slp_file):
    """
    Calculate SAM index using Marshall (2003) method

    SAM = normalized [mean_SLP(40°S) - mean_SLP(65°S)]

    Parameters:
    -----------
    slp_file : Path
        Path to SLP data file (monthly, 100 years)

    Returns:
    --------
    sam_monthly : xr.DataArray
        Monthly SAM index (1200 values)
    sam_jja : xr.DataArray
        JJA seasonal mean SAM index (100 values)
    """

    print(f'\n{"="*80}')
    print(f'Calculating SAM index from: {slp_file.name}')
    print(f'{"="*80}')

    # Load SLP data and decode time
    ds = xr.open_dataset(slp_file, decode_times=False)
    slp = ds['psl']  # Pa

    # Parse time coordinate (format: YYYYMMDD.fraction)
    time_values = ds.time.values
    years = (time_values // 10000).astype(int)
    months = ((time_values % 10000) // 100).astype(int)

    print(f'SLP shape: {slp.shape}')
    print(f'Time range: {len(slp.time)} months')
    print(f'Year range: {years[0]} to {years[-1]}')
    print(f'Lat range: {slp.lat.min().values:.2f} to {slp.lat.max().values:.2f}')
    print(f'Lon range: {slp.lon.min().values:.2f} to {slp.lon.max().values:.2f}')

    # Find nearest latitudes
    lat_40s_idx = np.abs(slp.lat.values - SAM_LAT_40S).argmin()
    lat_65s_idx = np.abs(slp.lat.values - SAM_LAT_65S).argmin()

    lat_40s_actual = slp.lat.values[lat_40s_idx]
    lat_65s_actual = slp.lat.values[lat_65s_idx]

    print(f'\nSAM latitude bands:')
    print(f'  40°S: using {lat_40s_actual:.2f}°')
    print(f'  65°S: using {lat_65s_actual:.2f}°')

    # Extract zonal mean SLP at each latitude
    slp_40s = slp.isel(lat=lat_40s_idx).mean(dim='lon')  # Zonal mean at 40°S
    slp_65s = slp.isel(lat=lat_65s_idx).mean(dim='lon')  # Zonal mean at 65°S

    # Calculate raw SAM index (monthly)
    sam_raw = slp_40s - slp_65s

    print(f'\nRaw SAM statistics (monthly):')
    print(f'  Mean: {sam_raw.mean().values:.2f} Pa')
    print(f'  Std:  {sam_raw.std().values:.2f} Pa')
    print(f'  Min:  {sam_raw.min().values:.2f} Pa')
    print(f'  Max:  {sam_raw.max().values:.2f} Pa')

    # Normalize SAM index
    sam_monthly = (sam_raw - sam_raw.mean()) / sam_raw.std()

    print(f'\nNormalized SAM statistics (monthly):')
    print(f'  Mean: {sam_monthly.mean().values:.6f} (should be ~0)')
    print(f'  Std:  {sam_monthly.std().values:.6f} (should be ~1)')
    print(f'  Min:  {sam_monthly.min().values:.2f}')
    print(f'  Max:  {sam_monthly.max().values:.2f}')

    # Calculate JJA seasonal means
    # Add month and year as coordinates
    sam_monthly = sam_monthly.assign_coords(
        month=('time', months),
        year=('time', years)
    )

    # Select JJA months
    sam_jja_months = sam_monthly.where(
        sam_monthly.month.isin(JJA_MONTHS),
        drop=True
    )

    # Group by year and calculate mean
    sam_jja = sam_jja_months.groupby('year').mean()

    print(f'\nJJA SAM statistics (annual):')
    print(f'  Years: {len(sam_jja.year)}')
    print(f'  Mean: {sam_jja.mean().values:.6f} (should be ~0)')
    print(f'  Std:  {sam_jja.std().values:.2f}')
    print(f'  Min:  {sam_jja.min().values:.2f}')
    print(f'  Max:  {sam_jja.max().values:.2f}')

    ds.close()

    return sam_monthly, sam_jja


def identify_composite_years(sam_jja, threshold_std=1.2):
    """
    Identify high/low SAM years based on threshold

    Parameters:
    -----------
    sam_jja : xr.DataArray
        JJA seasonal mean SAM index
    threshold_std : float
        Threshold in standard deviations (default: 1.2)

    Returns:
    --------
    high_sam_years : np.ndarray
        Years with high SAM (> mean + threshold*std)
    low_sam_years : np.ndarray
        Years with low SAM (< mean - threshold*std)
    """

    mean_sam = sam_jja.mean().values
    std_sam = sam_jja.std().values

    threshold_high = mean_sam + threshold_std * std_sam
    threshold_low = mean_sam - threshold_std * std_sam

    high_sam_years = sam_jja.year.values[sam_jja.values > threshold_high]
    low_sam_years = sam_jja.year.values[sam_jja.values < threshold_low]

    print(f'\n{"="*80}')
    print(f'Composite Year Selection (threshold = {threshold_std:.1f}σ)')
    print(f'{"="*80}')
    print(f'SAM mean: {mean_sam:.6f}')
    print(f'SAM std:  {std_sam:.2f}')
    print(f'High SAM threshold: > {threshold_high:+.2f}')
    print(f'Low SAM threshold:  < {threshold_low:+.2f}')
    print(f'\nHigh SAM years ({len(high_sam_years)}): {high_sam_years}')
    print(f'Low SAM years ({len(low_sam_years)}):  {low_sam_years}')

    return high_sam_years, low_sam_years


def load_wmt_jja_data(wmt_file, jja_years):
    """
    Load WMT data and calculate JJA means for specified years

    Parameters:
    -----------
    wmt_file : Path
        Path to 100-year WMT file
    jja_years : np.ndarray
        Years to extract (e.g., [2301, 2305, ...])

    Returns:
    --------
    wmt_jja : xr.Dataset
        JJA mean WMT data for specified years
    """

    # Load with proper time decoding (WMT files use "seconds since YYYY-01-01")
    ds = xr.open_dataset(wmt_file, decode_times=True)

    # Extract month and year from datetime
    months = ds.time.dt.month.values
    years = ds.time.dt.year.values

    # Add coordinates
    ds = ds.assign_coords(
        month=('time', months),
        year=('time', years)
    )

    # Create mask for JJA months
    jja_mask = np.isin(months, JJA_MONTHS)

    # Create mask for selected years
    year_mask = np.isin(years, jja_years)

    # Combined mask: JJA months AND selected years
    combined_mask = jja_mask & year_mask

    # Select data
    ds_jja_years = ds.isel(time=combined_mask)

    # Group by year and calculate mean
    wmt_jja = ds_jja_years.groupby('year').mean()

    ds.close()

    return wmt_jja


def calculate_wmt_composite(exp_key, region_key, high_sam_years, low_sam_years):
    """
    Calculate WMT composite for high vs low SAM years

    Parameters:
    -----------
    exp_key : str
        Experiment name (e.g., 'pi')
    region_key : str
        Region name (e.g., 'southern_ocean')
    high_sam_years : np.ndarray
        High SAM years
    low_sam_years : np.ndarray
        Low SAM years

    Returns:
    --------
    composite_ds : xr.Dataset
        Dataset with high_sam, low_sam, and diff composites
    """

    # 2026-11-27 (user GN): sigma2 bins 0.1 wide; the 0.2-bin inputs stay in wmt_results/, old outputs in versions_bin0p2/
    wmt_file = BASE_PATH / exp_key / 'wmt_results_bin0p1' / f'wmt_{region_key}_100years_{exp_key}.nc'

    if not wmt_file.exists():
        print(f'  ⚠ WMT file not found: {wmt_file}')
        return None

    print(f'\n  Loading: {wmt_file.name}')

    # Load WMT data for high and low SAM years
    wmt_high = load_wmt_jja_data(wmt_file, high_sam_years)
    wmt_low = load_wmt_jja_data(wmt_file, low_sam_years)

    print(f'  High SAM: {len(wmt_high.year)} years')
    print(f'  Low SAM:  {len(wmt_low.year)} years')

    # Calculate composites (mean across years)
    high_sam_mean = wmt_high.mean(dim='year')
    low_sam_mean = wmt_low.mean(dim='year')
    diff = high_sam_mean - low_sam_mean

    # Create composite dataset
    composite_ds = xr.Dataset()

    # Copy sigma2 coordinate
    composite_ds['sigma2'] = high_sam_mean['sigma2_l_target']

    # Store composites for each variable
    for var in high_sam_mean.data_vars:
        if var != 'sigma2_l_target':
            composite_ds[f'{var}_high_sam'] = high_sam_mean[var]
            composite_ds[f'{var}_low_sam'] = low_sam_mean[var]
            composite_ds[f'{var}_diff'] = diff[var]

    # Add metadata
    composite_ds.attrs['experiment'] = exp_key
    composite_ds.attrs['region'] = region_key
    composite_ds.attrs['n_high_sam_years'] = len(high_sam_years)
    composite_ds.attrs['n_low_sam_years'] = len(low_sam_years)
    composite_ds.attrs['high_sam_years'] = ','.join(map(str, high_sam_years))
    composite_ds.attrs['low_sam_years'] = ','.join(map(str, low_sam_years))
    composite_ds.attrs['threshold_std'] = THRESHOLD_STD
    composite_ds.attrs['season'] = 'JJA'
    composite_ds.attrs['description'] = 'WMT composite: High SAM - Low SAM (JJA mean)'

    return composite_ds


def main():
    """Main processing loop"""

    OUTPUT_PATH.mkdir(exist_ok=True)

    for exp_key in EXPERIMENTS:

        print(f'\n{"#"*80}')
        print(f'# Processing experiment: {exp_key.upper()}')
        print(f'{"#"*80}')

        # Step 1: Calculate SAM index
        slp_file = BASE_PATH / exp_key / '100years' / 'slp_mergetime.nc'

        if not slp_file.exists():
            print(f'\n⚠ SLP file not found: {slp_file}')
            print(f'   Skipping {exp_key}...\n')
            continue

        sam_monthly, sam_jja = calculate_sam_index(slp_file)

        # Step 2: Identify composite years
        high_sam_years, low_sam_years = identify_composite_years(
            sam_jja,
            threshold_std=THRESHOLD_STD
        )

        # Save SAM index
        sam_output_file = OUTPUT_PATH / f'sam_index_{exp_key}.nc'
        sam_ds = xr.Dataset({
            'sam_monthly': sam_monthly,
            'sam_jja': sam_jja
        })
        sam_ds.attrs['experiment'] = exp_key
        sam_ds.attrs['sam_lat_40s'] = SAM_LAT_40S
        sam_ds.attrs['sam_lat_65s'] = SAM_LAT_65S
        sam_ds.attrs['threshold_std'] = THRESHOLD_STD
        sam_ds.attrs['n_high_sam_years'] = len(high_sam_years)
        sam_ds.attrs['n_low_sam_years'] = len(low_sam_years)
        sam_ds.attrs['description'] = 'SAM index (Marshall method): normalized [SLP(40S) - SLP(65S)]'

        sam_ds.to_netcdf(sam_output_file)
        print(f'\n✓ Saved SAM index: {sam_output_file}')

        # Step 3: Calculate WMT composites for each region
        print(f'\n{"="*80}')
        print(f'Calculating WMT composites for {exp_key.upper()}')
        print(f'{"="*80}')

        for region_key in REGIONS:
            print(f'\nRegion: {region_key}')

            composite_ds = calculate_wmt_composite(
                exp_key, region_key, high_sam_years, low_sam_years
            )

            if composite_ds is not None:
                # Save composite
                output_file = OUTPUT_PATH / f'wmt_sam_composite_{region_key}_{exp_key}.nc'
                composite_ds.to_netcdf(output_file)
                print(f'  ✓ Saved: {output_file.name}')

        sam_ds.close()

    print(f'\n{"="*80}')
    print('All processing complete!')
    print(f'{"="*80}')
    print(f'\nOutput files saved to: {OUTPUT_PATH}')
    print(f'  - SAM indices: sam_index_{{exp}}.nc (5 files)')
    print(f'  - WMT composites: wmt_sam_composite_{{region}}_{{exp}}.nc (20 files)')
    print()


if __name__ == '__main__':
    main()
