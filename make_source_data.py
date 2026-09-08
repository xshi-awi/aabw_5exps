#!/usr/bin/env python
"""
Build the Source Data workbook required by Nature Communications, containing the
underlying values for every line-graph figure in the manuscript.

Sheets:
  Fig2_WMT_winter_<region>   winter transformation curves, 4 regions x 5 experiments
  Fig5_SAM_WMT_<region>      SAM composite transformation anomalies
  Fig7_age_profiles          basin-mean ideal age profiles
  Fig7_age_below2000m        basin-mean ideal age below 2000 m
  Fig8_domain_sensitivity    transformation over the two integration domains
  Fig6_MOC_cell_strength     abyssal and upper cell strengths
"""
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
from pathlib import Path

EXPS = ['pi', 'mh', 'lig', 'lgm', 'mis']
LABELS = ['PI', 'MH', 'LIG', 'LGM', 'MIS3']
REGIONS = ['southern_ocean', 'ross_sea', 'weddell_sea', 'adelie']
REGION_LABELS = ['Southern Ocean', 'Ross Sea', 'Weddell Sea', 'Adelie Land']
BASE = Path('.')
OUT = Path('nc_paper/SUBMISSION/Source_Data.xlsx')

sheets = {}


def wmt_components(ds, jja=False):
    """Return sigma2, total, heat, seaice, other in Sv."""
    sig = ds['sigma2_l_target'].values
    if jja:
        mon = ds['time.month']
        sel = mon.isin([6, 7, 8])
        heat = -ds['total_heat_surface_exchange_flux_nonadvective_heat'].where(sel, drop=True).mean('time').values / 1e9

        def g(n):
            k = f'surface_ocean_flux_advective_negative_rhs_{n}_salt'
            return -ds[k].where(sel, drop=True).mean('time').values / 1e9
    else:
        heat = -ds['total_heat_surface_exchange_flux_nonadvective_heat'].mean('time').values / 1e9

        def g(n):
            k = f'surface_ocean_flux_advective_negative_rhs_{n}_salt'
            return -ds[k].mean('time').values / 1e9

    ice = g('sea_ice_melt')
    oth = g('evaporation') + g('snow') + g('rain_and_ice') + g('rivers')
    return sig, heat + ice + oth, heat, ice, oth


# ------------------------------------------------------- Fig 2: winter WMT
for reg, reglab in zip(REGIONS, REGION_LABELS):
    frames = []
    for e, lab in zip(EXPS, LABELS):
        f = BASE / e / 'wmt_results' / f'wmt_{reg}_100years_{e}.nc'
        if not f.exists():
            continue
        ds = xr.open_dataset(f)
        sig, tot, heat, ice, oth = wmt_components(ds, jja=True)
        frames.append(pd.DataFrame({
            'sigma2_kg_m-3': sig,
            f'{lab}_total_Sv': tot,
            f'{lab}_heat_Sv': heat,
            f'{lab}_seaice_FW_Sv': ice,
            f'{lab}_other_FW_Sv': oth,
        }).set_index('sigma2_kg_m-3'))
        ds.close()
    if frames:
        sheets[f'Fig2_WMT_JJA_{reglab[:12].replace(" ", "_")}'] = pd.concat(frames, axis=1).reset_index()

# ------------------------------------------------------- Fig 5: SAM composites
for reg, reglab in zip(REGIONS, REGION_LABELS):
    frames = []
    for e, lab in zip(EXPS, LABELS):
        f = BASE / 'composite_sam' / f'wmt_sam_composite_{reg}_{e}.nc'
        if not f.exists():
            continue
        ds = xr.open_dataset(f)
        sig = ds['sigma2'].values if 'sigma2' in ds else ds['sigma2_l_target'].values
        # the plotted composite is the high-minus-low SAM difference
        def d(name):
            k = f'{name}_diff'
            return -ds[k].values / 1e9 if k in ds else np.full_like(sig, np.nan)
        heat = d('total_heat_surface_exchange_flux_nonadvective_heat')
        ice = d('surface_ocean_flux_advective_negative_rhs_sea_ice_melt_salt')
        oth = (d('surface_ocean_flux_advective_negative_rhs_evaporation_salt')
               + d('surface_ocean_flux_advective_negative_rhs_snow_salt')
               + d('surface_ocean_flux_advective_negative_rhs_rain_and_ice_salt')
               + d('surface_ocean_flux_advective_negative_rhs_rivers_salt'))
        frames.append(pd.DataFrame({
            'sigma2_kg_m-3': sig,
            f'{lab}_total_Sv': heat + ice + oth,
            f'{lab}_heat_Sv': heat,
            f'{lab}_seaice_FW_Sv': ice,
            f'{lab}_other_FW_Sv': oth,
        }).set_index('sigma2_kg_m-3'))
        ds.close()
    if frames:
        sheets[f'Fig5_SAM_{reglab[:12].replace(" ", "_")}'] = pd.concat(frames, axis=1).reset_index()

# ------------------------------------------------------- Fig 7: ideal age
BASINS = {
    'Atlantic': dict(lat=(-35, 65), lon=(-70, 20)),
    'Indian':   dict(lat=(-35, 25), lon=(20, 115)),
    'Pacific':  dict(lat=(-35, 65), lon=None),
    'Southern': dict(lat=(-78, -35), lon=None),
}
prof_frames = {}
below = {}
for e, lab in zip(EXPS, LABELS):
    d = xr.open_dataset(BASE / e / 'age_reg.nc')
    a = d['age'].isel(time=0)
    a = a.where(np.isfinite(a) & (a < 1e19))
    dep = -d['depth_coord'].values
    for bname, bsel in BASINS.items():
        sel = a.sel(lat=slice(*bsel['lat']))
        if bname == 'Pacific':
            sel = sel.where((sel.lon >= 120) | (sel.lon <= -70))
        elif bname != 'Southern':
            sel = sel.sel(lon=slice(*bsel['lon']))
        prof = sel.mean(dim=['lat', 'lon'], skipna=True).values
        prof_frames.setdefault('depth_m', dep)
        prof_frames[f'{bname}_{lab}_yr'] = prof
        below[(bname, lab)] = float(np.nanmean(prof[dep >= 2000]))
    d.close()
sheets['Fig7_age_profiles'] = pd.DataFrame(prof_frames)
sheets['Fig7_age_below2000m'] = pd.DataFrame(
    [{'basin': b, **{l: round(below[(b, l)], 1) for l in LABELS}} for b in BASINS])

# ------------------------------------------------------- Fig 8: domain sensitivity
frames = []
for e, lab in zip(EXPS, LABELS):
    f = BASE / f'wmt_siz_{e}.nc'
    if not f.exists():
        continue
    d = xr.open_dataset(f)
    sig = d['sigma2_l_target'].values
    for dom, domlab in [('so60', 'south_of_60S'), ('siz', 'sea_ice_zone')]:
        heat = -d[f'{dom}_total_heat_heat'].mean('time').values / 1e9

        def g(n, dom=dom, heat=heat):
            k = f'{dom}_surface_ocean_flux_advective_negative_rhs_{n}_salt'
            return -d[k].mean('time').values / 1e9 if k in d else np.zeros_like(heat)

        ice = g('sea_ice_melt')
        oth = g('evaporation') + g('snow') + g('rain_and_ice') + g('rivers')
        frames.append(pd.DataFrame({
            'sigma2_kg_m-3': sig,
            f'{lab}_{domlab}_total_Sv': heat + ice + oth,
            f'{lab}_{domlab}_heat_Sv': heat,
            f'{lab}_{domlab}_seaice_Sv': ice,
            f'{lab}_{domlab}_other_Sv': oth,
        }).set_index('sigma2_kg_m-3'))
    d.close()
if frames:
    sheets['Fig8_domain_sensitivity'] = pd.concat(frames, axis=1).reset_index()

# ------------------------------------------------------- Fig 6: MOC cell strengths
rows = []
for e, lab in zip(EXPS, LABELS):
    f = nc.Dataset(f'/work/ba1066/a270064/production/{e}_age/diag_gmoc.nc')
    moc = np.array(f.variables['MOC'][:])
    lats = np.array(f.variables['lats'][:])
    deps = np.array(f.variables['deps'][:])
    m = moc[-20:].mean(axis=0)
    aabw = float(np.nanmin(m[np.ix_(deps <= -2000, lats < -30)]))
    nadw = float(np.nanmax(m[np.ix_((deps <= -500) & (deps >= -4000), (lats > 0) & (lats < 70))]))
    rows.append({'experiment': lab, 'AABW_cell_Sv': round(aabw, 2), 'NADW_cell_Sv': round(nadw, 2)})
    f.close()
sheets['Fig6_MOC_cell_strength'] = pd.DataFrame(rows)

# ------------------------------------------------------- write
OUT.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(OUT, engine='openpyxl') as xw:
    for name, df in sheets.items():
        df.to_excel(xw, sheet_name=name[:31], index=False)

print(f'wrote {OUT}')
for name, df in sheets.items():
    print(f'  {name[:31]:34s} {df.shape[0]:5d} rows x {df.shape[1]:3d} cols')
