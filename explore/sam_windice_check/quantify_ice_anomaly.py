"""Quantify the high-minus-low SAM JJA sea ice concentration anomaly of Supplementary Fig. A10
(same composites as plot_sam_wind_ice.py, 1.2 sigma). Area change (10^3 km2) by pack zone and sector.
Run from project root: python3 -s explore/sam_windice_check/quantify_ice_anomaly.py
PI ice: pi/100years/a_ice_mergetime.nc (model years 2300-2399; not the overwritten clim file)."""
import numpy as np, xarray as xr, sys, os
sys.path.insert(0, os.getcwd())
import importlib.util
EXPS = ['pi', 'mh', 'lig', 'lgm', 'mis']
MESH = {'pi': 'mesh_core2', 'mh': 'mesh_core2', 'lig': 'mesh_core2', 'lgm': 'mesh_glac1d', 'mis': 'mesh_glac1d_38k'}
THRESH = 1.2

def sam_phases(exp):
    d = xr.open_dataset(f'{exp}/100years/slp_mergetime.nc')
    v = [x for x in d.data_vars if d[x].ndim == 3][0]
    lat = d['lat'].values
    zm = d[v].mean('lon')
    p40 = zm.isel(lat=int(np.argmin(np.abs(lat + 40)))).values
    p65 = zm.isel(lat=int(np.argmin(np.abs(lat + 65)))).values
    p40 = (p40 - p40.mean()) / p40.std(); p65 = (p65 - p65.mean()) / p65.std()
    sam = (p40 - p65).reshape(-1, 12)[:, 5:8].mean(axis=1)
    m, s = sam.mean(), sam.std()
    return np.where(sam > m + THRESH * s)[0], np.where(sam < m - THRESH * s)[0]

out = []
for e in EXPS:
    hi, lo = sam_phases(e)
    m = xr.open_dataset(f'/home/a/a270064/bb1029/inputs/{MESH[e]}/fesom.mesh.diag.nc')
    lat = np.degrees(m['nodes'].isel(n2=1).values); lon = np.degrees(m['nodes'].isel(n2=0).values)
    na = m['nod_area']; area = (na.isel({na.dims[0]: 0}) if na.ndim == 2 else na).values
    ai = xr.open_dataset(f'{e}/100years/a_ice_mergetime.nc')['a_ice'].values
    ai = ai.reshape(-1, 12, ai.shape[-1])[:, 5:8].mean(axis=1)
    mean = ai.mean(axis=0); d = ai[hi].mean(0) - ai[lo].mean(0)
    so = lat < -40
    lon360 = np.mod(lon, 360)
    pac = so & (lon360 >= 150) & (lon360 < 290)      # 150E-70W: Ross, Amundsen, Bellingshausen
    atl = so & ~((lon360 >= 150) & (lon360 < 290))
    inner = mean > 0.9; edge = (mean > 0.15) & (mean <= 0.9); out_ = mean <= 0.15
    A = lambda msk: np.nansum(d[msk] * area[msk]) / 1e9   # 10^3 km2
    row = dict(exp=e, n_hi=len(hi), n_lo=len(lo), total=A(so),
               inner=A(so & inner), edge=A(so & edge), outside=A(so & out_),
               pac=A(pac), atl=A(atl), pac_pos=np.nansum(np.clip(d[pac], 0, None) * area[pac]) / 1e9,
               atl_neg=np.nansum(np.clip(d[atl], None, 0) * area[atl]) / 1e9,
               abs_total=np.nansum(np.abs(d[so]) * area[so]) / 1e9)
    # zonal-mean anomaly profile, 2 deg bins
    bins = np.arange(-78, -40, 2.0)
    prof = []
    for b0 in bins:
        k = so & (lat >= b0) & (lat < b0 + 2)
        prof.append(np.nansum(d[k] * area[k]) / 1e9)
    row['prof'] = prof
    out.append(row)
    print(e, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items() if k != 'prof'})
    print('   lat-band area change (10^3 km2), bins from', bins[0], 'step 2:', ' '.join(f'{x:.0f}' for x in prof))
