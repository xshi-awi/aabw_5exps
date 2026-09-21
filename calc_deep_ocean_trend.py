#!/usr/bin/env python
"""
Deep-ocean drift diagnostic.

Reviewer 3 asked whether the simulations are at equilibrium, and the surface is
the easy case: what matters for a study about abyssal ventilation is whether the
DEEP ocean is still adjusting. Surface temperature equilibrates in decades, the
abyss in millennia, so a flat global mean surface temperature is not by itself
evidence that the deep ocean has stopped drifting.

This computes volume-weighted mean potential temperature and salinity below
2000 m, per year, over all available output years for each experiment, and
reports the trend over the final 100 years that are actually analysed.

Output: deep_ocean_trend.npz plus a printed table.
"""
import sys
from pathlib import Path
import numpy as np
import xarray as xr

PROD = Path('/work/ba1066/a270064/production')
EXPS = ['pi', 'mh', 'lig', 'lgm', 'mis']
LABELS = {'pi': 'PI', 'mh': 'MH', 'lig': 'LIG', 'lgm': 'LGM', 'mis': 'MIS3'}
MESH = {'pi': 'mesh_core2', 'mh': 'mesh_core2', 'lig': 'mesh_core2',
        'lgm': 'mesh_glac1d', 'mis': 'mesh_glac1d_38k'}
ZCUT = 2000.0          # metres; "deep ocean" for this purpose


def mesh_volume(exp):
    """Per-node, per-level volume weights, and the level depths."""
    m = xr.open_dataset(f'/home/a/a270064/bb1029/inputs/{MESH[exp]}/fesom.mesh.diag.nc')
    # nod_area has a level dimension: area of each node at each level
    dim = 'nz' if 'nz' in m['nod_area'].dims else 'nl'
    area = m['nod_area'].values            # (nlev, nod2)
    # core2 mesh calls the depth axis 'zbar'; the glac1d meshes call it 'nz'.
    # Values are identical (0, 5, 10 ... 6000, 6250 m), verified.
    zkey = 'zbar' if 'zbar' in m.variables else 'nz'
    zbar = np.abs(m[zkey].values)          # layer interfaces, positive down
    nlev = area.shape[0]
    dz = np.diff(zbar)                     # thickness of each layer
    zmid = 0.5 * (zbar[:-1] + zbar[1:])
    nz1 = len(dz)
    vol = area[:nz1, :] * dz[:, None]      # (nz1, nod2)
    m.close()
    return vol, zmid, nz1


def series(exp):
    vol, zmid, nz1 = mesh_volume(exp)
    deep = zmid >= ZCUT
    w = np.where(deep[:, None], vol, 0.0)

    tfiles = sorted((PROD / f'{exp}_age' / 'outdata' / 'fesom').glob('temp.fesom.*.nc'))
    sfiles = sorted((PROD / f'{exp}_age' / 'outdata' / 'fesom').glob('salt.fesom.*.nc'))
    years, tm, sm = [], [], []
    wsum = w.sum()
    for tf in tfiles:
        yr = int(tf.name.split('.')[2][:4])
        try:
            dt = xr.open_dataset(tf)
            t = dt['temp'].values                      # (12, nod2, nz1) or (12, nz1, nod2)
            dt.close()
        except Exception as e:
            print(f'  skip {tf.name}: {e}', file=sys.stderr)
            continue
        t = t.mean(axis=0)                             # annual mean
        if t.shape[0] != nz1:                          # (nod2, nz1) -> (nz1, nod2)
            t = t.T
        t = np.where(np.isfinite(t) & (np.abs(t) < 1e10), t, np.nan)
        val = np.nansum(np.where(np.isnan(t), 0.0, t) * w) / wsum
        years.append(yr)
        tm.append(val)

        sf = tf.parent / tf.name.replace('temp.', 'salt.')
        if sf.exists():
            ds = xr.open_dataset(sf)
            sarr = ds['salt'].values.mean(axis=0)
            ds.close()
            if sarr.shape[0] != nz1:
                sarr = sarr.T
            sarr = np.where(np.isfinite(sarr) & (np.abs(sarr) < 1e10), sarr, np.nan)
            sm.append(np.nansum(np.where(np.isnan(sarr), 0.0, sarr) * w) / wsum)
        else:
            sm.append(np.nan)
    return np.array(years), np.array(tm), np.array(sm)


store = {}
print(f'Volume-weighted mean below {ZCUT:.0f} m')
print()
print('%-5s %6s %10s %12s %14s %14s' %
      ('exp', 'nyears', 'T_final', 'S_final', 'dT/100yr', 'dS/100yr'))
print('-' * 70)
for e in EXPS:
    yrs, t, s = series(e)
    if len(yrs) < 20:
        print(f'{LABELS[e]}: too few years ({len(yrs)})')
        continue
    last = slice(-100, None)
    yl = yrs[last].astype(float)
    tl, sl = t[last], s[last]
    dt = np.polyfit(yl, tl, 1)[0] * 100
    ok = np.isfinite(sl)
    dsl = np.polyfit(yl[ok], sl[ok], 1)[0] * 100 if ok.sum() > 10 else np.nan
    store[e] = dict(years=yrs, temp=t, salt=s, dT=dt, dS=dsl)
    print('%-5s %6d %10.4f %12.4f %14.5f %14.6f' %
          (LABELS[e], len(yrs), tl.mean(), np.nanmean(sl), dt, dsl))

np.savez('deep_ocean_trend.npz', **{f'{e}_{k}': v for e, d in store.items()
                                    for k, v in d.items()})
print()
print('saved deep_ocean_trend.npz')
