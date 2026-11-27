#!/usr/bin/env python
"""Winter (JJA) surface WMT of manuscript Fig. 2 recomputed with narrower sigma2 bins.

Same recipe as the published curves (pi/wmt_100years_timeseries_pi.py with xwmt/xbudget, reproduced
exactly in fig_formation_location/calc_formation_location.py):
  sigma2 = gsw.sigma2(sos, tos);  alpha, beta at p_ref = gsw.p_from_z(-2000, lat)
  F_heat = -(alpha/3992)(SW+LW+LH+SH),  F_ice = -beta sos fsitherm,  F_other = -beta sos (wfo - fsitherm)
  G(bin) = 1035/1e9 * sum(F * area) / dsigma  [Sv], each month binned, then JJA mean per year,
  then mean and standard deviation over the years (as plot_wmt_4regions_5exps_from_100years.py).
Regions as in the paper: lat in [-90,-60]; Ross lon [-180,-60], Weddell [-60,79], Adelie [79,180]
(both ends inclusive, as in the published masks).
Bin widths 0.2 (published edges 0, 0.2, ..., 39.8), 0.1, 0.05, 0.02, all with edges on multiples of the
width. Usage: python3 -s calc_binwidth.py <exp>  ->  data/G_bw_<exp>.nc"""
import os
import sys
import tempfile
import time

import netCDF4 as nc
import numpy as np

try:
    import gsw
except ImportError:
    d = tempfile.mkdtemp()
    os.symlink('/home/a/a270064/.local/lib/python3.9/site-packages/gsw', f'{d}/gsw')
    sys.path.insert(0, d)
    import gsw

EXP = sys.argv[1]
ROOT = '/work/ba1066/a270064/cc_projects/aabw_5exps'
HERE = os.path.dirname(os.path.abspath(__file__))
CP, RHO = 3992.0, 1035.0
WIDTHS = {'0p20': 0.2, '0p10': 0.1, '0p05': 0.05, '0p02': 0.02}
EDGES = {k: np.round(np.arange(0, 40.0 + 1e-9, w), 6) for k, w in WIDTHS.items()}
EDGES['0p20'] = np.arange(0, 40, 0.2)               # published edges: 0 ... 39.8 (199 bins)
REG = {'southern_ocean': (-180, 180), 'ross_sea': (-180, -60), 'weddell_sea': (-60, 79), 'adelie': (79, 180)}
COMPS = ['heat', 'ice', 'other']
TB = 60       # JJA months per block (20 years)

t0 = time.time()
f = nc.Dataset(f'{ROOT}/{EXP}/surface_density_tendency_100years_{EXP}.nc')
f.set_auto_mask(False)
lat = f['lat'][:].astype(np.float64)
lon = f['lon'][:].astype(np.float64)
lon = np.where(lon > 180, lon - 360, lon)
area = f['areacello'][:].astype(np.float64)
sub = np.where(lat <= -60)[0]
lat, lon, area = lat[sub], lon[sub], area[sub]
masks = {r: (lon >= a) & (lon <= b) for r, (a, b) in REG.items()}
tv = f['time']
dates = nc.num2date(tv[:], tv.units, getattr(tv, 'calendar', 'standard'))
mon = np.array([d.month for d in dates])
yr = np.array([d.year for d in dates])
jja = np.where(np.isin(mon, [6, 7, 8]))[0]
years = np.unique(yr[jja])
assert all((yr[jja] == y).sum() == 3 for y in years), 'incomplete JJA'
p_ref = gsw.p_from_z(-2000.0, lat)
nb = {k: e.size - 1 for k, e in EDGES.items()}
# per month: region x comp x bin
Gm = {k: np.zeros((jja.size, len(REG), len(COMPS), nb[k])) for k in EDGES}
VARS = ['tos', 'sos', 'SW_flux', 'LW_flux', 'LH_flux', 'SH_flux', 'wfo', 'fsitherm']
for i0 in range(0, jja.size, TB):
    idx = jja[i0:i0 + TB]
    # read each year's three JJA months as one contiguous slice
    starts = idx[::3]
    assert (np.diff(idx.reshape(-1, 3), axis=1) == 1).all()
    D = {v: np.concatenate([f[v][j:j + 3, :][:, sub] for j in starts]).astype(np.float64) for v in VARS}
    S, T = D['sos'], D['tos']
    sig = gsw.sigma2(S, T)
    al, be = gsw.alpha(S, T, p_ref[None, :]), gsw.beta(S, T, p_ref[None, :])
    Q = D['SW_flux'] + D['LW_flux'] + D['LH_flux'] + D['SH_flux']
    F = {'heat': -(al / CP) * Q, 'ice': -be * S * D['fsitherm'], 'other': -be * S * (D['wfo'] - D['fsitherm'])}
    for k, e in EDGES.items():
        w = WIDTHS[k]
        kb = np.searchsorted(e, sig, side='right') - 1
        ok = (kb >= 0) & (kb < nb[k])
        for ri, r in enumerate(REG):
            sel = ok & masks[r][None, :]
            tt = np.broadcast_to(np.arange(idx.size)[:, None], sig.shape)[sel]
            flat = tt * nb[k] + kb[sel]
            for ci, c in enumerate(COMPS):
                s_ = np.bincount(flat, weights=(F[c] * area[None, :])[sel], minlength=idx.size * nb[k])
                Gm[k][i0:i0 + idx.size, ri, ci, :] = s_.reshape(idx.size, nb[k]) * RHO / 1e9 / w
    print(f'{EXP}: JJA months {i0}-{i0 + idx.size - 1} ({time.time() - t0:.0f} s)', flush=True)
f.close()

os.makedirs(f'{HERE}/data', exist_ok=True)
o = nc.Dataset(f'{HERE}/data/G_bw_{EXP}.nc', 'w')
o.createDimension('region', len(REG))
o.createDimension('comp', len(COMPS))
o.createDimension('year', years.size)
o.regions = ' '.join(REG)
o.comps = ' '.join(COMPS)
yidx = np.searchsorted(years, yr[jja])
for k in EDGES:
    o.createDimension(f'bin_{k}', nb[k])
    o.createVariable(f'center_{k}', 'f8', (f'bin_{k}',))[:] = 0.5 * (EDGES[k][1:] + EDGES[k][:-1])
    ym = np.zeros((years.size, len(REG), len(COMPS), nb[k]))
    for yi in range(years.size):
        ym[yi] = Gm[k][yidx == yi].mean(axis=0)
    v = o.createVariable(f'G_{k}', 'f4', ('year', 'region', 'comp', f'bin_{k}'), zlib=True)
    v[:] = ym
    v.units = 'Sv, JJA mean of each year, positive = towards denser water'
o.experiment = EXP
o.close()
print(f'wrote data/G_bw_{EXP}.nc ({time.time() - t0:.0f} s)')
