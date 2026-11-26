#!/usr/bin/env python
"""Per-node, per-density-bin accumulation of the surface density flux that
drives the paper's surface WMT, so that the transformation can be split by
region afterwards (analyze_formation_location.py).

Reproduces the xwmt/xbudget recipe used for the published curves
(pi/wmt_100years_timeseries_pi.py + plot_wmt_4regions_5exps_from_100years.py):
  sigma2 = gsw.sigma2(sos, tos)   (xwmt default: sos taken as SA, tos as CT)
  alpha, beta = gsw.alpha/beta(sos, tos, p_ref), p_ref = gsw.p_from_z(-2000, lat)
  F_heat  = -(alpha / 3992) * (SW + LW + LH + SH)            [kg m-2 s-1]
  F_ice   = -beta * sos * fsitherm                            [kg m-2 s-1]
  F_other = -beta * sos * (wfo - fsitherm)   (evs+prlq+prsn+friver)
  G(bin)  = rho_ref/1e9 * sum(F * area) / dsigma,  rho_ref = 1035  [Sv]
  bins    = np.arange(0, 40, 0.2) edges, monthly binning, then time mean.
The basal salt term (sfdsi) is not in the published curves and is left out.
Three binnings are accumulated: 'sig2c' (the published 0.2 bins, used to
check the reproduction), 'sig2f' (sigma2 in 0.02 bins, 33.0-38.6, plus one
underflow and one overflow bin) and 'sig0f' (sigma0 with alpha/beta at p=0,
0.02 bins 23.0-28.6, plus under/overflow), a stand-in for neutral density; it
is NOT gamma_n.

Usage: python3 -s calc_formation_location.py <exp>     (exp in pi mh lig lgm mis)
Output: data/acc_<exp>.nc  (gitignored)
"""
import sys
import os
import time
import tempfile
import numpy as np
import netCDF4 as nc

try:
    import gsw
except ImportError:  # gsw lives in ~/.local next to a pandas that breaks xarray
    d = tempfile.mkdtemp()
    os.symlink('/home/a/a270064/.local/lib/python3.9/site-packages/gsw', f'{d}/gsw')
    sys.path.insert(0, d)
    import gsw

EXP = sys.argv[1]
ROOT = '/work/ba1066/a270064/cc_projects/aabw_5exps'
HERE = f'{ROOT}/fig_formation_location'
INP = '/home/a/a270064/bb1029/inputs'
MESH = {'pi': 'mesh_core2', 'mh': 'mesh_core2', 'lig': 'mesh_core2',
        'lgm': 'mesh_glac1d', 'mis': 'mesh_glac1d_38k'}[EXP]
LAT_SUB = -50.0          # keep nodes south of this for maps and integrals
CP, RHO_REF = 3992.0, 1035.0
EDGES = np.arange(0, 40, 0.2)          # identical to the published pipeline
FINE2 = np.concatenate([[-np.inf], np.round(np.arange(33.0, 38.6001, 0.02), 4), [np.inf]])
FINE0 = np.concatenate([[-np.inf], np.round(np.arange(23.0, 28.6001, 0.02), 4), [np.inf]])
BINNINGS = {'sig2c': ('sig2', EDGES), 'sig2f': ('sig2', FINE2), 'sig0f': ('sig0', FINE0)}
TBLOCK = 172                            # time chunk size of the input file

t0 = time.time()
# ------------------------------------------------------------------ mesh
m = nc.Dataset(f'{INP}/{MESH}/fesom.mesh.diag.nc')
m.set_auto_mask(False)
lon = np.degrees(m['nodes'][0])
lat = np.degrees(m['nodes'][1])
area = m['nod_area'][0].astype(np.float64)
m.close()
sub = np.where(lat < LAT_SUB)[0]
ns = sub.size
print(f'{EXP}: mesh {MESH}, {lat.size} nodes, {ns} south of {LAT_SUB}')

# ------------------------------------------------------------------ input
f = nc.Dataset(f'{ROOT}/{EXP}/surface_density_tendency_100years_{EXP}.nc')
f.set_auto_mask(False)
nt = f.dimensions['time'].size
assert f.dimensions['nod2'].size == lat.size
# the file carries the same mesh coordinates; check once
assert np.allclose(f['lat'][:], lat, atol=1e-3), 'lat mismatch'
assert np.allclose(f['areacello'][:], area, rtol=1e-5), 'area mismatch'
tv = f['time']
dates = nc.num2date(tv[:], tv.units, getattr(tv, 'calendar', 'standard'))
month = np.array([d.month for d in dates])
# LIG has 1176 months (98 years: two years are missing in its 100years/ input,
# and the published LIG WMT has the same 1176 months)
assert nt % 12 == 0 and (np.bincount(month, minlength=13)[1:] == nt // 12).all()

p_ref2 = gsw.p_from_z(-2000.0, lat[sub])          # xwmt sigma2 reference
VARS = ['tos', 'sos', 'SW_flux', 'LW_flux', 'LH_flux', 'SH_flux', 'wfo', 'fsitherm']

COMPS = ['heat', 'ice', 'other']
COORDS = ['sig2', 'sig0']
SEAS = {'jja': np.isin(month, [6, 7, 8]), 'ann': np.ones(nt, bool)}
NBS = {b: e.size - 1 for b, (_, e) in BINNINGS.items()}
acc = {(s, b, k): np.zeros(ns * NBS[b]) for s in SEAS for b in BINNINGS for k in COMPS}
cnt = {(s, b): np.zeros(ns * NBS[b]) for s in SEAS for b in BINNINGS}   # months in bin
fmean = {(s, k): np.zeros(ns) for s in SEAS for k in COMPS}  # sig2-coordinate F
sigmean = {s: np.zeros(ns) for s in SEAS}
nmon = {s: 0 for s in SEAS}

for b0 in range(0, nt, TBLOCK):
    b1 = min(nt, b0 + TBLOCK)
    D = {v: f[v][b0:b1, :][:, sub].astype(np.float64) for v in VARS}
    S, T = D['sos'], D['tos']
    assert np.isfinite(S).all() and np.isfinite(T).all()
    Q = D['SW_flux'] + D['LW_flux'] + D['LH_flux'] + D['SH_flux']
    fw_ice = D['fsitherm']
    fw_oth = D['wfo'] - D['fsitherm']
    cdat = {}
    for coord in COORDS:
        if coord == 'sig2':
            sig = gsw.sigma2(S, T)
            pr = p_ref2[None, :]
        else:
            sig = gsw.sigma0(S, T)
            pr = 0.0
        al = gsw.alpha(S, T, pr)
        be = gsw.beta(S, T, pr)
        F = {'heat': -(al / CP) * Q, 'ice': -be * S * fw_ice, 'other': -be * S * fw_oth}
        cdat[coord] = (sig, F)
    for s, sel in SEAS.items():
        tsel = sel[b0:b1]
        if not tsel.any():
            continue
        nmon[s] += int(tsel.sum())
        for b, (coord, edges) in BINNINGS.items():
            sig, F = cdat[coord]
            nb = NBS[b]
            k = np.searchsorted(edges, sig[tsel], side='right') - 1
            ok = (k >= 0) & (k < nb)
            ii = (k + (np.arange(ns) * nb)[None, :])[ok]
            for kk in COMPS:
                w = (F[kk][tsel] * area[sub][None, :])[ok]
                acc[(s, b, kk)] += np.bincount(ii, weights=w, minlength=ns * nb)
            cnt[(s, b)] += np.bincount(ii, minlength=ns * nb)
        sig, F = cdat['sig2']
        sigmean[s] += sig[tsel].sum(axis=0)
        for kk in COMPS:
            fmean[(s, kk)] += F[kk][tsel].sum(axis=0)
    print(f'  months {b0}-{b1 - 1} done ({time.time() - t0:.0f} s)')
f.close()
assert nmon['jja'] == nt // 4 and nmon['ann'] == nt, nmon

# ------------------------------------------------------------------ write
os.makedirs(f'{HERE}/data', exist_ok=True)
fn = f'{HERE}/data/acc_{EXP}.nc'
o = nc.Dataset(fn, 'w')
o.createDimension('node', ns)
o.createVariable('node_index', 'i4', ('node',))[:] = sub
o.createVariable('lon', 'f8', ('node',))[:] = lon[sub]
o.createVariable('lat', 'f8', ('node',))[:] = lat[sub]
o.createVariable('area', 'f8', ('node',))[:] = area[sub]
for b, (coord, edges) in BINNINGS.items():
    nb = NBS[b]
    o.createDimension(f'bin_{b}', nb)
    o.createDimension(f'edge_{b}', nb + 1)
    o.createVariable(f'edges_{b}', 'f8', (f'edge_{b}',))[:] = edges
for s in SEAS:
    for b in BINNINGS:
        nb = NBS[b]
        for kk in COMPS:
            v = o.createVariable(f'acc_{s}_{b}_{kk}', 'f4', ('node', f'bin_{b}'), zlib=True, complevel=4)
            v[:] = (acc[(s, b, kk)] / nmon[s]).reshape(ns, nb)
            v.units = 'kg s-1 (time-mean F*area of months whose surface density falls in the bin)'
        v = o.createVariable(f'cnt_{s}_{b}', 'f4', ('node', f'bin_{b}'), zlib=True, complevel=4)
        v[:] = (cnt[(s, b)] / nmon[s]).reshape(ns, nb)
        v.units = 'fraction of months in bin'
    o.createVariable(f'sig2_mean_{s}', 'f8', ('node',))[:] = sigmean[s] / nmon[s]
    for kk in COMPS:
        v = o.createVariable(f'F_{s}_{kk}', 'f8', ('node',))
        v[:] = fmean[(s, kk)] / nmon[s]
        v.units = 'kg m-2 s-1, positive = densifying (sigma2 alpha/beta)'
o.experiment = EXP
o.mesh = MESH
o.rho_ref = RHO_REF
o.cp = CP
o.n_months_jja = nmon['jja']
o.n_months_ann = nmon['ann']
o.source = f'{ROOT}/{EXP}/surface_density_tendency_100years_{EXP}.nc'
o.close()
print(f'wrote {fn} ({os.path.getsize(fn) / 1e6:.0f} MB, {time.time() - t0:.0f} s)')
