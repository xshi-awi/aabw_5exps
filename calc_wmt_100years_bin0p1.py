#!/usr/bin/env python
"""Monthly surface WMT of the four paper regions in sigma2 bins 0.1 kg m-3 wide (user 11-27 12:04, button GN:
"换0.1全部重算"). Writes <exp>/wmt_results_bin0p1/wmt_<region>_100years_<exp>.nc with the same dimensions,
time axis and variable names as the published 0.2-bin files in <exp>/wmt_results/, so every plotting script
only needs the folder name changed.

Recipe of the published xwmt/xbudget files (reproduced to 1e-5 Sv in explore/wmt_binwidth and
fig_formation_location): sigma2 = gsw.sigma2(sos, tos); alpha, beta at p_ref = gsw.p_from_z(-2000, lat);
F_heat = -(alpha/3992)(SW+LW+LH+SH); F_x = -beta*sos*x for x in evs, prsn, prlq, friver, fsitherm.
Stored value V = -1035 * sum(F*area)/dsigma per month and bin, so that -V/1e9 is the transformation in Sv,
the sign and scale of the published variables. Regions: lat in [-90,-60]; lon [-180,180], [-180,-60],
[-60,79], [79,180], both ends inclusive.
The same run also bins at 0.2 (published edges) and checks every month and variable against the published
file, and checks that sum(G*dsigma) is the same at 0.1 and 0.2 (conservation).
Usage: python3 -s calc_wmt_100years_bin0p1.py <exp>"""
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
W_NEW = 0.1
# same range as the published edges (last edge 39.8): a few hypersaline outliers above 39.8 (PI: 15 node-months,
# max 40.3) are left out in the published files and must be left out here too, or the totals differ
E_NEW = np.round(np.arange(0, 39.8 + 1e-9, W_NEW), 6)       # 0.0 ... 39.8, 398 bins
E_OLD = np.arange(0, 40, 0.2)                                # published: 0.0 ... 39.8, 199 bins
REG = {'southern_ocean': (-180, 180), 'ross_sea': (-180, -60), 'weddell_sea': (-60, 79), 'adelie': (79, 180)}
VN = {'heat': 'total_heat_surface_exchange_flux_nonadvective_heat',
      'evs': 'surface_ocean_flux_advective_negative_rhs_evaporation_salt',
      'prsn': 'surface_ocean_flux_advective_negative_rhs_snow_salt',
      'fsitherm': 'surface_ocean_flux_advective_negative_rhs_sea_ice_melt_salt',
      'prlq': 'surface_ocean_flux_advective_negative_rhs_rain_and_ice_salt',
      'friver': 'surface_ocean_flux_advective_negative_rhs_rivers_salt'}
COMPS = list(VN)
TB = 60

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
nt = f.dimensions['time'].size
p_ref = gsw.p_from_z(-2000.0, lat)
nb_new, nb_old = E_NEW.size - 1, E_OLD.size - 1
V_new = {(r, c): np.zeros((nt, nb_new)) for r in REG for c in COMPS}
V_old = {(r, c): np.zeros((nt, nb_old)) for r in REG for c in COMPS}
READ = ['tos', 'sos', 'SW_flux', 'LW_flux', 'LH_flux', 'SH_flux', 'evs', 'prsn', 'prlq', 'friver', 'fsitherm', 'wfo']
wfo_err = 0.0
for b0 in range(0, nt, TB):
    b1 = min(nt, b0 + TB)
    D = {v: f[v][b0:b1, :][:, sub].astype(np.float64) for v in READ}
    S, T = D['sos'], D['tos']
    sig = gsw.sigma2(S, T)
    al, be = gsw.alpha(S, T, p_ref[None, :]), gsw.beta(S, T, p_ref[None, :])
    F = {'heat': -(al / 3992.0) * (D['SW_flux'] + D['LW_flux'] + D['LH_flux'] + D['SH_flux'])}
    for c in COMPS[1:]:
        F[c] = -be * S * D[c]
    # the published "other" term was wfo - fsitherm; check the four parts add up to it
    wfo_err = max(wfo_err, float(np.abs(D['evs'] + D['prsn'] + D['prlq'] + D['friver'] + D['fsitherm'] - D['wfo']).max()))
    n = b1 - b0
    for edges, nb, V, w in ((E_NEW, nb_new, V_new, W_NEW), (E_OLD, nb_old, V_old, 0.2)):
        kb = np.searchsorted(edges, sig, side='right') - 1
        ok = (kb >= 0) & (kb < nb)
        tt = np.broadcast_to(np.arange(n)[:, None], sig.shape)
        for r in REG:
            sel = ok & masks[r][None, :]
            flat = tt[sel] * nb + kb[sel]
            for c in COMPS:
                s_ = np.bincount(flat, weights=(F[c] * area[None, :])[sel], minlength=n * nb)
                V[(r, c)][b0:b1] = -1035.0 * s_.reshape(n, nb) / w
    print(f'{EXP}: months {b0}-{b1 - 1} ({time.time() - t0:.0f} s)', flush=True)
f.close()
print(f'{EXP}: max |evs+prsn+prlq+friver+fsitherm - wfo| = {wfo_err:.3e} kg m-2 s-1')

out_dir = f'{ROOT}/{EXP}/wmt_results_bin0p1'
os.makedirs(out_dir, exist_ok=True)
for r in REG:
    pub = nc.Dataset(f'{ROOT}/{EXP}/wmt_results/wmt_{r}_100years_{EXP}.nc')
    assert pub.dimensions['time'].size == nt
    # 1. the 0.2 recomputation must match the published file
    worst = 0.0
    for c in COMPS:
        P = pub[VN[c]][:].astype(np.float64)                 # (time, 199)
        worst = max(worst, float(np.abs(V_old[(r, c)] - P).max()) / 1e9)
    # 2. conservation: sum over bins of G*dsigma must be the same at 0.1 and 0.2
    cons = max(float(np.abs(V_new[(r, c)].sum(1) * W_NEW - V_old[(r, c)].sum(1) * 0.2).max()) / 1e9 for c in COMPS)
    print(f'{EXP} {r}: max |0.2 recomputed - published| = {worst:.2e} Sv; '
          f'max |sum(G dsigma) 0.1 - 0.2| = {cons:.2e} Sv kg m-3')
    assert worst < 1e-3 and cons < 1e-6, (worst, cons)
    o = nc.Dataset(f'{out_dir}/wmt_{r}_100years_{EXP}.nc', 'w')
    o.createDimension('time', nt)
    o.createDimension('sigma2_l_target', nb_new)
    tv = o.createVariable('time', 'f8', ('time',))
    tv[:] = pub['time'][:]
    for a in pub['time'].ncattrs():
        tv.setncattr(a, pub['time'].getncattr(a))
    sv = o.createVariable('sigma2_l_target', 'f8', ('sigma2_l_target',))
    sv[:] = 0.5 * (E_NEW[1:] + E_NEW[:-1])
    sv.units = 'kg m-3'
    sv.long_name = 'sigma2 bin centre, bins 0.1 kg m-3 wide'
    for c in COMPS:
        v = o.createVariable(VN[c], 'f8', ('time', 'sigma2_l_target'), zlib=True)
        v[:] = V_new[(r, c)]
        for a in pub[VN[c]].ncattrs():
            if a != '_FillValue':
                v.setncattr(a, pub[VN[c]].getncattr(a))
    o.bin_width = W_NEW
    o.recipe = 'calc_wmt_100years_bin0p1.py; same recipe and sign as the 0.2-bin files in ../wmt_results'
    o.close()
    pub.close()
print(f'{EXP}: wrote {out_dir} ({time.time() - t0:.0f} s)')
