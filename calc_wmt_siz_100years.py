#!/usr/bin/env python
"""Fig. 8 (domain sensitivity) from the 100 years of monthly output, in sigma2 bins 0.1 kg m-3 wide
(user 11-27 12:44: the 12-month climatology of calc_wmt_siz_pellichero.py gives a bin-to-bin zigzag at 0.1).

Domains exactly as calc_wmt_siz_pellichero.py: siz = climatological monthly maximum of <exp>/a_ice_clim.nc
above 0.15 and lat < 0; so60 = lat < -60; open_only = so60 and not siz (mesh fesom.mesh.diag.nc).
Fluxes and recipe as the main WMT (calc_wmt_100years_bin0p1.py): each month binned, here averaged by calendar
month over the years, so the output has the 12-step time axis and the variable names of wmt_siz_<exp>.nc
(value -1035*sum(F*A)/dsigma, so that -V/1e9 is Sv) and plot_wmt_domain_sensitivity.py only changes the file.
Also writes the 0.2-bin version for comparison. Usage: python3 -s calc_wmt_siz_100years.py <exp>
Output: wmt_siz100_<exp>_bin0p1.nc and wmt_siz100_<exp>_bin0p2.nc in the project root."""
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
MESH = {'pi': 'mesh_core2', 'mh': 'mesh_core2', 'lig': 'mesh_core2', 'lgm': 'mesh_glac1d', 'mis': 'mesh_glac1d_38k'}[EXP]
WID = {'bin0p1': (0.1, np.round(np.arange(0, 39.8 + 1e-9, 0.1), 6)), 'bin0p2': (0.2, np.arange(0, 40, 0.2))}
SALT = {'evs': 'surface_ocean_flux_advective_negative_rhs_evaporation_salt',
        'prsn': 'surface_ocean_flux_advective_negative_rhs_snow_salt',
        'fsitherm': 'surface_ocean_flux_advective_negative_rhs_sea_ice_melt_salt',
        'prlq': 'surface_ocean_flux_advective_negative_rhs_rain_and_ice_salt',
        'friver': 'surface_ocean_flux_advective_negative_rhs_rivers_salt'}
COMPS = ['heat'] + list(SALT)
TB = 60
t0 = time.time()

m = nc.Dataset(f'/home/a/a270064/bb1029/inputs/{MESH}/fesom.mesh.diag.nc')
lat_m = np.degrees(m['nodes'][1, :])
m.close()
a = nc.Dataset(f'{ROOT}/{EXP}/a_ice_clim.nc')
aice = np.squeeze(a['a_ice'][:]).astype(np.float64)
a.close()
amax = aice.max(axis=0)
siz = (amax > 0.15) & (lat_m < 0)
so60 = lat_m < -60
DOM = {'siz': siz, 'so60': so60, 'open_only': so60 & ~siz}
sub = np.where(siz | so60)[0]

f = nc.Dataset(f'{ROOT}/{EXP}/surface_density_tendency_100years_{EXP}.nc')
f.set_auto_mask(False)
lat = f['lat'][:].astype(np.float64)
assert np.allclose(lat, lat_m, atol=1e-3), 'mesh and data latitudes differ'
area = f['areacello'][:].astype(np.float64)[sub]
p_ref = gsw.p_from_z(-2000.0, lat[sub])
tv = f['time']
mon = np.array([d.month for d in nc.num2date(tv[:], tv.units, getattr(tv, 'calendar', 'standard'))]) - 1
nt = mon.size
nmon = np.bincount(mon, minlength=12)
dm = {k: v[sub] for k, v in DOM.items()}
ACC = {(w, dn, c): np.zeros((12, e.size - 1)) for w, (_, e) in WID.items() for dn in DOM for c in COMPS}
READ = ['tos', 'sos', 'SW_flux', 'LW_flux', 'LH_flux', 'SH_flux'] + list(SALT)
for b0 in range(0, nt, TB):
    b1 = min(nt, b0 + TB)
    D = {v: f[v][b0:b1, :][:, sub].astype(np.float64) for v in READ}
    S, T = D['sos'], D['tos']
    sig = gsw.sigma2(S, T)
    al, be = gsw.alpha(S, T, p_ref[None, :]), gsw.beta(S, T, p_ref[None, :])
    F = {'heat': -(al / 3992.0) * (D['SW_flux'] + D['LW_flux'] + D['LH_flux'] + D['SH_flux'])}
    for c in SALT:
        F[c] = -be * S * D[c]
    mm = mon[b0:b1]
    for w, (dw, edges) in WID.items():
        nb = edges.size - 1
        kb = np.searchsorted(edges, sig, side='right') - 1
        ok = (kb >= 0) & (kb < nb)
        mmb = np.broadcast_to(mm[:, None], sig.shape)
        for dn in DOM:
            sel = ok & dm[dn][None, :]
            flat = mmb[sel] * nb + kb[sel]
            for c in COMPS:
                s_ = np.bincount(flat, weights=(F[c] * area[None, :])[sel], minlength=12 * nb)
                ACC[(w, dn, c)] += s_.reshape(12, nb)
    print(f'{EXP}: months {b0}-{b1 - 1} ({time.time() - t0:.0f} s)', flush=True)
f.close()

for w, (dw, edges) in WID.items():
    o = nc.Dataset(f'{ROOT}/wmt_siz100_{EXP}_{w}.nc', 'w')
    o.createDimension('time', 12)
    o.createDimension('sigma2_l_target', edges.size - 1)
    o.createVariable('time', 'f8', ('time',))[:] = np.arange(12)
    o.createVariable('sigma2_l_target', 'f8', ('sigma2_l_target',))[:] = 0.5 * (edges[1:] + edges[:-1])
    for dn in DOM:
        for c in COMPS:
            name = f'{dn}_total_heat_heat' if c == 'heat' else f'{dn}_{SALT[c]}'
            o.createVariable(name, 'f8', ('time', 'sigma2_l_target'))[:] = \
                -1035.0 * ACC[(w, dn, c)] / nmon[:, None] / dw
    o.experiment = EXP.upper()
    o.description = ('WMT over the seasonal sea ice zone (climatological max ice > 15%), south of 60S, and '
                     'its ice-free part; 100 years of monthly output, binned each month, averaged by calendar month')
    o.bin_width = dw
    o.siz_area_m2 = float(area[dm['siz']].sum())
    o.so60_area_m2 = float(area[dm['so60']].sum())
    o.open_area_m2 = float(area[dm['open_only']].sum())
    o.close()
print(f'{EXP}: wrote wmt_siz100_{EXP}_bin0p1.nc and _bin0p2.nc ({time.time() - t0:.0f} s)')
