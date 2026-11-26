#!/usr/bin/env python
"""Where the deep winter mixed layers are, near the Antarctic coast or offshore,
in the PI simulation and in ORAS5 (JJA 1990-2009, opa0). User note 2026-11-26:
"ORAS5 convection is mostly in coastal polynyas, ours too but weaker, mostly
in the open Weddell". Native grids, south of 60S, area weighted.
Model: fig_formation_location/data/regions_pi.nc (JJA MLD1, coast_km, PI ice
from a_ice_clim_beta.nc). ORAS5: somxl010 + ileadfra, cell area e1t*e2t,
coastline = ocean T points next to the largest land body south of 60S.
Run: python3 -s coastal_mld.py > numbers.txt
"""
import numpy as np
import netCDF4 as nc
from scipy.ndimage import label, binary_dilation
from scipy.spatial import cKDTree

R = 6371.0
O = '/pool/data/ICDC/ocean_syntheses/oras5/ORCA025'
REG = '/work/ba1066/a270064/cc_projects/aabw_5exps/fig_formation_location/data/regions_pi.nc'


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def jja(var, kind):
    acc, n = None, 0
    for y in range(1990, 2010):
        for mo in ('06', '07', '08'):
            d = nc.Dataset(f'{O}/{var}/opa0/{var}_ORAS5_1m_{y}{mo}_{kind}_02.nc')
            a = np.ma.filled(d[var][0].astype(float), np.nan)
            d.close()
            acc = a if acc is None else acc + a
            n += 1
    return acc / n, n


m = nc.Dataset(f'{O}/mesh/mesh_mask.nc')
lat = m['gphit'][0]; lon = m['glamt'][0]
area = (m['e1t'][0] * m['e2t'][0]).astype(float)
ocean = m['tmaskutil'][0] == 1
m.close()
mld, n = jja('somxl010', 'grid_T')
ice, _ = jja('ileadfra', 'icemod')
south = lat < -55
lab, _ = label((~ocean) & south)
sizes = np.bincount(lab.ravel()); sizes[0] = 0
ant = lab == sizes.argmax()
coastpts = ocean & binary_dilation(ant)
d_o = 2 * R * np.arcsin(np.clip(cKDTree(xyz(lat[coastpts], lon[coastpts])).query(
    xyz(lat.ravel(), lon.ravel()))[0] / 2, 0, 1)).reshape(lat.shape)
sel = ocean & (lat <= -60) & np.isfinite(mld)
obs = dict(mld=mld[sel], area=area[sel], coast=d_o[sel], ice=ice[sel], lat=lat[sel], lon=lon[sel])

r = nc.Dataset(REG); r.set_auto_mask(False)
s = r['lat'][:] <= -60
mod = dict(mld=np.abs(r['mld1_jja'][:][s]), area=r['area'][:][s], coast=r['coast_km'][:][s],
           ice=r['aice_jja'][:][s], lat=r['lat'][:][s], lon=r['lon'][:][s])
r.close()

print(f'ORAS5 JJA months averaged: {n}; ORAS5 coastline points {coastpts.sum()}')
for name, D in [('AWI-ESM PI (MLD1)', mod), ('ORAS5 1990-2009 (somxl010)', obs)]:
    A = D['area']; near = D['coast'] <= 200
    wed = (D['lon'] >= -60) & (D['lon'] <= 30)
    print(f'\n== {name}: area south of 60S {A.sum()/1e12:.2f}e12 m2, within 200 km of coast {A[near].sum()/1e12:.2f}e12 m2')
    print(f'  mean MLD: within 200 km {np.average(D["mld"][near], weights=A[near]):.0f} m, '
          f'beyond {np.average(D["mld"][~near], weights=A[~near]):.0f} m')
    for thr in (100, 150, 200, 300, 400):
        k = D['mld'] > thr
        if A[k].sum() == 0:
            print(f'  MLD > {thr} m: no area'); continue
        print(f'  MLD > {thr:3d} m: area {A[k].sum()/1e12:.3f}e12 m2; within 200 km of coast {100*A[k&near].sum()/A[k].sum():.0f}%'
              f' (of which JJA ice < 0.7: {100*A[k&near&(D["ice"]<0.7)].sum()/max(A[k&near].sum(),1):.0f}%);'
              f' offshore Weddell 60W-30E {100*A[k&~near&wed].sum()/A[k].sum():.0f}%;'
              f' max {D["mld"][k].max():.0f} m')
    for L in (100, 300):
        nn = D['coast'] <= L
        k = D['mld'] > 200
        if A[k].sum():
            print(f'  sensitivity: MLD > 200 m within {L} km of coast {100*A[k&nn].sum()/A[k].sum():.0f}%')
    # coastal band, deepest 10% of its area
    q = D['mld'][near]; w = A[near]; o = np.argsort(q); c = np.cumsum(w[o]) / w.sum()
    print(f'  coastal band MLD percentiles (area): p50 {np.interp(.5,c,q[o]):.0f}, p90 {np.interp(.9,c,q[o]):.0f}, p99 {np.interp(.99,c,q[o]):.0f} m')
