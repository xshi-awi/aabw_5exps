#!/usr/bin/env python
"""
Recompute PI (and all experiments) WMT over a Pellichero et al. (2018)-comparable
domain: the SEASONAL SEA ICE ZONE (SIZ), defined as nodes where the climatological
maximum sea ice concentration exceeds 15%.

This is the like-for-like comparison requested by Reviewers #2 and #3: our published
WMT integrates over everything south of 60 S, which includes a large open-ocean
area that never sees sea ice and where thermal forcing necessarily dominates.
Pellichero et al. analysed only the ice-covered sector.

Usage:  python calc_wmt_siz_pellichero.py <exp>       # exp in pi mh lig lgm mis
Output: wmt_siz_<exp>.nc  with total / heat / salt-component transformation in sigma2.

Uses climatological monthly means (12 months) so that a single run is cheap; the
annual mean transformation is what Pellichero report.
"""
import sys, warnings
import numpy as np
import xarray as xr
import xgcm, xbudget, xwmt
from pathlib import Path
from scipy.interpolate import griddata

exp = sys.argv[1] if len(sys.argv) > 1 else 'pi'

MESH = {
    'pi':  '/home/a/a270064/bb1029/inputs/mesh_core2',
    'mh':  '/home/a/a270064/bb1029/inputs/mesh_core2',
    'lig': '/home/a/a270064/bb1029/inputs/mesh_core2',
    'lgm': '/home/a/a270064/bb1029/inputs/mesh_glac1d',
    'mis': '/home/a/a270064/bb1029/inputs/mesh_glac1d_38k',
}[exp]

D = Path(exp)
print('=' * 70)
print(f'{exp.upper()}: WMT over seasonal sea ice zone (Pellichero-comparable domain)')
print('=' * 70)

# ---------------------------------------------------------------- mesh
mesh = xr.open_dataset(f'{MESH}/fesom.mesh.diag.nc')
nod_area_dims = mesh['nod_area'].dims
if 'nl' in nod_area_dims:
    areacello_1d = mesh['nod_area'].isel(nl=0).values
else:
    areacello_1d = mesh['nod_area'].isel(nz=0).values
fesom_lon_1d = np.degrees(mesh['nodes'].isel(n2=0).values)
fesom_lat_1d = np.degrees(mesh['nodes'].isel(n2=1).values)
nnodes = len(fesom_lon_1d)

# ---------------------------------------------------------------- ECHAM heat fluxes
ech = xr.open_dataset(D / 'echam_clim.nc')
echam_lon = ech['lon'].values
echam_lat = ech['lat'].values
echam_lon_c = np.where(echam_lon > 180, echam_lon - 360, echam_lon)
lon2d, lat2d = np.meshgrid(echam_lon_c, echam_lat)
points = np.column_stack([lon2d.ravel(), lat2d.ravel()])
xi = np.column_stack([fesom_lon_1d, fesom_lat_1d])

def to_fesom(v):
    """v: (12, nlat, nlon) -> (12, nnodes)"""
    out = np.zeros((v.shape[0], nnodes), dtype=np.float32)
    for t in range(v.shape[0]):
        out[t] = griddata(points, v[t].ravel(), xi, method='nearest')
    return out

print('interpolating ECHAM heat fluxes to FESOM nodes ...')
var92  = to_fesom(ech['var92'].values)    # LW
var95  = to_fesom(ech['var95'].values)    # SW
var111 = to_fesom(ech['var111'].values)   # latent
var120 = to_fesom(ech['var120'].values)   # sensible
ntime = var92.shape[0]

# ---------------------------------------------------------------- FESOM fields
sst = xr.open_dataset(D / 'sst_clim.nc')['sst'].values
sss = xr.open_dataset(D / 'sss_clim.nc')['sss'].values
prec = xr.open_dataset(D / 'prec_clim.nc')['prec'].values
snow = xr.open_dataset(D / 'snow_clim.nc')['snow'].values
evap = xr.open_dataset(D / 'evap_clim.nc')['evap'].values
runoff = xr.open_dataset(D / 'runoff_clim.nc')['runoff'].values
fw = xr.open_dataset(D / 'fw_clim.nc')['fw'].values
aice = xr.open_dataset(D / 'a_ice_clim.nc')['a_ice'].values

# ---------------------------------------------------------------- domains
amax = aice.max(axis=0)
siz_mask = (amax > 0.15) & (fesom_lat_1d < 0)          # seasonal ice zone
so60_mask = (fesom_lat_1d < -60)                        # published domain
open_mask = so60_mask & ~siz_mask                       # <60S but ice-free year-round

print(f'  area <60S            : {areacello_1d[so60_mask].sum():.4e} m2')
print(f'  area seasonal ice zone: {areacello_1d[siz_mask].sum():.4e} m2')
print(f'  area <60S, ice-free   : {areacello_1d[open_mask].sum():.4e} m2')

DOMAINS = {'siz': siz_mask, 'so60': so60_mask, 'open_only': open_mask}

CONFIGS = {
    'total_heat': dict(rsntds=var95, rlntds=var92, hflso=var111, hfsso=var120),
    'turbulent':  dict(rsntds=np.zeros_like(var95), rlntds=np.zeros_like(var92),
                       hflso=var111, hfsso=var120),
    'radiative':  dict(rsntds=var95, rlntds=var92,
                       hflso=np.zeros_like(var111), hfsso=np.zeros_like(var120)),
}

SALT_VARS = [
    'surface_ocean_flux_advective_negative_rhs_evaporation_salt',
    'surface_ocean_flux_advective_negative_rhs_snow_salt',
    'surface_ocean_flux_advective_negative_rhs_sea_ice_melt_salt',
    'surface_ocean_flux_advective_negative_rhs_rain_and_ice_salt',
    'surface_ocean_flux_advective_negative_rhs_rivers_salt',
    'surface_exchange_flux_nonadvective_salt',
]
RHO_WATER = 1000.0
S_ice = 5.0

results = {}
for dom_name, dom_mask in DOMAINS.items():
    print(f'\n--- domain: {dom_name} ---')
    for cfg_name, cfg in CONFIGS.items():
        ds = xr.Dataset()
        ds.coords['time'] = np.arange(ntime)
        ds.coords['xh'] = np.arange(nnodes)
        ds.coords['xq'] = np.arange(nnodes + 1)
        ds.coords['yh'] = [0]
        ds.coords['yq'] = [0, 1]

        ds['tos'] = (['time', 'yh', 'xh'], sst[:, np.newaxis, :]); ds['tos'].attrs['units'] = 'degC'
        ds['sos'] = (['time', 'yh', 'xh'], sss[:, np.newaxis, :]); ds['sos'].attrs['units'] = 'psu'

        area_masked = np.where(dom_mask, areacello_1d, 0.0)
        ds['areacello'] = (['yh', 'xh'], area_masked[np.newaxis, :]); ds['areacello'].attrs['units'] = 'm2'
        ds['lon'] = (['yh', 'xh'], fesom_lon_1d[np.newaxis, :])
        ds['lat'] = (['yh', 'xh'], fesom_lat_1d[np.newaxis, :])

        for k, v in cfg.items():
            ds[k] = (['time', 'yh', 'xh'], v[:, np.newaxis, :])

        prlq = prec[:, np.newaxis, :] * RHO_WATER
        prsn = snow[:, np.newaxis, :] * RHO_WATER
        evs  = evap[:, np.newaxis, :] * RHO_WATER
        friv = runoff[:, np.newaxis, :] * RHO_WATER
        fw2  = -fw[:, np.newaxis, :] * RHO_WATER
        fsitherm = fw2 - evs - prlq - prsn - friv

        ds['prlq'] = (['time', 'yh', 'xh'], prlq)
        ds['prsn'] = (['time', 'yh', 'xh'], prsn)
        ds['evs'] = (['time', 'yh', 'xh'], evs)
        ds['friver'] = (['time', 'yh', 'xh'], friv)
        ds['fsitherm'] = (['time', 'yh', 'xh'], fsitherm)
        ds['wfo'] = (['time', 'yh', 'xh'], fw2)
        for var in ['ficeberg', 'vprec']:
            ds[var] = (['time', 'yh', 'xh'], np.zeros((ntime, 1, nnodes), dtype=np.float32))
        ds['sfdsi'] = (['time', 'yh', 'xh'], fsitherm * S_ice * 0.001)

        grid = xgcm.Grid(ds,
                         coords={'X': {'center': 'xh', 'outer': 'xq'},
                                 'Y': {'center': 'yh', 'outer': 'yq'}},
                         metrics={('X', 'Y'): 'areacello'},
                         boundary={'X': 'extend', 'Y': 'extend'},
                         autoparse_metadata=False)

        bud = xbudget.load_preset_budget(model='MOM6_surface')
        xbudget.collect_budgets(ds, bud)
        dec = xbudget.aggregate(bud, decompose=['surface_exchange_flux', 'advective',
                                                'surface_ocean_flux_advective_negative_rhs'])
        for b in ['heat', 'salt', 'mass']:
            if b in dec and 'lambda' in dec[b]:
                del dec[b]['lambda']

        wm = xwmt.WaterMassTransformations(grid, dec)
        bins = np.arange(0, 40, 0.2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            G = wm.integrate_transformations('sigma2', bins=bins, sum_components=True)
            G.load()

        hv = 'surface_exchange_flux_nonadvective_heat'
        if hv in G.data_vars:
            results[f'{dom_name}_{cfg_name}_heat'] = G[hv]
        if cfg_name == 'total_heat':
            for v in SALT_VARS:
                if v in G.data_vars:
                    results[f'{dom_name}_{v}'] = G[v]
        print(f'  {cfg_name}: done')

out = xr.Dataset()
first = list(results.values())[0]
for c in first.coords:
    out.coords[c] = first.coords[c]
for k, v in results.items():
    out[k] = v
out.attrs['experiment'] = exp.upper()
out.attrs['description'] = ('WMT over seasonal sea ice zone (amax>15%), full <60S domain, '
                            'and ice-free-only sub-domain; climatological monthly means')
out.attrs['siz_area_m2'] = float(areacello_1d[siz_mask].sum())
out.attrs['so60_area_m2'] = float(areacello_1d[so60_mask].sum())
out.attrs['open_area_m2'] = float(areacello_1d[open_mask].sum())
fn = f'wmt_siz_{exp}.nc'
out.to_netcdf(fn)
print(f'\nsaved {fn}')
