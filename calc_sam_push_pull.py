#!/usr/bin/env python
"""
SAM "push and pull" decomposition, following Reviewer 3's suggestion that wind
changes affect both the Ekman divergence (the push, i.e. upwelling of deep water)
and the salt pump via sea ice export (the pull, i.e. dense water formation).

For each experiment and each SAM phase (high minus low, 1.2 sigma threshold on the
JJA-mean Marshall SAM index) we compute:

  PUSH   circumpolar Ekman transport at 60 S, from the zonal wind stress,
         M_Ek = -tau_x / (rho f) integrated zonally  [Sv]
  PULL   northward sea ice area export across 60 S, and the coastal
         (south of 65 S) sea ice growth implied by the freshwater flux  [Sv equivalent]

Outputs sam_push_pull.nc plus a printed table.
"""
import numpy as np
import xarray as xr
from pathlib import Path

EXPS = ['pi', 'mh', 'lig', 'lgm', 'mis']
LABELS = ['PI', 'MH', 'LIG', 'LGM', 'MIS3']
MESH = {'pi': 'mesh_core2', 'mh': 'mesh_core2', 'lig': 'mesh_core2',
        'lgm': 'mesh_glac1d', 'mis': 'mesh_glac1d_38k'}
THRESH = 1.2
RHO0 = 1027.0
OMEGA = 7.292115e-5

# --------------------------------------------------------------- SAM index
def sam_index(exp):
    """Marshall-style SAM: normalised zonal-mean SLP difference 40S minus 65S, JJA."""
    d = xr.open_dataset(f'{exp}/100years/slp_mergetime.nc')
    v = [x for x in d.data_vars if d[x].ndim == 3][0]
    slp = d[v]
    lat = d['lat'].values
    zm = slp.mean('lon')
    i40 = int(np.argmin(np.abs(lat - (-40))))
    i65 = int(np.argmin(np.abs(lat - (-65))))
    p40 = zm.isel(lat=i40).values
    p65 = zm.isel(lat=i65).values
    p40 = (p40 - p40.mean()) / p40.std()
    p65 = (p65 - p65.mean()) / p65.std()
    sam_m = p40 - p65                       # monthly, 1200 values
    sam_m = sam_m.reshape(-1, 12)           # (100 years, 12 months)
    sam_jja = sam_m[:, 5:8].mean(axis=1)    # JJA mean per year
    d.close()
    return sam_jja


def phases(sam):
    m, s = sam.mean(), sam.std()
    hi = np.where(sam > m + THRESH * s)[0]
    lo = np.where(sam < m - THRESH * s)[0]
    return hi, lo


# --------------------------------------------------------------- PUSH: Ekman
def ekman_transport(exp, years_hi, years_lo):
    """Zonally integrated northward Ekman transport at 60 S, JJA, in Sv."""
    d = xr.open_dataset(f'sam_wind/{exp}_taux.nc')
    taux = d['var180']                       # (1200, lat, lon), Pa
    lat = d['lat'].values
    lon = d['lon'].values
    i60 = int(np.argmin(np.abs(lat - (-60))))
    tx = taux.isel(lat=i60).values           # (1200, lon)
    # JJA per year
    tx = tx.reshape(-1, 12, len(lon))[:, 5:8, :].mean(axis=1)   # (100, lon)
    f = 2 * OMEGA * np.sin(np.deg2rad(lat[i60]))
    # zonal integral of -tau_x/(rho f) over the latitude circle
    R = 6.371e6
    dx = np.deg2rad(360.0 / len(lon)) * R * np.cos(np.deg2rad(lat[i60]))
    M = (-tx / (RHO0 * f)) * dx              # m3/s per grid cell
    M = M.sum(axis=1) / 1e6                  # Sv per year
    d.close()
    return M[years_hi].mean(), M[years_lo].mean(), M


# --------------------------------------------------------------- PULL: sea ice
def ice_export_and_growth(exp, years_hi, years_lo):
    """
    Northward sea ice area flux proxy across 60 S is not directly available, so we
    use two robust quantities that the model does provide:
      (a) sea ice area south of 60 S  (a_ice weighted by node area)
      (b) coastal net freshwater flux south of 65 S, whose negative part is the
          brine/salt input associated with net ice growth, expressed in Sv.
    """
    m = xr.open_dataset(f'/home/a/a270064/bb1029/inputs/{MESH[exp]}/fesom.mesh.diag.nc')
    lat = np.degrees(m['nodes'].isel(n2=1).values)
    area = (m['nod_area'].isel(nz=0).values if 'nz' in m['nod_area'].dims
            else m['nod_area'].isel(nl=0).values)

    ai = xr.open_dataset(f'{exp}/100years/a_ice_mergetime.nc')['a_ice'].values
    fw = xr.open_dataset(f'{exp}/100years/fw_mergetime.nc')['fw'].values

    ai = ai.reshape(-1, 12, ai.shape[-1])[:, 5:8, :].mean(axis=1)   # (100, nod)
    fw = fw.reshape(-1, 12, fw.shape[-1])[:, 5:8, :].mean(axis=1)

    so = lat < -60
    coast = lat < -65

    ice_area = (ai[:, so] * area[so]).sum(axis=1) / 1e12            # 10^6 km2
    # Sign convention check (PI, coastal <65S): JJA = +0.164 Sv, DJF = -0.470 Sv.
    # Sea ice grows in JJA, so a POSITIVE value here corresponds to the ocean LOSING
    # freshwater, i.e. brine input. This matches the WMT pipeline, which negates fw
    # (fw2 = -fw*rho) before forming the salt flux. We therefore report it as
    # "coastal brine-equivalent freshwater loss", positive = stronger brine input.
    fw_coast = (fw[:, coast] * area[coast]).sum(axis=1) / 1e6       # Sv
    m.close()
    return (ice_area[years_hi].mean(), ice_area[years_lo].mean(), ice_area,
            fw_coast[years_hi].mean(), fw_coast[years_lo].mean(), fw_coast)


rows = []
store = {}
for e, lab in zip(EXPS, LABELS):
    sam = sam_index(e)
    hi, lo = phases(sam)
    ek_hi, ek_lo, ek_all = ekman_transport(e, hi, lo)
    ia_hi, ia_lo, ia_all, fw_hi, fw_lo, fw_all = ice_export_and_growth(e, hi, lo)
    rows.append(dict(exp=lab, n_hi=len(hi), n_lo=len(lo),
                     ekman_hi=ek_hi, ekman_lo=ek_lo, d_ekman=ek_hi - ek_lo,
                     ice_hi=ia_hi, ice_lo=ia_lo, d_ice=ia_hi - ia_lo,
                     fwc_hi=fw_hi, fwc_lo=fw_lo, d_fwc=fw_hi - fw_lo))
    store[e] = dict(sam=sam, ekman=ek_all, ice=ia_all, fw=fw_all, hi=hi, lo=lo)

print()
print('SAM push and pull decomposition, JJA, high minus low SAM (1.2 sigma)')
print('%-6s %4s %4s | %9s %9s | %9s %9s | %9s %9s' %
      ('exp', 'nhi', 'nlo', 'Ek_hi', 'dEk(Sv)', 'ice_hi', 'dice', 'fwc_hi', 'dfwc(Sv)'))
print('-' * 92)
for r in rows:
    print('%-6s %4d %4d | %9.2f %9.3f | %9.2f %9.3f | %9.4f %9.4f' %
          (r['exp'], r['n_hi'], r['n_lo'], r['ekman_hi'], r['d_ekman'],
           r['ice_hi'], r['d_ice'], r['fwc_hi'], r['d_fwc']))

print()
print('Ek     = northward Ekman transport at 60S [Sv]  (the "push")')
print('ice    = JJA sea ice area south of 60S [10^6 km2]')
print('fwc    = coastal (<65S) brine-equivalent freshwater loss [Sv]')
print('         (the "pull": POSITIVE dfwc under high SAM = stronger brine input)')

np.save('sam_push_pull.npy', store, allow_pickle=True)
import json
Path('sam_push_pull.json').write_text(json.dumps(rows, indent=2, default=float))
print('\nsaved sam_push_pull.npy and sam_push_pull.json')
