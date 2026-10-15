#!/usr/bin/env python
"""
Water mass transformation from ORAS5, over the seasonal sea ice zone.

Xiaoxu asked what the observational-side transformation looks like if we build
it from a reanalysis anyway, accepting that the sea ice thermodynamic term is
missing. ORAS5 is the only product on Levante carrying surface fluxes for this
region: sohefldo (net downward heat, W/m2) and sowaflup (net upward water,
kg/m2/s). Its water flux was tested earlier and shows no freezing signal, so it
is evaporation minus precipitation minus runoff only.

That omission is the point of the exercise rather than a flaw in it. Comparing
this curve with our model's, and with the model's own heat-and-other-freshwater
curve with the ice term removed, shows how much of the dense-class
transformation the missing term accounts for.

Method follows the model pipeline: bin the surface density flux into sigma_2
classes and divide by the class width.

    F(sigma) = (1/dsigma) * sum over cells in class of
               [ -alpha/cp * Q  +  beta*S/(1-S) * FW ] * area

Q is the net downward heat flux and FW the net freshwater input to the ocean,
so sowaflup is negated. alpha, beta and cp come from the EOS-80 routines in
seawater, consistent with the rest of this project.

Period: JJA and annual, 1999-2018, ensemble member opa0. The record
on Levante covers 1979-2018; this is its Argo-rich half, where the
assimilation actually constrains the Southern Ocean, and it overlaps
the periods Pellichero et al. use.
Domain: everything south of 60 S, the integration domain of the paper itself,
rather than the sea ice sector used for the Pellichero comparison.
"""
import glob
import numpy as np
import xarray as xr
import seawater as sw

ROOT = '/pool/data/ICDC/ocean_syntheses/oras5/ORCA025'
YEARS = range(1999, 2019)
OUT = 'wmt_oras5_so60.nc'

# the same sigma_2 grid the model diagnostics use
SIG = np.arange(30.0, 39.0 + 1e-9, 0.2)
CENTRES = 0.5 * (SIG[:-1] + SIG[1:])
DSIG = 0.2

CP = 3996.0          # J/(kg K), as in the model's WMT code


def read(var, year, month, grid='grid_T'):
    f = f'{ROOT}/{var}/opa0/{var}_ORAS5_1m_{year}{month:02d}_{grid}_02.nc'
    hits = glob.glob(f)
    if not hits:
        return None
    d = xr.open_dataset(hits[0], decode_times=False)
    a = d[var].isel(time_counter=0).values.astype('f8')
    d.close()
    return np.where(np.isfinite(a) & (np.abs(a) < 1e10), a, np.nan)


def area_and_latlon():
    d = xr.open_dataset(f'{ROOT}/mesh/mesh_mask.nc', decode_times=False)
    e1 = d['e1t'].squeeze().values
    e2 = d['e2t'].squeeze().values
    lat = d['gphit'].squeeze().values
    d.close()
    return e1 * e2, lat


def south_of_60(lat):
    """The paper's own domain: everything south of 60S."""
    m = lat < -60.0
    print('domain south of 60S: %d cells' % m.sum())
    return m


def transform(months, label, siz, area):
    """Bin the surface density flux into sigma_2 classes."""
    heat = np.zeros(len(CENTRES))
    fresh = np.zeros(len(CENTRES))
    n = 0
    for y in YEARS:
        for mo in months:
            Q = read('sohefldo', y, mo)          # W/m2, downward positive
            W = read('sowaflup', y, mo)          # kg/m2/s, upward positive
            T = read('sosstsst', y, mo)
            S = read('sosaline', y, mo)
            if any(v is None for v in (Q, W, T, S)):
                continue
            ok = siz & np.isfinite(Q) & np.isfinite(W) & np.isfinite(T) \
                 & np.isfinite(S) & (S > 20)
            t, s = T[ok], S[ok]
            sig2 = sw.dens(s, t, 2000) - 1000
            alpha = sw.alpha(s, t, 2000)
            beta = sw.beta(s, t, 2000)
            # density flux, positive = densification
            fh = -alpha / CP * Q[ok]
            # sowaflup is positive upward, i.e. a loss of freshwater from the
            # ocean, which makes the surface denser. The density flux is
            # therefore +beta*S/(1-S) * W with W as stored.
            fw = beta * s / (1.0 - s / 1000.0) * W[ok]
            a = area[ok]
            idx = np.digitize(sig2, SIG) - 1
            good = (idx >= 0) & (idx < len(CENTRES))
            np.add.at(heat, idx[good], (fh * a)[good])
            np.add.at(fresh, idx[good], (fw * a)[good])
            n += 1
    if n == 0:
        raise SystemExit('no ORAS5 months read')
    heat /= n * DSIG * 1e6        # m3/s -> Sv
    fresh /= n * DSIG * 1e6
    print('%s: %d months, peak total %.2f Sv at sigma2 %.2f'
          % (label, n, np.nanmax(heat + fresh),
             CENTRES[np.nanargmax(heat + fresh)]))
    return heat, fresh


area, lat = area_and_latlon()
siz = south_of_60(lat)

out = xr.Dataset(coords={'sigma2': CENTRES})
for months, label in ([6, 7, 8], 'jja'), (list(range(1, 13)), 'annual'):
    h, f = transform(months, label, siz, area)
    out['heat_' + label] = ('sigma2', h)
    out['fresh_' + label] = ('sigma2', f)
    out['total_' + label] = ('sigma2', h + f)

out.attrs['note'] = ('ORAS5 1999-2018, member opa0, south of 60S, the '
                     "paper's own domain. sowaflup carries no sea ice "
                     'thermodynamic term, so the freshwater component is '
                     'E-P-R only.')
out.to_netcdf(OUT)
print('wrote', OUT)
