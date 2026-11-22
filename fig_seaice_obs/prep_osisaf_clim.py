#!/usr/bin/env python
"""
OSI SAF sea ice concentration CDR v3.0 (OSI-450-a), Southern Hemisphere, daily
25 km EASE2 grid -> monthly means for every year 1979-2008, plus daily extent
and area statistics. Output: fig_seaice_obs/osisaf_sh_monthly_1979_2008.nc

Before Aug 1987 the record has data every other day, and some months have
gaps (Dec 1987 has 2 days, Jan 1988 19). Each monthly mean averages whatever
days exist, pixel by pixel; n_days records how many went in.
"""
import os
import sys
import glob
import numpy as np
import netCDF4
from multiprocessing import Pool

ROOT = '/pool/data/ICDC/ice_and_snow/osisaf_iceconc/DATA'
PAT = '{root}/{y:04d}/{m:02d}/ice_conc_sh_ease2-250_cdr-v3p0_{y:04d}{m:02d}??1200.nc'
Y0, Y1 = 1979, 2008
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'osisaf_sh_monthly_1979_2008.nc')
CELL_KM2 = 625.0   # 25 km x 25 km on an equal-area grid; checked against xc below


def one_year(y):
    """Monthly mean concentration (fraction) and daily extent/area stats."""
    mean = np.full((12, 432, 432), np.nan, np.float32)
    ndays = np.zeros(12, np.int16)
    dext = np.full(12, np.nan)      # mean of daily extents, 1e6 km2
    dare = np.full(12, np.nan)      # mean of daily areas, 1e6 km2
    for m in range(1, 13):
        files = sorted(glob.glob(PAT.format(root=ROOT, y=y, m=m)))
        if not files:
            continue
        s = np.zeros((432, 432))
        n = np.zeros((432, 432))
        ext, are = [], []
        for f in files:
            with netCDF4.Dataset(f) as d:
                c = d.variables['ice_conc'][0, :, :]   # masked, % after scale
            c = np.ma.filled(c.astype(np.float64), np.nan) / 100.0
            c = np.clip(c, 0.0, 1.0)
            ok = np.isfinite(c)
            s[ok] += c[ok]
            n[ok] += 1
            ext.append(np.sum(c[ok] >= 0.15) * CELL_KM2 / 1e6)
            are.append(np.sum(c[ok]) * CELL_KM2 / 1e6)
        with np.errstate(invalid='ignore', divide='ignore'):
            mean[m - 1] = np.where(n > 0, s / n, np.nan)
        ndays[m - 1] = len(files)
        dext[m - 1] = np.mean(ext)
        dare[m - 1] = np.mean(are)
    print(y, 'days per month', list(ndays), flush=True)
    return y, mean, ndays, dext, dare


def main():
    f0 = sorted(glob.glob(PAT.format(root=ROOT, y=1990, m=9)))[0]
    with netCDF4.Dataset(f0) as d:
        xc = d.variables['xc'][:].data
        yc = d.variables['yc'][:].data
        lat = d.variables['lat'][:].data
        lon = d.variables['lon'][:].data
        attrs = {k: d.getncattr(k) for k in
                 ('title', 'product_id', 'product_version', 'doi', 'references',
                  'contributor_name', 'institution')}
    dx = np.unique(np.round(np.diff(xc), 6))
    dy = np.unique(np.round(np.diff(yc), 6))
    print('grid spacing xc', dx, 'yc', dy, 'km -> cell area',
          abs(dx[0] * dy[0]), 'km2')
    assert abs(abs(dx[0] * dy[0]) - CELL_KM2) < 1e-6

    years = list(range(Y0, Y1 + 1))
    nproc = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    with Pool(nproc) as p:
        res = sorted(p.map(one_year, years), key=lambda r: r[0])

    ny = len(years)
    conc = np.stack([r[1] for r in res])            # (ny,12,y,x)
    ndays = np.stack([r[2] for r in res])
    dext = np.stack([r[3] for r in res])
    dare = np.stack([r[4] for r in res])

    with netCDF4.Dataset(OUT, 'w') as o:
        o.createDimension('year', ny)
        o.createDimension('month', 12)
        o.createDimension('yc', 432)
        o.createDimension('xc', 432)
        v = o.createVariable('year', 'i4', ('year',)); v[:] = years
        v = o.createVariable('month', 'i4', ('month',)); v[:] = np.arange(1, 13)
        v = o.createVariable('xc', 'f8', ('xc',)); v[:] = xc; v.units = 'km'
        v = o.createVariable('yc', 'f8', ('yc',)); v[:] = yc; v.units = 'km'
        v = o.createVariable('lat', 'f4', ('yc', 'xc')); v[:] = lat
        v.units = 'degrees_north'
        v = o.createVariable('lon', 'f4', ('yc', 'xc')); v[:] = lon
        v.units = 'degrees_east'
        v = o.createVariable('conc', 'f4', ('year', 'month', 'yc', 'xc'),
                             zlib=True, complevel=4, fill_value=np.nan)
        v[:] = conc
        v.units = '1'
        v.long_name = 'monthly mean sea ice concentration (fraction), mean of available days'
        v = o.createVariable('n_days', 'i2', ('year', 'month')); v[:] = ndays
        v = o.createVariable('daily_extent_mean', 'f8', ('year', 'month'))
        v[:] = dext; v.units = '1e6 km2'
        v.long_name = 'monthly mean of daily extents (conc >= 15%)'
        v = o.createVariable('daily_area_mean', 'f8', ('year', 'month'))
        v[:] = dare; v.units = '1e6 km2'
        o.cell_area_km2 = CELL_KM2
        o.grid = '+proj=laea +lon_0=0 +datum=WGS84 +ellps=WGS84 +lat_0=-90.0'
        for k, val in attrs.items():
            o.setncattr('source_' + k, val)
        o.history = 'fig_seaice_obs/prep_osisaf_clim.py'
    print('wrote', OUT)


if __name__ == '__main__':
    main()
