#!/usr/bin/env python3
"""PMIP4 LGM vs piControl / midHolocene Southern Hemisphere westerlies (Reviewer 3, Gray et al. 2023).

Run (foreground, serially):
    module load python3/2022.01-gcc-11.2.0
    python3 -s pmip_westerlies.py compute AWI-ESM-1-1-LR     # one model -> profiles/<model>_*.nc + _metrics.json
    python3 -s pmip_westerlies.py ours                       # our AWI-ESM runs from the project's ECHAM output
    python3 -s pmip_westerlies.py summary                    # results.csv + summary.txt from all *_metrics.json

Latitudes are signed (SH negative). Shift = |lat_LGM| - |lat_ref| in degrees, positive = poleward.
Metric u10jja: JJA (months 6,7,8) all-longitude zonal-mean uas, max within 30-70S (the paper's metric).
Metric tauu_ocn: annual-mean zonal-mean tauu over ocean points (sftlf < 50 %, experiment's own sftlf).
Metric ua850: annual-mean all-longitude zonal-mean ua at 850 hPa (below-ground cells excluded).
"""
import sys, os, re, glob, json, csv, warnings
import numpy as np
import xarray as xr

ROOT = '/work/ik1017/CMIP6/data/CMIP6'
PROJ = '/work/ba1066/a270064/cc_projects/aabw_5exps'
OUT = os.path.dirname(os.path.abspath(__file__))
PROF = os.path.join(OUT, 'profiles')
NYEARS = 100
WIN = (30.0, 70.0)
EXPS = ['piControl', 'midHolocene', 'lgm']
SHORT = {'piControl': 'pi', 'midHolocene': 'mh', 'lgm': 'lgm'}
ACT = {'piControl': 'CMIP', 'midHolocene': 'PMIP', 'lgm': 'PMIP'}
MODELS = {
    'AWI-ESM-1-1-LR': ('AWI', 'r1i1p1f1'),
    'INM-CM4-8': ('INM', 'r1i1p1f1'),
    'MIROC-ES2L': ('MIROC', 'r1i1p1f2'),
    'MPI-ESM1-2-LR': ('MPI-M', 'r1i1p1f1'),
}
THR = {'uas': 1000.0, 'ua': 1000.0, 'tauu': 100.0}   # |x| above this = undeclared fill value
METRICS = ['u10jja', 'tauu_ocn', 'ua850']


# ---------------------------------------------------------------- peak finding
# Same math as historical_zone/src/common/latfind.py (peak + _parabolic_peak).
def _parabolic_peak(x, y, i):
    if i <= 0 or i >= len(x) - 1:
        return x[i], y[i]
    x0, x1, x2 = x[i - 1], x[i], x[i + 1]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if denom == 0:
        return x1, y1
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    if a == 0:
        return x1, y1
    xv = -b / (2 * a)
    if xv < min(x0, x2) or xv > max(x0, x2):
        return x1, y1
    yv = a * xv * xv + b * xv + (y1 - a * x1 * x1 - b * x1)
    return xv, yv


def sh_peak(lat, y):
    """Max of y within 30-70S. Returns dict(lat_fit, val_fit, lat_grid, val_grid, flag)."""
    lat = np.asarray(lat, float)
    y = np.asarray(y, float)
    a = np.abs(lat)
    m = (lat < 0) & (a >= WIN[0]) & (a <= WIN[1]) & np.isfinite(y)
    la, ya = lat[m], y[m]
    if la.size < 3:
        return dict(lat_fit=np.nan, val_fit=np.nan, lat_grid=np.nan, val_grid=np.nan, flag='empty')
    o = np.argsort(np.abs(la))
    la, ya = la[o], ya[o]
    i = int(np.argmax(ya))
    xv, yv = _parabolic_peak(la, ya, i)
    flag = 'edge' if i in (0, len(la) - 1) else 'ok'
    return dict(lat_fit=float(xv), val_fit=float(yv), lat_grid=float(la[i]), val_grid=float(ya[i]), flag=flag)


# ---------------------------------------------------------------- file discovery
def latest_dir(model, exp, table, var):
    inst, member = MODELS[model]
    base = f'{ROOT}/{ACT[exp]}/{inst}/{model}/{exp}/{member}/{table}/{var}'
    grids = sorted(glob.glob(base + '/*'))
    if not grids:
        return None
    grid = [g for g in grids if g.endswith('/gn')] or grids
    grid = grid[0]
    vers = sorted(os.listdir(grid))          # lexicographically latest version
    return os.path.join(grid, vers[-1])


def file_years(f):
    m = re.search(r'_(\d+)-(\d+)\.nc$', os.path.basename(f))
    return int(m.group(1)[:-2]), int(m.group(2)[:-2])


def select_files(d, nyears=NYEARS):
    fs = sorted(glob.glob(d + '/*.nc'))
    yrs = [file_years(f) for f in fs]
    y_end = max(b for _, b in yrs)
    y0 = max(min(a for a, _ in yrs), y_end - nyears + 1)
    return [f for f, (a, b) in zip(fs, yrs) if b >= y0], y0, y_end


# ---------------------------------------------------------------- accumulation
def monthly_clim(files, var, y0, y1, plev=None):
    """Monthly climatology (12, lat, lon) over years y0..y1, plus bookkeeping."""
    s = c = None
    seen = set()
    nbad = 0
    lat = lon = None
    for f in files:
        with xr.open_dataset(f, use_cftime=True) as ds:
            da = ds[var]
            if plev is not None:
                da = da.sel(plev=plev, method='nearest')
                assert abs(float(da.plev) - plev) < 1.0, (f, float(da.plev))
            yr = da['time'].dt.year.values
            mo = da['time'].dt.month.values
            keep = np.where((yr >= y0) & (yr <= y1))[0]
            if keep.size == 0:
                continue
            x = da.isel(time=keep).values.astype('f8')
            if lat is None:
                lat, lon = ds['lat'].values.astype('f8'), ds['lon'].values.astype('f8')
        bad = ~np.isfinite(x) | (np.abs(x) > THR[var])
        nbad += int((np.isfinite(x) & (np.abs(x) > THR[var])).sum())
        x[bad] = np.nan
        if s is None:
            s = np.zeros((12,) + x.shape[1:])
            c = np.zeros((12,) + x.shape[1:])
        for k, (yy, mm) in enumerate(zip(yr[keep], mo[keep])):
            if (yy, mm) in seen:      # duplicate time step in overlapping files
                continue
            seen.add((yy, mm))
            v = x[k]
            ok = np.isfinite(v)
            s[mm - 1][ok] += v[ok]
            c[mm - 1][ok] += 1
    nmon = len(seen)
    years = sorted({yy for yy, _ in seen})
    with np.errstate(invalid='ignore', divide='ignore'):
        clim = np.where(c > 0, s / c, np.nan)
    info = dict(y0=years[0], y1=years[-1], nyears=len(years), nmonths=nmon,
                complete=(nmon == 12 * len(years)), n_fill_masked=nbad)
    return lat, lon, clim, info


def zm(field, mask=None):
    f = field.copy()
    if mask is not None:
        f[~mask] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(f, axis=-1), np.isfinite(f).sum(axis=-1)


def ocean_mask(model, exp, lat, lon):
    d = latest_dir(model, exp, 'fx', 'sftlf')
    if d is None:
        return None, None
    f = sorted(glob.glob(d + '/*.nc'))[0]
    with xr.open_dataset(f) as ds:
        sl = ds['sftlf']
        if not (np.allclose(sl.lat.values, lat) and np.allclose(sl.lon.values, lon)):
            sl = sl.sel(lat=lat, lon=lon, method='nearest')
        sl = sl.values.astype('f8')
    return sl < 50.0, f


# ---------------------------------------------------------------- per model
def compute(model):
    os.makedirs(PROF, exist_ok=True)
    res = dict(model=model, source='PMIP4', member=MODELS[model][1], exps={})
    for exp in EXPS:
        e = {}
        prof = {}
        lat_ref = None
        for var, metric in [('uas', 'u10jja'), ('tauu', 'tauu_ocn'), ('ua', 'ua850')]:
            d = latest_dir(model, exp, 'Amon', var)
            if d is None:
                e[metric] = dict(missing=f'no Amon/{var}')
                print(f'  {model} {exp} {var}: MISSING')
                continue
            files, y0, y1 = select_files(d)
            lat, lon, clim, info = monthly_clim(files, var, y0, y1, plev=85000.0 if var == 'ua' else None)
            info['dir'] = d
            if lat_ref is None:
                lat_ref = lat
            assert np.allclose(lat, lat_ref)
            if var == 'uas':
                fld = clim[[5, 6, 7]].mean(axis=0)         # JJA = months 6,7,8
                z, nv = zm(fld)
                prof['uas_jja_zm'] = z
                prof['uas_ann_zm'] = zm(clim.mean(axis=0))[0]
            elif var == 'tauu':
                fld = clim.mean(axis=0)
                om, sf = ocean_mask(model, exp, lat, lon)
                info['sftlf'] = sf
                z, nv = zm(fld, om)
                prof['tauu_ann_zm_ocean'] = z
                prof['tauu_ann_zm_all'] = zm(fld)[0]
                prof['tauu_n_ocean'] = nv.astype('f8')
            else:
                fld = clim.mean(axis=0)
                z, nv = zm(fld)
                prof['ua850_ann_zm'] = z
                prof['ua850_n_valid'] = nv.astype('f8')
            info['field_min'] = float(np.nanmin(fld))
            info['field_max'] = float(np.nanmax(fld))
            pk = sh_peak(lat, z)
            info.update(pk)
            e[metric] = info
            print(f'  {model} {exp} {var}: years {info["y0"]}-{info["y1"]} n={info["nyears"]} '
                  f'complete={info["complete"]} fill={info["n_fill_masked"]} '
                  f'range=[{info["field_min"]:.3g},{info["field_max"]:.3g}] '
                  f'peak lat_fit={pk["lat_fit"]:.2f} lat_grid={pk["lat_grid"]:.2f} val={pk["val_grid"]:.4g} {pk["flag"]}',
                  flush=True)
        ds = xr.Dataset({k: ('lat', v) for k, v in prof.items()}, coords={'lat': lat_ref})
        ds.attrs['model'] = model
        ds.attrs['experiment'] = exp
        ds.to_netcdf(os.path.join(PROF, f'{model}_{exp}.nc'))
        res['exps'][exp] = e
    with open(os.path.join(PROF, f'{model}_metrics.json'), 'w') as fh:
        json.dump(res, fh, indent=1, default=_np2py)


def _np2py(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


# ---------------------------------------------------------------- our AWI-ESM runs
def ours():
    """Our AWI-ESM (ECHAM6 T63) PI/MH/LGM from the project directory, same metrics."""
    os.makedirs(PROF, exist_ok=True)
    res = dict(model='AWI-ESM (this study)', source='this study', member='-', exps={})
    for exp, tag in [('piControl', 'pi'), ('midHolocene', 'mh'), ('lgm', 'lgm')]:
        e = {}
        fc = f'{PROJ}/{tag}/echam_clim.nc'
        with xr.open_dataset(fc, decode_times=False) as ds:
            tv = ds['time'].values
            assert len(tv) == 12
            mon = [int(str(int(t))[-4:-2]) for t in tv]           # 'day as %Y%m%d.%f'
            assert mon == list(range(1, 13)), mon
            lat = ds['lat'].values.astype('f8')
            u = ds['var165'].values.astype('f8')                     # 10 m u, monthly clim
            slm = ds['var172'].isel(time=0).values.astype('f8')     # land-sea mask (1 = land)
        fld = u[[5, 6, 7]].mean(axis=0)
        z, _ = zm(fld)
        pk = sh_peak(lat, z)
        e['u10jja'] = dict(pk, y0='clim', y1='clim', nyears='echam_clim', dir=fc,
                           field_min=float(np.nanmin(fld)), field_max=float(np.nanmax(fld)))
        ft = f'{PROJ}/sam_wind/{tag}_taux.nc'
        with xr.open_dataset(ft, decode_times=False) as ds:
            t = ds['var180'].values.astype('f8')
            assert np.allclose(ds['lat'].values, lat)
        nt = t.shape[0]
        fld = t.mean(axis=0)
        zt, nv = zm(fld, slm < 0.5)
        pk = sh_peak(lat, zt)
        e['tauu_ocn'] = dict(pk, y0='all', y1='all', nyears=nt // 12, nmonths=nt, dir=ft,
                             sftlf=f'{fc}:var172<0.5',
                             field_min=float(np.nanmin(fld)), field_max=float(np.nanmax(fld)))
        e['ua850'] = dict(missing='not computed for our runs')
        for k in ('u10jja', 'tauu_ocn'):
            p = e[k]
            print(f'  ours {exp} {k}: lat_fit={p["lat_fit"]:.2f} lat_grid={p["lat_grid"]:.2f} '
                  f'val_grid={p["val_grid"]:.4g} {p["flag"]} range=[{p["field_min"]:.3g},{p["field_max"]:.3g}]')
        xr.Dataset({'uas_jja_zm': ('lat', z), 'tauu_ann_zm_ocean': ('lat', zt)},
                   coords={'lat': lat}).to_netcdf(os.path.join(PROF, f'AWI-ESM_this_study_{exp}.nc'))
        res['exps'][exp] = e
    with open(os.path.join(PROF, 'AWI-ESM_this_study_metrics.json'), 'w') as fh:
        json.dump(res, fh, indent=1, default=_np2py)


# ---------------------------------------------------------------- summary
def _get(e, exp, metric, key):
    try:
        v = e['exps'][exp][metric].get(key, np.nan)
        return np.nan if v is None else v
    except KeyError:
        return np.nan


PROFVAR = {'u10jja': 'uas_jja_zm', 'tauu_ocn': 'tauu_ann_zm_ocean', 'ua850': 'ua850_ann_zm'}


def centroid(pref, exp, metric):
    """Supplementary: cos(lat)-weighted centroid latitude of the positive part of the profile, 30-70S."""
    f = os.path.join(PROF, f'{pref}_{exp}.nc')
    if not os.path.exists(f):
        return np.nan
    with xr.open_dataset(f) as ds:
        if PROFVAR[metric] not in ds:
            return np.nan
        lat = ds['lat'].values
        y = ds[PROFVAR[metric]].values
    m = (lat <= -WIN[0]) & (lat >= -WIN[1]) & np.isfinite(y) & (y > 0)
    w = y[m] * np.cos(np.deg2rad(lat[m]))
    return float((lat[m] * w).sum() / w.sum()) if w.sum() > 0 else np.nan


def summary():
    order = list(MODELS) + ['AWI-ESM_this_study']
    recs = []
    for m in order:
        f = os.path.join(PROF, f'{m}_metrics.json')
        if os.path.exists(f):
            recs.append((m, json.load(open(f))))
    rows = []
    for pref, r in recs:
        row = dict(model=r['model'], source=r['source'], member=r['member'])
        for exp in EXPS:
            u = r['exps'][exp]['u10jja']
            row[f'years_{SHORT[exp]}'] = f'{u.get("y0")}-{u.get("y1")}'
            row[f'nyears_{SHORT[exp]}'] = u.get('nyears')
        for metric in METRICS:
            for exp in EXPS:
                s = SHORT[exp]
                for key, col in [('lat_fit', 'lat'), ('lat_grid', 'latgrid'), ('val_grid', 'val'),
                                 ('val_fit', 'valfit'), ('flag', 'flag')]:
                    v = _get(r, exp, metric, key)
                    row[f'{metric}_{col}_{s}'] = v
                row[f'{metric}_cen_{s}'] = centroid(pref, exp, metric)
                ye = r['exps'][exp].get(metric, {})
                if 'y0' in ye and r['source'] == 'PMIP4':
                    yu = r['exps'][exp]['u10jja']
                    if (ye['y0'], ye['y1']) != (yu['y0'], yu['y1']):
                        row.setdefault('years_note', '')
                        row['years_note'] += f'{metric} {s}: {ye["y0"]}-{ye["y1"]}; '
            for a, b in [('lgm', 'pi'), ('lgm', 'mh')]:
                la, lb = row[f'{metric}_lat_{a}'], row[f'{metric}_lat_{b}']
                ga, gb = row[f'{metric}_latgrid_{a}'], row[f'{metric}_latgrid_{b}']
                va, vb = row[f'{metric}_val_{a}'], row[f'{metric}_val_{b}']
                row[f'{metric}_shift_{a}_{b}'] = abs(la) - abs(lb)
                row[f'{metric}_shiftgrid_{a}_{b}'] = abs(ga) - abs(gb)
                row[f'{metric}_dval_{a}_{b}'] = va - vb
                row[f'{metric}_pct_{a}_{b}'] = 100.0 * (va - vb) / vb
                row[f'{metric}_censhift_{a}_{b}'] = abs(row[f'{metric}_cen_{a}']) - abs(row[f'{metric}_cen_{b}'])
        rows.append(row)
    cols = []
    for row in rows:
        for k in row:
            if k not in cols:
                cols.append(k)
    with open(os.path.join(OUT, 'results.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in row.items()})

    lines = ['Shift = |lat_LGM| - |lat_ref| (deg), positive = poleward. lat = parabolic fit; '
             'grid = raw grid-row shift. Strength from grid-max value.', '']
    pm = [r for r in rows if r['source'] == 'PMIP4']
    for metric in METRICS:
        for a, b in [('lgm', 'pi'), ('lgm', 'mh')]:
            lines.append(f'== {metric}  {a.upper()} - {b.upper()}')
            for r in rows:
                lines.append(f'  {r["model"]:24s} ref {r[f"{metric}_lat_{b}"]:7.2f} {r[f"{metric}_val_{b}"]:8.4g}  '
                             f'lgm {r[f"{metric}_lat_{a}"]:7.2f} {r[f"{metric}_val_{a}"]:8.4g}  '
                             f'shift {r[f"{metric}_shift_{a}_{b}"]:+6.2f} (grid {r[f"{metric}_shiftgrid_{a}_{b}"]:+6.2f})  '
                             f'dval {r[f"{metric}_dval_{a}_{b}"]:+8.4f}  pct {r[f"{metric}_pct_{a}_{b}"]:+6.1f}%  '
                             f'centroid shift {r[f"{metric}_censhift_{a}_{b}"]:+6.2f}')
            sh = np.array([r[f'{metric}_shift_{a}_{b}'] for r in pm], float)
            pc = np.array([r[f'{metric}_pct_{a}_{b}'] for r in pm], float)
            ok = np.isfinite(sh) & np.isfinite(pc)
            if ok.any():
                lines.append(f'  PMIP4 n={ok.sum()}: shift mean {sh[ok].mean():+.2f} range [{sh[ok].min():+.2f}, {sh[ok].max():+.2f}]; '
                             f'poleward {int((sh[ok] > 0).sum())} equatorward {int((sh[ok] < 0).sum())} zero {int((sh[ok] == 0).sum())}; '
                             f'pct mean {pc[ok].mean():+.1f}% range [{pc[ok].min():+.1f}, {pc[ok].max():+.1f}]%; '
                             f'stronger {int((pc[ok] > 0).sum())} weaker {int((pc[ok] < 0).sum())}')
                cs = np.array([r[f'{metric}_censhift_{a}_{b}'] for r in pm], float)[ok]
                lines.append(f'  PMIP4 centroid shift mean {cs.mean():+.2f} range [{cs.min():+.2f}, {cs.max():+.2f}]; '
                             f'poleward {int((cs > 0).sum())} equatorward {int((cs < 0).sum())}')
            lines.append('')
    txt = '\n'.join(lines)
    open(os.path.join(OUT, 'summary.txt'), 'w').write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'compute':
        for m in sys.argv[2:] or list(MODELS):
            print(f'== {m}', flush=True)
            compute(m)
    elif cmd == 'ours':
        ours()
    elif cmd == 'summary':
        summary()
