#!/usr/bin/env python
"""Fig R2.2: JJA high-minus-low SAM anomaly of the heat- and freshwater-driven
surface density flux on ONE colour scale, plus area statistics (numbers.txt).

Conversion, signs and pitfalls are documented in CLAUDE.md in this folder.
The freshwater term carries RHO_FW so that both terms are in kg m-2 s-1; the
JJA figure script of manuscript Fig. 4 leaves that factor out (1000x too small).

Usage (login node, foreground):
    python3 plot_sam_density_tendency.py            # figure + numbers.txt
    python3 plot_sam_density_tendency.py --validate # recompute 1.0 sigma, compare with files
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import gsw
import netCDF4
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.ticker as mticker
import cartopy.crs as ccrs

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
COMP = BASE / 'composite_sam'

EXPS = ['pi', 'mh', 'lig', 'lgm', 'mis']
LABELS = {'pi': 'PI', 'mh': 'MH', 'lig': 'LIG', 'lgm': 'LGM', 'mis': 'MIS3'}
MESH = {'pi': 'mesh_core2', 'mh': 'mesh_core2', 'lig': 'mesh_core2',
        'lgm': 'mesh_glac1d', 'mis': 'mesh_glac1d_38k'}
MESH_ROOT = Path('/home/a/a270064/bb1029/inputs')

JJA_IDX = [5, 6, 7]          # Jun, Jul, Aug in the 12-step climatologies
JJA_MONTHS = [6, 7, 8]
SEC_PER_MONTH = 30.0 * 86400.0
RHO_FW = 1000.0              # same factor as wfo = -fw * 1000 in the WMT pipeline
MMDAY_TO_MS = 1.0 / (86400.0 * 1000.0)
ICE_THR = 0.15
LAT_SO = -50.0
UNIT = 'kg m$^{-2}$ month$^{-1}$'
UNIT_TXT = 'kg m-2 month-1'


# ----------------------------------------------------------------- inputs
def load_mesh(exp):
    m = xr.open_dataset(MESH_ROOT / MESH[exp] / 'fesom.mesh.diag.nc')
    lon = np.degrees(m['nodes'].isel(n2=0).values)       # radians in file
    lat = np.degrees(m['nodes'].isel(n2=1).values)
    na = m['nod_area']
    area = na.isel({na.dims[0]: 0}).values               # nl or nz
    elem = m['elem'].values.T - 1                        # 1-based in file
    m.close()
    return lon, lat, area, elem


def sam_years(exp, thr):
    d = xr.open_dataset(COMP / f'sam_index_{exp}.nc')
    s = d['sam_jja']
    mu, sd = float(s.mean()), float(s.std())             # ddof=0, as composite scripts
    yrs = s['year'].values
    hi = yrs[s.values > mu + thr * sd]
    lo = yrs[s.values < mu - thr * sd]
    d.close()
    return hi, lo


def unit_xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def t63_to_mesh(t_lon, t_lat, field, ocean, lon, lat):
    """Nearest ECHAM ocean cell (great-circle sense) for every mesh node."""
    lo2, la2 = np.meshgrid(t_lon, t_lat)
    tree = cKDTree(unit_xyz(lo2[ocean], la2[ocean]))
    _, idx = tree.query(unit_xyz(lon, lat))
    return field[ocean][idx]


def heat_composite_files(exp, lat):
    """Heat composite on the mesh (W m-2) from heatflux_sam_composite_<exp>.nc.
    Nodes whose nearest T63 cell is ECHAM land carry 0 W m-2 there (over-water
    fluxes are 0 on land); those nodes are refilled from the T63 composite."""
    h = xr.open_dataset(COMP / f'heatflux_sam_composite_{exp}.nc')
    dq = h['total_heat_diff'].values.astype(float)
    zero = (np.abs(h['total_heat_high'].values) < 1e-2) & (np.abs(h['total_heat_low'].values) < 1e-2)
    lon = h['lon'].values
    t = xr.open_dataset(COMP / f'heatflux_sam_composite_t63_{exp}.nc')
    ocean = np.abs(t['total_heat_high'].values) > 1e-2
    fill = t63_to_mesh(t['lon'].values, t['lat'].values, t['total_heat_diff'].values,
                       ocean, lon[zero], lat[zero])
    dq[zero] = fill
    meta = dict(n_hi=int(h.attrs['n_high_sam_years']), n_lo=int(h.attrs['n_low_sam_years']),
                thr=float(h.attrs['threshold_std']), zero=zero)
    h.close(); t.close()
    return dq, meta


def fw_composite_files(exp):
    f = xr.open_dataset(COMP / f'fwflux_sam_composite_{exp}.nc')
    df = f['total_fw_diff'].values.astype(float)          # mm/day, + = into ocean
    assert float(f.attrs['threshold_std']) > 0
    f.close()
    return df


def heat_composite_recompute(exp, hi, lo, lon, lat):
    d100 = BASE / exp / '100years'
    tot = None
    for code in (92, 95, 111, 120):                       # LW, SW, LH, SH over water
        d = xr.open_dataset(d100 / f'{code}_mergetime_echam.nc', decode_times=False)
        tv = d['time'].values
        yr, mo = (tv // 10000).astype(int), ((tv % 10000) // 100).astype(int)
        v = d[f'var{code}'].values.astype(float)
        tot = v if tot is None else tot + v
        t_lon, t_lat = d['lon'].values, d['lat'].values
        d.close()
    jja = np.isin(mo, JJA_MONTHS)
    q_hi = tot[jja & np.isin(yr, hi)].mean(0)
    q_lo = tot[jja & np.isin(yr, lo)].mean(0)
    ocean = np.abs(q_hi) > 1e-2
    return t63_to_mesh(t_lon, t_lat, q_hi - q_lo, ocean, lon, lat)


def fw_composite_recompute(exp, hi, lo):
    ds = xr.open_dataset(BASE / exp / '100years' / 'fw_mergetime.nc', decode_times=False)
    tm = netCDF4.num2date(ds['time'].values, ds['time'].attrs['units'],
                          ds['time'].attrs.get('calendar', 'standard'))
    yr = np.array([t.year for t in tm]); mo = np.array([t.month for t in tm])
    jja = np.isin(mo, JJA_MONTHS)
    i_hi = np.where(jja & np.isin(yr, hi))[0]
    i_lo = np.where(jja & np.isin(yr, lo))[0]
    f_hi = -ds['fw'].isel(time=i_hi).values.astype(float).mean(0)   # -fw: + = into ocean
    f_lo = -ds['fw'].isel(time=i_lo).values.astype(float).mean(0)
    ds.close()
    return (f_hi - f_lo) / MMDAY_TO_MS                                 # mm/day


# pi/a_ice_clim.nc was overwritten 2026-03-15 with a 2090-2099 mean of another run;
# the paper's PI period (pi_age 2300-2399) is in a_ice_clim_beta.nc.
CLIM_FILE = {('pi', 'a_ice'): 'a_ice_clim_beta.nc'}


def read_clim(exp, v):
    """JJA mean of a 12-month climatology; refuses files not built from <exp>_age."""
    fn = CLIM_FILE.get((exp, v), f'{v}_clim.nc')
    ds = xr.open_dataset(BASE / exp / fn, decode_times=False)
    assert f'production/{exp}_age/' in ds.attrs.get('history', ''), \
        f'{exp}/{fn} is not built from production/{exp}_age'
    out = ds[v].isel(time=JJA_IDX).mean('time').values.astype(float)
    ds.close()
    return out


def jja_state(exp, lon, lat):
    sst, sss, ai = (read_clim(exp, v) for v in ('sst', 'sss', 'a_ice'))   # a_ice: 0-1
    SA = gsw.SA_from_SP(sss, 0.0, lon, lat)
    CT = gsw.CT_from_t(SA, sst, 0.0)
    alpha = gsw.alpha(SA, CT, 0.0)
    beta = gsw.beta(SA, CT, 0.0)                          # kg/g
    cp = gsw.cp_t_exact(SA, sst, 0.0)
    return dict(alpha=alpha, beta=beta, cp=cp, S=sss, ai=ai, sst=sst)


def to_density_flux(dq_wm2, df_mmday, st):
    """kg m-2 s-1, positive = densification."""
    H = -(st['alpha'] / st['cp']) * dq_wm2                # Q + into ocean
    F = -RHO_FW * st['beta'] * st['S'] * df_mmday * MMDAY_TO_MS   # FW + into ocean
    return H, F


def mean_state(exp, lon, lat, st):
    """JJA climatological H and F (not anomalies), same conversion and masks."""
    e = xr.open_dataset(BASE / exp / 'echam_clim.nc', decode_times=False)
    q = sum(e[f'var{c}'].isel(time=JJA_IDX).mean('time').values.astype(float)
            for c in (92, 95, 111, 120))
    q_mesh = t63_to_mesh(e['lon'].values, e['lat'].values, q, np.abs(q) > 1e-2, lon, lat)
    e.close()
    fw_in = -read_clim(exp, 'fw') / MMDAY_TO_MS                       # mm/day, + into ocean
    return to_density_flux(q_mesh, fw_in, st)


# ------------------------------------------------------------- statistics
def region_stats(H, F, area, mask):
    A = area[mask]; h = H[mask]; f = F[mask]
    m = lambda x: float(np.sum(x * A) / np.sum(A))
    return dict(area=float(A.sum()) / 1e12,                       # 10^6 km2
                abs_h=m(np.abs(h)) * SEC_PER_MONTH, abs_f=m(np.abs(f)) * SEC_PER_MONTH,
                sgn_h=m(h) * SEC_PER_MONTH, sgn_f=m(f) * SEC_PER_MONTH,
                frac=float(A[np.abs(f) > np.abs(h)].sum() / A.sum()),
                int_h=float(np.sum(h * A)) / 1e6, int_f=float(np.sum(f * A)) / 1e6)


def all_stats(H, F, area, lat, ai):
    so = lat < LAT_SO
    return {'ice': region_stats(H, F, area, so & (ai > ICE_THR)),
            'free': region_stats(H, F, area, so & (ai <= ICE_THR)),
            'all': region_stats(H, F, area, so)}


def run(thr, from_files):
    out = {}
    for exp in EXPS:
        lon, lat, area, elem = load_mesh(exp)
        st = jja_state(exp, lon, lat)
        if from_files:
            dq, meta = heat_composite_files(exp, lat)
            df = fw_composite_files(exp)
            assert dq.size == lon.size == df.size, f'{exp}: node count mismatch'
            assert abs(meta['thr'] - thr) < 1e-6, f'{exp}: files are {meta["thr"]} sigma'
            n_hi, n_lo = meta['n_hi'], meta['n_lo']
            so = lat < LAT_SO
            meta['zero_area_so'] = float(area[meta['zero'] & so].sum() / area[so].sum())
        else:
            hi, lo = sam_years(exp, thr)
            dq = heat_composite_recompute(exp, hi, lo, lon, lat)
            df = fw_composite_recompute(exp, hi, lo)
            n_hi, n_lo, meta = len(hi), len(lo), {}
        nan = int(np.isnan(dq).sum() + np.isnan(df).sum())
        assert nan == 0, f'{exp}: {nan} NaN in composites'
        H, F = to_density_flux(dq, df, st)
        Hm, Fm = mean_state(exp, lon, lat, st)
        out[exp] = dict(lon=lon, lat=lat, area=area, elem=elem, ai=st['ai'], H=H, F=F,
                        dq=dq, df=df, st=st, n_hi=n_hi, n_lo=n_lo, meta=meta,
                        stats=all_stats(H, F, area, lat, st['ai']),
                        mstats=all_stats(Hm, Fm, area, lat, st['ai']))
        print(f'{exp}: {n_hi} high / {n_lo} low years, ice-area frac |F|>|H| = '
              f'{out[exp]["stats"]["ice"]["frac"]:.2f}, ice-free = {out[exp]["stats"]["free"]["frac"]:.2f}')
    return out


# ------------------------------------------------------------------ figure
def nice_vmax(x):
    mag = 10 ** np.floor(np.log10(x))
    for n in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if n * mag >= x:
            return n * mag
    return 10 * mag


def plot(res, outbase, thr):
    proj = ccrs.SouthPolarStereo()
    pc = ccrs.PlateCarree()
    vals = np.concatenate([np.abs(np.r_[r['H'][r['lat'] < LAT_SO], r['F'][r['lat'] < LAT_SO]])
                           for r in res.values()]) * SEC_PER_MONTH
    vmax = nice_vmax(np.percentile(vals, 98))
    levels = np.linspace(-vmax, vmax, 21)
    cmap = plt.get_cmap('RdBu_r')

    ncol, nrow = 5, 2
    FW, L, R, gap = 17.0, 0.065, 0.99, 0.006
    w = (R - L - gap * (ncol - 1)) / ncol
    pw_in = w * FW                                       # panel width in inches
    bot_in, top_in, vgap_in = 1.25, 0.55, 0.08
    FH = bot_in + top_in + nrow * pw_in + vgap_in
    fig = plt.figure(figsize=(FW, FH))
    h = pw_in / FH

    rows = [('H', 'Heat flux term'), ('F', 'Freshwater flux term')]
    for r, (key, rlab) in enumerate(rows):
        for c, exp in enumerate(EXPS):
            d = res[exp]
            x0 = L + c * (w + gap)
            y0 = (bot_in + (nrow - 1 - r) * (pw_in + vgap_in)) / FH
            ax = fig.add_axes([x0, y0, w, h], projection=proj)
            ax.set_extent([-180, 180, -90, LAT_SO], crs=pc)
            ax.patch.set_facecolor('#c8c8c8')            # no mesh = land / ice sheet

            sel = d['lat'] < -30
            remap = -np.ones(sel.size, int); remap[sel] = np.arange(sel.sum())
            keep = sel[d['elem']].all(1)
            tris = remap[d['elem'][keep]]
            xy = proj.transform_points(pc, d['lon'][sel], d['lat'][sel])
            tri = mtri.Triangulation(xy[:, 0], xy[:, 1], tris)
            z = d[key][sel] * SEC_PER_MONTH
            cs = ax.tricontourf(tri, z, levels=levels, cmap=cmap, extend='both', zorder=1)
            for coll in cs.collections:                  # keeps the PDF small
                coll.set_rasterized(True)
            ax.tricontour(tri, d['ai'][sel], levels=[ICE_THR], colors='k',
                          linewidths=0.9, zorder=3)
            ax.set_extent([-180, 180, -90, LAT_SO], crs=pc)

            gl = ax.gridlines(crs=pc, draw_labels=False, linewidth=0.5, color='#555555',
                              alpha=0.5, linestyle='--', zorder=4)
            gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 60))
            gl.ylocator = mticker.FixedLocator([-80, -70, -60, -50])
            for la in (-50, -60, -70, -80):
                ax.text(150, la, f'{abs(la)}°S', transform=pc, ha='center', va='center',
                        fontsize=8, color='#111111', zorder=6,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.4))
            ax.text(0.03, 0.97, f'({chr(97 + r * ncol + c)})', transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top', ha='left', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9))
            if r == 0:
                ax.set_title(LABELS[exp], fontsize=18, fontweight='bold', pad=6)
            if c == 0:
                fig.text(x0 - 0.012, y0 + h / 2, rlab, rotation=90, ha='right', va='center',
                         fontsize=15, fontweight='bold')

    cax = fig.add_axes([0.25, 0.78 / FH, 0.5, 0.2 / FH])
    fig.text(0.765, 0.88 / FH, 'black line: JJA 15% sea ice edge', ha='left', va='center',
             fontsize=12)
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(-vmax, vmax), cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', extend='both',
                      ticks=np.linspace(-vmax, vmax, 9))
    cb.ax.tick_params(labelsize=12)
    cb.set_label(f'Surface density flux anomaly, high minus low SAM ({UNIT}); '
                 'positive = densification', fontsize=13)
    for ext, dpi in (('png', 200), ('pdf', 300)):
        fig.savefig(f'{outbase}.{ext}', dpi=dpi)
    plt.close(fig)
    print(f'saved {outbase}.png/.pdf (vmax = {vmax} {UNIT_TXT}, threshold {thr} sigma)')
    return vmax


# ----------------------------------------------------------------- numbers
def table(res, thr, source, key='stats'):
    head = (f'--- threshold +/-{thr} sigma ({source}) ---' if key == 'stats' else
            f'--- {source} ---')
    L = [head,
         'area-weighted over ocean south of 50S; means in ' + UNIT_TXT + '; integrals in 10^6 kg s-1',
         'region: ice = JJA ice conc > 0.15 (own state, climatology); free = <= 0.15',
         f'{"state":5s} {"yrs":>6s} {"reg":4s} {"area":>6s} {"|H|":>7s} {"|F|":>7s} {"|F|/|H|":>7s} '
         f'{"H":>7s} {"F":>7s} {"fr|F|>|H|":>9s} {"intH":>7s} {"intF":>7s}']
    for exp in EXPS:
        r = res[exp]
        for reg in ('ice', 'free', 'all'):
            s = r[key][reg]
            L.append(f'{LABELS[exp]:5s} {r["n_hi"]:>2d}/{r["n_lo"]:<3d} {reg:4s} {s["area"]:6.2f} '
                     f'{s["abs_h"]:7.3f} {s["abs_f"]:7.3f} {s["abs_f"] / s["abs_h"]:7.2f} '
                     f'{s["sgn_h"]:+7.3f} {s["sgn_f"]:+7.3f} {s["frac"]:9.2f} '
                     f'{s["int_h"]:+7.2f} {s["int_f"]:+7.2f}')
    return L


def classify(res):
    """Rule-based reading of the manuscript sentence, per state."""
    L = ['rules: part A "FW larger wherever sea ice is present" holds if fr|F|>|H| >= 0.90 in the',
         'ice region, holds for most of it if 0.50-0.90, fails below 0.50. Part B "heat larger only',
         'in the ice-free sector" holds if fr|F|>|H| < 0.50 and |H| > |F| (area mean) in the ice-free region.']
    for exp in EXPS:
        si, sf = res[exp]['stats']['ice'], res[exp]['stats']['free']
        a = 'holds' if si['frac'] >= 0.9 else ('holds for most' if si['frac'] >= 0.5 else 'fails')
        b = 'holds' if (sf['frac'] < 0.5 and sf['abs_h'] > sf['abs_f']) else 'fails'
        L.append(f'{LABELS[exp]:5s} A {a:15s} (ice fr={si["frac"]:.2f}, |F|/|H|={si["abs_f"] / si["abs_h"]:.1f})   '
                 f'B {b:6s} (free fr={sf["frac"]:.2f}, |F|/|H|={sf["abs_f"] / sf["abs_h"]:.2f})')
    return L


def verdict(r10, r12):
    rng = lambda res, reg, f, key='stats': (min(f(res[e][key][reg]) for e in EXPS),
                                            max(f(res[e][key][reg]) for e in EXPS))
    fr = lambda s: 100 * s['frac']
    fr_h = lambda s: 100 * (1 - s['frac'])
    rat = lambda s: s['abs_f'] / s['abs_h']
    irat = lambda s: s['abs_h'] / s['abs_f']
    big = lambda res, reg, key='stats': [LABELS[e] for e in EXPS
                                         if abs(res[e][key][reg]['int_f']) > abs(res[e][key][reg]['int_h'])]
    a, b = rng(r10, 'ice', fr), rng(r10, 'ice', rat)
    c, d = rng(r10, 'free', fr_h), rng(r10, 'free', irat)
    a2, c2 = rng(r12, 'ice', fr), rng(r12, 'free', fr_h)
    h0, h1 = rng(r10, 'ice', fr_h)
    m1, m2 = rng(r10, 'ice', fr, 'mstats'), rng(r10, 'free', fr_h, 'mstats')
    return [
        'VERDICT on revised.tex line 180 ("freshwater larger wherever sea ice is present, heat larger',
        'only in the ice-free sector, the same partition that governs the mean state"):',
        f'Point by point it HOLDS in all five states. Ice-covered area: |F| > |H| over {a[0]:.0f}-{a[1]:.0f}% of',
        f'the area (1.2 sigma: {a2[0]:.0f}-{a2[1]:.0f}%), area-mean |F| is {b[0]:.1f}-{b[1]:.1f} x |H|. Ice-free area:',
        f'|H| > |F| over {c[0]:.0f}-{c[1]:.0f}% (1.2 sigma: {c2[0]:.0f}-{c2[1]:.0f}%), area-mean |H| is {d[0]:.1f}-{d[1]:.1f} x |F|.',
        f'"Wherever" is slightly too strong: heat is the larger term over {h0:.0f}-{h1:.0f}% of the ice area.',
        'It does NOT carry over to area-integrated net anomalies, because F is a coast/offshore dipole',
        'and H changes sign between sectors. |integral F| > |integral H| over the ice area in: ' +
        ', '.join(big(r10, 'ice')) + '; over the ice-free area in: ' + ', '.join(big(r10, 'free')) + '.',
        f'Mean-state clause: the JJA climatology shows the same partition but weaker, |F| > |H| over',
        f'{m1[0]:.0f}-{m1[1]:.0f}% of the ice area and |H| > |F| over {m2[0]:.0f}-{m2[1]:.0f}% of the ice-free area;',
        'net over the ice area |integral F| > |integral H| only in: ' + ', '.join(big(r10, 'ice', 'mstats')) + '.']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true',
                    help='recompute the 1.0 sigma composites from monthly data and compare')
    args = ap.parse_args()

    if args.validate:
        a, b = run(1.0, True), run(1.0, False)
        for exp in EXPS:
            so = a[exp]['lat'] < LAT_SO
            for k in ('dq', 'df'):
                x, y = a[exp][k][so], b[exp][k][so]
                print(f'{exp} {k}: corr {np.corrcoef(x, y)[0, 1]:.4f}  mean file {x.mean():+.3f} '
                      f'recomputed {y.mean():+.3f}  max|diff| {np.abs(x - y).max():.3f}')
        return

    res10 = run(1.0, True)
    vmax = plot(res10, HERE / 'figR22_sam_density_tendency', 1.0)
    res12 = run(1.2, False)
    plot(res12, HERE / 'figR22_sam_density_tendency_1p2sigma', 1.2)

    lines = ['Fig R2.2 numbers: JJA high-minus-low SAM surface density flux, heat term H vs',
             'freshwater term F (total FW, sea ice included). Written by plot_sam_density_tendency.py.',
             '',
             'H = -(alpha/cp) dQ,  dQ = SW+LW+SH+LH over water from ECHAM, W m-2, + into ocean',
             'F = -rho_fw beta S dFW,  dFW = -fw from FESOM, m s-1, + into ocean, rho_fw = 1000',
             'alpha, beta, cp from gsw at each state\'s JJA climatological SST/SSS; S = JJA SSS.',
             'Both in kg m-2 s-1 (shown x 30 days); positive = densification.',
             f'Figure colour range: +/-{vmax} {UNIT_TXT}.',
             '']
    lines += table(res10, 1.0, 'composite_sam/*_sam_composite_<exp>.nc, as plotted in Figs 5 and 6')
    lines.append('heat-file nodes refilled from the T63 composite (ECHAM land under nearest cell), '
                 'area fraction S of 50S: ' +
                 ', '.join(f'{LABELS[e]} {res10[e]["meta"]["zero_area_so"]:.3f}' for e in EXPS))
    lines += [''] + classify(res10) + ['']
    lines += table(res12, 1.2, 'recomputed from monthly 100years data, threshold stated in the captions')
    lines += [''] + classify(res12) + ['']
    lines += table(res10, None, 'mean state for comparison: JJA climatology (not an anomaly), '
                   'H from echam_clim.nc, F from fw_clim.nc; yrs column = SAM years, unused here',
                   key='mstats') + ['']
    lines += verdict(res10, res12)
    (HERE / 'numbers.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
