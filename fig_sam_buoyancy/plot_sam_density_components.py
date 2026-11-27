#!/usr/bin/env python
"""Supplementary Figs. A3 and A4 in density units: JJA high-minus-low SAM (1.0 sigma)
heat and freshwater flux components converted to surface density flux, ONE colour
scale shared by both figures. Also writes numbers_components.txt and
numbers_heatloss15.txt (area where total_heat_diff < -15 W m-2).

Conversion, signs, masks and ECHAM-land refill are those of plot_sam_density_tendency.py
(imported below); see CLAUDE.md in this folder. FW components in the composite files are
all + = into ocean (total = -fw, other = evap+prec+snow+runoff, sea ice = total - other).

Usage (login node, foreground, about 1 min):
    python3 -s plot_sam_density_components.py
"""

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import matplotlib.patheffects as patheffects
import cartopy.crs as ccrs

import plot_sam_density_tendency as base
from plot_sam_density_tendency import (EXPS, LABELS, COMP, SEC_PER_MONTH, RHO_FW,
                                       MMDAY_TO_MS, ICE_THR, LAT_SO)

HERE = Path(__file__).resolve().parent
UNIT = 'kg m$^{-2}$ month$^{-1}$'
UNIT_TXT = 'kg m-2 month-1'
DQ_LINE = -15.0                       # W m-2, contour on the total heat row of A3
DQ_COLOR = '#5b1a8c'                  # dark purple
# 2026-11-27 A4 readability pass: the letter prints both figures 5.81 in wide, so they are
# drawn at that width and every font size and line width below is the printed size.
PRINT_W = 5.81                        # in
PNG_DPI = 585                         # 5.81 in x 585 = 3399 px, about the old 3400 px
PDF_DPI = 600                         # raster of the filled contours inside the PDF
# 2026-11-27 10:34 (user): no sea ice edge in A7/A8; every latitude circle and every
# meridian labelled in every panel, small (6 pt printed, may go below the 7.5 pt floor).
GRID_LONS = np.arange(-180, 180, 60)          # meridians drawn and labelled
GRID_LATS = [-80, -70, -60, -50]              # latitude circles drawn and labelled
FS_GRID = 6                                   # printed pt
# Latitude labels sit along 20E: a search over meridians and label positions (box under
# each label scored by the area integral of |field|/vmax in all 30 panels, the brine rows
# counted twice) found this the least hidden signal. Per label: side of the meridian
# (L, R, C = centred) and whether it sits on the pole side of its circle.
LATLAB_LON = 20
LAT_LAYOUT = {-80: ('C', True), -70: ('C', True), -60: ('L', False), -50: ('R', True)}
HALO = [patheffects.withStroke(linewidth=1.2, foreground='white')]

HEAT_VARS = ['total_heat_diff', 'radiative_diff', 'turbulent_diff']
FW_VARS = ['total_fw_diff', 'seaice_fw_diff', 'other_fw_diff']
# Row labels get extra line breaks so that they fit beside one printed panel (2026-11-27).
ROWS_A3 = [('H_total', 'Total heat flux'),
           ('H_rad', 'Radiative\n(shortwave\n+ longwave)'),
           ('H_turb', 'Turbulent\n(sensible\n+ latent)')]
ROWS_A4 = [('F_total', 'Total\nfreshwater\nflux (sea ice\nincluded)'),
           ('F_ice', 'Sea ice'),
           ('F_other', 'Other\n(net\nprecipitation\n+ runoff)')]
KEYS = [k for k, _ in ROWS_A3 + ROWS_A4]


# ----------------------------------------------------------------- inputs
def heat_components(exp, lon, lat):
    """W m-2 on the mesh, + into ocean. Nodes whose nearest T63 cell is ECHAM land
    (high and low total both ~0) are refilled from the nearest T63 ocean cell, as in
    base.heat_composite_files, for all three components with the same index."""
    h = xr.open_dataset(COMP / f'heatflux_sam_composite_{exp}.nc')
    zero = (np.abs(h['total_heat_high'].values) < 1e-2) & (np.abs(h['total_heat_low'].values) < 1e-2)
    t = xr.open_dataset(COMP / f'heatflux_sam_composite_t63_{exp}.nc')
    ocean = np.abs(t['total_heat_high'].values) > 1e-2
    out, raw = {}, {}
    for v in HEAT_VARS:
        x = h[v].values.astype(float)
        raw[v] = x.copy()
        x[zero] = base.t63_to_mesh(t['lon'].values, t['lat'].values, t[v].values,
                                   ocean, lon[zero], lat[zero])
        out[v] = x
    area = h['areacello'].values.astype(float)
    h.close(); t.close()
    return out, raw, zero, area


def fw_components(exp):
    f = xr.open_dataset(COMP / f'fwflux_sam_composite_{exp}.nc')
    assert abs(float(f.attrs['threshold_std']) - 1.0) < 1e-6
    out = {v: f[v].values.astype(float) for v in FW_VARS}            # mm/day, + into ocean
    f.close()
    return out


def load_all():
    res = {}
    for exp in EXPS:
        lon, lat, area, elem = base.load_mesh(exp)
        st = base.jja_state(exp, lon, lat)
        dq, dq_raw, zero, areacello = heat_components(exp, lon, lat)
        df = fw_components(exp)
        assert dq['total_heat_diff'].size == lon.size == df['total_fw_diff'].size
        assert np.allclose(areacello, area), f'{exp}: areacello differs from mesh nod_area'
        # same refilled total as figR22
        chk, _ = base.heat_composite_files(exp, lat)
        assert np.allclose(chk, dq['total_heat_diff']), f'{exp}: refill differs from figR22'
        d = {}
        for key, v in zip(['H_total', 'H_rad', 'H_turb'], HEAT_VARS):
            d[key] = -(st['alpha'] / st['cp']) * dq[v] * SEC_PER_MONTH
        for key, v in zip(['F_total', 'F_ice', 'F_other'], FW_VARS):
            d[key] = -RHO_FW * st['beta'] * st['S'] * df[v] * MMDAY_TO_MS * SEC_PER_MONTH
        for k in KEYS:
            assert not np.isnan(d[k]).any(), f'{exp} {k}: NaN'
        # triangulation of the nodes south of 30S (projected), as in figR22
        sel = lat < -30
        remap = -np.ones(sel.size, int); remap[sel] = np.arange(sel.sum())
        keep = sel[elem].all(1)
        res[exp] = dict(lon=lon, lat=lat, area=area, ai=st['ai'], dq=dq['total_heat_diff'],
                        dq_raw=dq_raw['total_heat_diff'], zero=zero, sel=sel,
                        tris=remap[elem[keep]], **d)
        print(f'{exp}: loaded, refilled nodes {zero.sum()}')
    return res


# ------------------------------------------------------------------ figure
def shared_scale(res):
    vals = np.concatenate([np.abs(r[k][r['lat'] < LAT_SO]) for r in res.values() for k in KEYS])
    wts = np.concatenate([r['area'][r['lat'] < LAT_SO] for r in res.values() for k in KEYS])
    p98 = float(np.percentile(vals, 98))
    vmax = base.nice_vmax(p98)
    o = np.argsort(vals)
    cw = np.cumsum(wts[o]) / wts.sum()
    p98w = float(vals[o][np.searchsorted(cw, 0.98)])
    info = dict(p98=p98, p98w=p98w, vmax=vmax,
                pct_at_vmax=float(100 * np.mean(vals <= vmax)),
                pct_at_vmax_w=float(100 * wts[vals <= vmax].sum() / wts.sum()))
    return vmax, info


def lon_text(lo):
    lo = ((lo + 180) % 360) - 180
    if lo == 0:
        return '0°'
    if abs(lo) == 180:
        return '180°'
    return f'{abs(lo)}°{"E" if lo > 0 else "W"}'


def grid_labels(ax, proj, pc):
    """Meridian labels just outside the square frame where each meridian meets it
    (rotated on the left and right sides); latitude labels along LATLAB_LON as set in
    LAT_LAYOUT, with a thin white halo."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    cx, cy = proj.transform_point(0, -90, pc)
    for lo in GRID_LONS:
        px, py = proj.transform_point(lo, -60, pc)
        dx, dy = px - cx, py - cy
        n = np.hypot(dx, dy)
        dx, dy = dx / n, dy / n
        tx = (x1 - cx) / dx if dx > 1e-9 else ((x0 - cx) / dx if dx < -1e-9 else np.inf)
        ty = (y1 - cy) / dy if dy > 1e-9 else ((y0 - cy) / dy if dy < -1e-9 else np.inf)
        t = min(tx, ty)
        fx = (cx + t * dx - x0) / (x1 - x0)
        fy = (cy + t * dy - y0) / (y1 - y0)
        if tx < ty:                                  # left or right side
            right = fx > 0.5
            kw = dict(xytext=(1.5 if right else -1.5, 0), ha='left' if right else 'right',
                      va='center', rotation=90)
        else:                                        # top or bottom
            top = fy > 0.5
            kw = dict(xytext=(0, 1.5 if top else -1.5), ha='center',
                      va='bottom' if top else 'top', rotation=0)
        ax.annotate(lon_text(lo), xy=(fx, fy), xycoords='axes fraction',
                    textcoords='offset points', fontsize=FS_GRID, color='k',
                    annotation_clip=False, **kw)
    for la in GRID_LATS:
        side, pole = LAT_LAYOUT[la]
        px, py = proj.transform_point(LATLAB_LON, la, pc)
        up = py > cy                                 # meridian in the upper half
        va = ('top' if up else 'bottom') if pole else 'center'
        oy = (-1.0 if up else 1.0) if pole else 0.0
        ox = {'L': -2.0, 'R': 2.0, 'C': 0.0}[side]
        ha = {'L': 'right', 'R': 'left', 'C': 'center'}[side]
        ax.annotate(f'{abs(la)}°S', xy=(px, py), xycoords='data', xytext=(ox, oy),
                    textcoords='offset points', ha=ha, va=va, fontsize=FS_GRID, color='k',
                    path_effects=HALO, zorder=6, annotation_clip=False)


def plot(res, rows, outbase, vmax, dq_line=False):
    proj = ccrs.SouthPolarStereo()
    pc = ccrs.PlateCarree()
    levels = np.linspace(-vmax, vmax, 21)
    cmap = plt.get_cmap('RdBu_r')

    ncol, nrow = 5, len(rows)
    FW = PRINT_W
    lab_in = (1.5 + 7.0) / 72                 # one 6 pt meridian label and its pad
    gap_in = 2 * lab_in + 3 / 72              # E labels of one panel, W labels of the next
    rowlab_in = 0.56                          # rotated row labels, up to 4 lines
    left_in = 0.02 + rowlab_in + 0.04 + lab_in
    right_in = lab_in + 0.02
    pw_in = (FW - left_in - right_in - gap_in * (ncol - 1)) / ncol
    vgap_in = (8.5 + 2 + 12.5) / 72           # 180° label below, letter and 0° label above
    top_in = 0.40                             # column titles above letters and 0° labels
    bot_in = 1.14 if dq_line else 0.92        # colour bar, its label and (A7) the legend
    FH = bot_in + top_in + nrow * pw_in + (nrow - 1) * vgap_in
    fig = plt.figure(figsize=(FW, FH))
    L, w, gap, h = left_in / FW, pw_in / FW, gap_in / FW, pw_in / FH

    for r, (key, rlab) in enumerate(rows):
        for c, exp in enumerate(EXPS):
            d = res[exp]
            sel = d['sel']
            x0 = L + c * (w + gap)
            y0 = (bot_in + (nrow - 1 - r) * (pw_in + vgap_in)) / FH
            ax = fig.add_axes([x0, y0, w, h], projection=proj)
            ax.set_extent([-180, 180, -90, LAT_SO], crs=pc)
            ax.patch.set_facecolor('#c8c8c8')            # no mesh = land / ice sheet
            xy = proj.transform_points(pc, d['lon'][sel], d['lat'][sel])
            tri = mtri.Triangulation(xy[:, 0], xy[:, 1], d['tris'])
            cs = ax.tricontourf(tri, d[key][sel], levels=levels, cmap=cmap, extend='both', zorder=1)
            for coll in cs.collections:
                coll.set_rasterized(True)
            if dq_line and r == 0:
                ax.tricontour(tri, d['dq'][sel], levels=[DQ_LINE], colors=DQ_COLOR,
                              linewidths=0.9, linestyles='--', zorder=3.5)
            ax.set_extent([-180, 180, -90, LAT_SO], crs=pc)
            ax.spines['geo'].set_linewidth(0.6)

            gl = ax.gridlines(crs=pc, draw_labels=False, linewidth=0.5, color='#444444',
                              alpha=0.6, linestyle='--', zorder=4)
            gl.xlocator = mticker.FixedLocator(GRID_LONS)
            gl.ylocator = mticker.FixedLocator(GRID_LATS)
            grid_labels(ax, proj, pc)
            # panel letter above the panel, so that it hides no data
            fig.text(x0, y0 + h + 0.03 / FH, f'({chr(97 + r * ncol + c)})', fontsize=9,
                     fontweight='bold', color='k', va='bottom', ha='left')
            if r == 0:
                ax.set_title(LABELS[exp], fontsize=10, fontweight='bold', color='k', pad=14)
            if c == 0:
                fig.text(x0 - (0.04 + lab_in) / FW, y0 + h / 2, rlab, rotation=90, ha='right',
                         va='center', fontsize=8.5, fontweight='bold', color='k',
                         multialignment='center', linespacing=1.1)

    cax = fig.add_axes([0.19, (bot_in - 0.30) / FH, 0.62, 0.10 / FH])
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(-vmax, vmax), cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', extend='both',
                      ticks=np.linspace(-vmax, vmax, 9))
    cb.ax.tick_params(labelsize=8, width=0.6, length=3, colors='k')
    cb.outline.set_linewidth(0.6)
    cb.set_label(f'Surface density flux anomaly, high minus low SAM ({UNIT});\n'
                 'positive = densification', fontsize=8.5, fontweight='bold', color='k',
                 labelpad=2, linespacing=1.15)
    if dq_line:
        handles = [Line2D([], [], color=DQ_COLOR, lw=1.0, ls='--',
                          label='total heat flux anomaly = $-$15 W m$^{-2}$ (panels a–e)')]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.05 / FH),
                   frameon=False, fontsize=8, handlelength=2.2, borderaxespad=0,
                   labelcolor='k')
    for ext, dpi in (('png', PNG_DPI), ('pdf', PDF_DPI)):
        fig.savefig(f'{outbase}.{ext}', dpi=dpi)
    plt.close(fig)
    print(f'saved {outbase}.png/.pdf')


# ----------------------------------------------------------------- numbers
def numbers_components(res, info):
    L = ['Supplementary Figs. A3/A4 (density units): area-weighted mean |value| south of 50S,',
         f'{UNIT_TXT}. JJA high-minus-low SAM, 1.0 sigma composites (composite_sam/*_sam_composite_<exp>.nc).',
         'Written by plot_sam_density_components.py. ice = own JJA conc > 0.15, free = <= 0.15.',
         'H = -(alpha/cp) dQ, F = -rho_fw beta S dFW (dFW + into ocean); positive = densification.',
         f'Shared colour range of both figures: +/-{info["vmax"]} {UNIT_TXT}. 98th percentile of |value| over',
         f'all 30 panels S of 50S: {info["p98"]:.2f} (node count), {info["p98w"]:.2f} (area weighted);',
         f'+/-{info["vmax"]} covers {info["pct_at_vmax"]:.1f}% of nodes, {info["pct_at_vmax_w"]:.1f}% of area.',
         '',
         f'{"state":5s} {"reg":4s} {"area":>6s} ' + ' '.join(f'{k:>8s}' for k in KEYS) + '   (mean |value|)']
    signed = ['', 'signed area-weighted means, same layout:',
              f'{"state":5s} {"reg":4s} {"area":>6s} ' + ' '.join(f'{k:>8s}' for k in KEYS)]
    for exp in EXPS:
        r = res[exp]
        so = r['lat'] < LAT_SO
        for reg, m in (('ice', so & (r['ai'] > ICE_THR)), ('free', so & (r['ai'] <= ICE_THR))):
            A = r['area'][m]
            a = lambda x: np.sum(np.abs(x[m]) * A) / A.sum()
            s = lambda x: np.sum(x[m] * A) / A.sum()
            L.append(f'{LABELS[exp]:5s} {reg:4s} {A.sum() / 1e12:6.2f} ' +
                     ' '.join(f'{a(r[k]):8.3f}' for k in KEYS))
            signed.append(f'{LABELS[exp]:5s} {reg:4s} {A.sum() / 1e12:6.2f} ' +
                          ' '.join(f'{s(r[k]):+8.3f}' for k in KEYS))
    L += signed + ['', 'area in 10^6 km2.']
    (HERE / 'numbers_components.txt').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def ice_edge_lat(r, binw=2.0):
    """Northernmost node with JJA conc > 0.15 in each longitude bin (south of 30S)."""
    m = (r['ai'] > ICE_THR) & (r['lat'] < -30)
    edges = np.arange(-180, 180 + binw, binw)
    b = np.digitize(((r['lon'][m] + 180) % 360) - 180, edges) - 1
    out = np.array([r['lat'][m][b == i].max() if np.any(b == i) else np.nan
                    for i in range(len(edges) - 1)])
    return out


def numbers_heatloss15(res, field='dq'):
    L = [f'Area where the 1.0 sigma JJA high-minus-low SAM total heat flux anomaly (total_heat_diff,',
         f'W m-2, + into ocean) is below {DQ_LINE:.0f} W m-2, i.e. high-SAM winters lose >= 15 W m-2 more heat.',
         'Mesh nodes, weights = areacello. ECHAM-land nodes refilled from the nearest T63 ocean cell (as in the',
         'figures). Ice = own JJA 15% edge (conc > 0.15; PI from pi/a_ice_clim_beta.nc). Areas in 10^3 km2.',
         'Written by plot_sam_density_components.py.', '']
    lon_edges = np.arange(-180, 181, 20)
    for exp in ('pi', 'mh', 'lig'):
        r = res[exp]
        so = r['lat'] < LAT_SO
        hit = so & (r[field] < DQ_LINE)
        A = r['area']
        tot = A[hit].sum()
        L.append(f'== {LABELS[exp]}: area {tot / 1e9:.0f} x10^3 km2 = {100 * tot / A[so].sum():.1f}% of ocean S of 50S '
                 f'({A[so].sum() / 1e12:.2f} x10^6 km2); raw file without refill: '
                 f'{A[so & (r["dq_raw"] < DQ_LINE)].sum() / 1e9:.0f}')
        ice = r['ai'] > ICE_THR
        L.append(f'   inside 15% edge {100 * A[hit & ice].sum() / tot:.1f}%  outside {100 * A[hit & ~ice].sum() / tot:.1f}%')
        bands = [('50-60S', (r['lat'] < -50) & (r['lat'] >= -60)),
                 ('60-70S', (r['lat'] < -60) & (r['lat'] >= -70)), ('S of 70S', r['lat'] < -70)]
        L.append('   lat band: ' + '  '.join(f'{n} {A[hit & m].sum() / 1e9:.0f} ({100 * A[hit & m].sum() / tot:.0f}%)'
                                            for n, m in bands))
        lo = ((r['lon'] + 180) % 360) - 180
        parts = []
        for a0, a1 in zip(lon_edges[:-1], lon_edges[1:]):
            m = hit & (lo >= a0) & (lo < a1)
            parts.append(f'{a0:+d}..{a1:+d}E {A[m].sum() / 1e9:5.0f} ({100 * A[m].sum() / tot:4.1f}%)')
        L.append('   20-deg longitude bins (lon -180..180):')
        for i in range(0, len(parts), 3):
            L.append('     ' + '   '.join(parts[i:i + 3]))
        L.append('')

    L += ['== LGM and MIS3: south vs north of the own JJA 15% ice edge',
          '   ice = conc > 0.15 (any latitude S of 30S); free S50 = conc <= 0.15 and S of 50S;',
          '   free 40-50S = conc <= 0.15 between 40S and 50S (outside the figure domain).',
          f'   {"state":5s} {"region":12s} {"area":>8s} {"mean dQ":>8s} {"A(dQ<-15)":>10s} {"frac":>6s}']
    for exp in ('lgm', 'mis'):
        r = res[exp]
        A = r['area']
        ice = (r['ai'] > ICE_THR) & (r['lat'] < -30)
        regs = [('ice', ice), ('ice S50', ice & (r['lat'] < LAT_SO)),
                ('free S50', ~ice & (r['lat'] < LAT_SO)),
                ('free 40-50S', ~ice & (r['lat'] < -40) & (r['lat'] >= -50))]
        for n, m in regs:
            a = A[m].sum()
            mean = np.sum(r[field][m] * A[m]) / a
            hit = A[m & (r[field] < DQ_LINE)].sum()
            L.append(f'   {LABELS[exp]:5s} {n:12s} {a / 1e9:8.0f} {mean:+8.2f} {hit / 1e9:10.0f} {hit / a:6.3f}')
        e = ice_edge_lat(r)
        L.append(f'   {LABELS[exp]:5s} JJA 15% edge, northernmost ice node per 2-deg lon bin: zonal mean '
                 f'{np.nanmean(e):.1f}, median {np.nanmedian(e):.1f}, range {np.nanmin(e):.1f} to {np.nanmax(e):.1f} '
                 f'(deg N), bins north of 50S: {100 * np.mean(e > -50):.0f}%')
    L.append('')
    L.append('   for reference, same edge measure: ' + ', '.join(
        f'{LABELS[e]} {np.nanmean(ice_edge_lat(res[e])):.1f}' for e in ('pi', 'mh', 'lig')))
    (HERE / 'numbers_heatloss15.txt').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    res = load_all()
    vmax, info = shared_scale(res)
    print(f'shared scale: p98 {info["p98"]:.3f} (area-w {info["p98w"]:.3f}) -> vmax {vmax}')
    plot(res, ROWS_A3, HERE / 'figA3_sam_heat_density', vmax, dq_line=True)
    plot(res, ROWS_A4, HERE / 'figA4_sam_fw_density', vmax)
    numbers_components(res, info)
    numbers_heatloss15(res)


if __name__ == '__main__':
    main()
