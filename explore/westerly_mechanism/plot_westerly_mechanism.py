#!/usr/bin/env python3
"""Two diagnostic figures from work/profiles.nc (analyze_westerlies.py) plus sector u10 from echam_clim.

fig_profiles_JJA.png : JJA zonal-mean profiles by state (u10, u250, surface/SST gradient, T850 gradient,
                       Eady 850-500, T250 gradient). Coloured ticks at the top = ECHAM sea-ice edge.
fig_anomalies_JJA.png: JJA anomalies vs PI (u10/u850, Eady 850-500 and 500-250, T850/T250 gradient,
                       T at 850/250 hPa incl. tropics, TOA insolation, Pacific-sector u10).
Run in the foreground: module load python3/2022.01-gcc-11.2.0; python3 -s plot_westerly_mechanism.py
"""
import re
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path('/work/ba1066/a270064/cc_projects/aabw_5exps')
HERE = ROOT / 'explore/westerly_mechanism'
d = xr.open_dataset(HERE / 'work/profiles.nc')
lat = d.lat.values
latf = d.lat_fesom.values
EXPS = ['pi', 'mh', 'lig', 'lgm', 'mis']
LABEL = {'pi': 'PI', 'mh': 'MH', 'lig': 'LIG', 'lgm': 'LGM', 'mis': 'MIS3'}
COL = {'pi': 'black', 'mh': '#4daf4a', 'lig': '#e41a1c', 'lgm': '#377eb8', 'mis': '#17becf'}
LS = {'pi': '-', 'mh': ':', 'lig': '-', 'lgm': '-', 'mis': '--'}

# ice edges and jet latitudes from numbers_main.txt (no re-computation, so figure = numbers file)
txt = (HERE / 'numbers_main.txt').read_text()
blk = txt.split('==================== JJA ====================')[1].split('==================== ANN')[0]
ice = {}
for line in re.search(r'Sea-ice edge from ECHAM.*?\n((?:.*\n){5})', blk).group(1).splitlines():
    k = line.split()
    ice[[e for e in EXPS if LABEL[e] == k[0]][0]] = float(k[1])


def v(name, e, s='JJA'):
    return d[f'{name}_{s}_{e}'].values


def jet_marks(ax, name, lo=-70, hi=-30):
    for e in EXPS:
        y = v(name, e)
        m = (lat >= lo) & (lat <= hi)
        i = np.where(m)[0][np.nanargmax(y[m])]
        ax.plot(lat[i], y[i], 'o', color=COL[e], ms=4)


def ice_ticks(ax):
    for e in EXPS:
        ax.axvline(ice[e], ymin=0.93, ymax=1.0, color=COL[e], lw=2.5)


def style(ax, title, ylab, xlim=(-75, -20), tsize=10, lsize=9, ksize=8, weight='normal'):
    # defaults are the original sizes (figure 2); figure 1 passes F1 below
    ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=tsize, loc='left', fontweight=weight)
    ax.set_ylabel(ylab, fontsize=lsize, fontweight=weight)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=ksize)
    ax.set_xlabel('latitude (°N)', fontsize=lsize, fontweight=weight)


# ---------------------------------------------------------------- figure 1: profiles
# Letter Fig. R23 prints this 5.81 in wide (0.9 linewidth). 2026-11-27 readability pass: 7.5 in square,
# so sizes print at x0.775: ticks 10 -> 7.8 pt, axis labels and legend 10.5 -> 8.1 pt,
# panel titles (with their letters) 12 bold -> 9.3 pt, suptitle 11 -> 8.5 pt.
F1 = dict(tsize=12, lsize=10.5, ksize=10, weight='bold')
fig = plt.figure(figsize=(7.5, 7.65))   # W/H >= 0.98 keeps the 5.81 in letter width
# spacer rows: the gap under row 1 holds its x label plus the two-line title of (c), the gap under row 2
# only a one-line title (fixed margins; tight_layout gives every row the space of the tallest title)
gs = fig.add_gridspec(5, 2, height_ratios=[1, 0.745, 1, 0.585, 1], hspace=0, wspace=0.235,
                      left=0.08, right=0.975, top=0.835, bottom=0.062)
ax = [fig.add_subplot(gs[r, c]) for r in (0, 2, 4) for c in (0, 1)]
for e in EXPS:
    kw = dict(color=COL[e], ls=LS[e], lw=1.6, label=LABEL[e])
    ax[0].plot(lat, v('u10', e), **kw)
    ax[1].plot(lat, v('u250', e), **kw)
    ax[2].plot(lat, v('ts_oc_grad', e), **kw)
    ax[2].plot(latf, v('sst_fesom_grad', e), color=COL[e], ls=LS[e], lw=1.0, alpha=0.6)
    ax[3].plot(lat, v('T850_grad', e), **kw)
    ax[4].plot(lat, v('eady850_500', e), **kw)
    ax[5].plot(lat, v('T250_grad', e), **kw)
jet_marks(ax[0], 'u10')
style(ax[0], '(a) 10 m zonal wind\n(dots: maximum)', 'm s$^{-1}$', **F1)
style(ax[1], '(b) 250 hPa zonal wind', 'm s$^{-1}$', **F1)
style(ax[2], '(c) dT/dy of surface temperature\nover ocean boxes (thin: FESOM SST)', 'K per 1000 km', **F1)
style(ax[3], '(d) dT/dy at 850 hPa', 'K per 1000 km', **F1)
style(ax[4], '(e) Eady growth rate 850–500 hPa', 'day$^{-1}$', **F1)
style(ax[5], '(f) dT/dy at 250 hPa', 'K per 1000 km', **F1)
for a in ax:
    ice_ticks(a)
# state legend in one row under the title: inside (a) it covered the curves at this font size
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.937), ncol=5, fontsize=10.5, frameon=False,
           handlelength=2.2, columnspacing=1.6)
fig.suptitle('JJA zonal means, AWI-ESM 100-yr climatologies;\ncoloured ticks at top = sea-ice edge (ECHAM, 15%)',
             fontsize=11, y=0.993)
# 240 dpi: 1800 px, >= 300 px per printed inch in the letter
fig.savefig(HERE / 'fig_profiles_JJA.png', dpi=240)
plt.close(fig)

# ---------------------------------------------------------------- figure 2: anomalies vs PI
# Pacific-sector u10 (145E-70W) straight from echam_clim
pac = {}
for e in EXPS:
    ds = xr.open_dataset(ROOT / e / 'echam_clim.nc', decode_times=False)['var165'].sortby('lat')
    lon = ds.lon.values
    lm = (lon >= 145) & (lon < 290)
    pac[e] = ds.isel(time=[5, 6, 7]).mean('time').values[:, lm].mean(-1)

fig, axs = plt.subplots(2, 4, figsize=(17, 8))
ax = axs.ravel()
for e in ['mh', 'lig', 'lgm', 'mis']:
    kw = dict(color=COL[e], ls=LS[e], lw=1.6, label=LABEL[e] + '−PI')
    ax[0].plot(lat, v('u10', e) - v('u10', 'pi'), **kw)
    ax[0].plot(lat, v('u850', e) - v('u850', 'pi'), color=COL[e], ls=LS[e], lw=0.8, alpha=0.6)
    ax[1].plot(lat, v('eady850_500', e) - v('eady850_500', 'pi'), **kw)
    ax[2].plot(lat, v('eady500_250', e) - v('eady500_250', 'pi'), **kw)
    ax[3].plot(lat, v('T850_grad', e) - v('T850_grad', 'pi'), **kw)
    ax[4].plot(lat, v('T250_grad', e) - v('T250_grad', 'pi'), **kw)
    ax[5].plot(lat, v('T850', e) - v('T850', 'pi'), **kw)
    ax[5].plot(lat, v('T250', e) - v('T250', 'pi'), color=COL[e], ls='--', lw=1.0)
    ax[6].plot(lat, v('insol', e) - v('insol', 'pi'), **kw)
for e in EXPS:
    ax[7].plot(lat, pac[e], color=COL[e], ls=LS[e], lw=1.6, label=LABEL[e])
style(ax[0], '(a) Δu10 (thick), Δu850 (thin)', 'm s$^{-1}$')
style(ax[1], '(b) ΔEady 850–500 hPa', 'day$^{-1}$')
style(ax[2], '(c) ΔEady 500–250 hPa', 'day$^{-1}$')
style(ax[3], '(d) Δ dT/dy at 850 hPa', 'K per 1000 km')
style(ax[4], '(e) Δ dT/dy at 250 hPa', 'K per 1000 km')
style(ax[5], '(f) ΔT at 850 hPa (solid), 250 hPa (dashed)', 'K', xlim=(-75, 30))
style(ax[6], '(g) Δ TOA incoming solar', 'W m$^{-2}$', xlim=(-90, 90))
style(ax[7], '(h) u10, Pacific sector 145°E–70°W', 'm s$^{-1}$')
for a in ax:
    a.axhline(0, color='grey', lw=0.6)
for a in [ax[0], ax[1], ax[2], ax[3], ax[4], ax[7]]:
    a.axvline(-45.23, color='k', lw=0.8, ls=':')   # PI u10 jet (fit), numbers_main.txt
    ice_ticks(a)
ax[0].legend(fontsize=8)
ax[7].legend(fontsize=8)
fig.suptitle('JJA anomalies relative to PI (dotted vertical line: PI 10 m jet latitude; ticks at top: sea-ice edge)', fontsize=10)
fig.tight_layout()
fig.savefig(HERE / 'fig_anomalies_JJA.png', dpi=150)
plt.close(fig)
print('wrote', HERE / 'fig_profiles_JJA.png', HERE / 'fig_anomalies_JJA.png')
