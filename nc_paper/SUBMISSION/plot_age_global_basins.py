#!/usr/bin/env python
"""
Global ideal-age structure for the five climate states.

Reviewer #3 asked for ideal age throughout the global ocean rather than a single
4000 m horizontal map, noting that "the deep ocean is usually considered as all
the water below 1000 m".

Panel row 1: basin-mean vertical profiles (Atlantic, Indian, Pacific, Southern).
Panel row 2: global zonal-mean age sections for the five states.

Uses the regridded global 1x1 degree, 47-level ideal age fields ({exp}/age_reg.nc).
"""
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

EXPS = ['pi', 'mh', 'lig', 'lgm', 'mis']
LABELS = ['PI', 'MH', 'LIG', 'LGM', 'MIS3']
# CVD-safe qualitative colours (Okabe-Ito)
COLS = {'pi': '#000000', 'mh': '#E69F00', 'lig': '#009E73',
        'lgm': '#0072B2', 'mis': '#CC79A7'}
LS = {'pi': '-', 'mh': '-', 'lig': '--', 'lgm': '-', 'mis': '--'}

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'pdf.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

BASINS = {
    'Atlantic':  dict(lat=(-35, 65),  lon=(-70, 20)),
    'Indian':    dict(lat=(-35, 25),  lon=(20, 115)),
    'Pacific':   dict(lat=(-35, 65),  lon=None),   # 120E-70W wrap
    'Southern':  dict(lat=(-78, -35), lon=None),
}


def load(exp):
    d = xr.open_dataset(f'{exp}/age_reg.nc')
    a = d['age'].isel(time=0)
    a = a.where(np.isfinite(a) & (a < 1e19))
    dep = -d['depth_coord'].values           # make positive downward
    return a, dep, d['lat'].values, d['lon'].values


data = {e: load(e) for e in EXPS}

fig = plt.figure(figsize=(23, 10.5))
gs = fig.add_gridspec(2, 5, hspace=0.34, wspace=0.26,
                      left=0.06, right=0.985, top=0.93, bottom=0.09)

letters = 'abcdefghij'
p = 0

# ---------------------------------------------------- row 1: basin profiles
for c, (bname, bsel) in enumerate(BASINS.items()):
    ax = fig.add_subplot(gs[0, c])
    for e, lab in zip(EXPS, LABELS):
        a, dep, lat, lon = data[e]
        sel = a.sel(lat=slice(bsel['lat'][0], bsel['lat'][1]))
        if bname == 'Pacific':
            sel = sel.where((sel.lon >= 120) | (sel.lon <= -70))
        elif bname == 'Southern':
            pass
        else:
            sel = sel.sel(lon=slice(bsel['lon'][0], bsel['lon'][1]))
        prof = sel.mean(dim=['lat', 'lon'], skipna=True).values
        ax.plot(prof, dep, color=COLS[e], lw=2.4, label=lab,
                linestyle=LS[e], solid_capstyle='round')
    ax.set_ylim(5500, 0)
    ax.set_title(bname, fontsize=19, fontweight='bold', pad=8)
    ax.text(0.03, 0.045, f'({letters[p]})', transform=ax.transAxes,
            fontsize=16, fontweight='bold')
    p += 1
    ax.set_xlabel('Ideal age (years)', fontsize=14)
    if c == 0:
        ax.set_ylabel('Depth (m)', fontsize=15)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.25, lw=0.6)
    if c == 0:
        ax.legend(frameon=False, fontsize=13, loc='lower right')

# ------------------------------------------- row 1, panel 5: deep-age bar summary
ax = fig.add_subplot(gs[0, 4])
bw = 0.16
xs = np.arange(len(BASINS))
for k, (e, lab) in enumerate(zip(EXPS, LABELS)):
    vals = []
    for bname, bsel in BASINS.items():
        a, dep, lat, lon = data[e]
        sel = a.sel(lat=slice(bsel['lat'][0], bsel['lat'][1]))
        if bname == 'Pacific':
            sel = sel.where((sel.lon >= 120) | (sel.lon <= -70))
        elif bname != 'Southern':
            sel = sel.sel(lon=slice(bsel['lon'][0], bsel['lon'][1]))
        prof = sel.mean(dim=['lat', 'lon'], skipna=True).values
        vals.append(np.nanmean(prof[dep >= 2000]))
    ax.bar(xs + (k - 2) * bw, vals, width=bw, color=COLS[e], label=lab,
           edgecolor='none')
ax.set_xticks(xs)
ax.set_xticklabels(list(BASINS.keys()), fontsize=12, rotation=20)
ax.set_ylabel('Ideal age below 2000 m (years)', fontsize=13)
ax.set_title('Deep-ocean mean', fontsize=19, fontweight='bold', pad=8)
ax.text(0.03, 0.93, f'({letters[p]})', transform=ax.transAxes,
        fontsize=16, fontweight='bold', va='top')
p += 1
ax.tick_params(labelsize=12)
ax.grid(axis='y', alpha=0.25, lw=0.6)
ax.legend(frameon=False, fontsize=11, ncol=2)

# ---------------------------------------------------- row 2: zonal-mean sections
lev = np.arange(0, 2101, 100)
for c, (e, lab) in enumerate(zip(EXPS, LABELS)):
    ax = fig.add_subplot(gs[1, c])
    a, dep, lat, lon = data[e]
    zm = a.mean(dim='lon', skipna=True).values
    cf = ax.contourf(lat, dep, zm, levels=lev, cmap='viridis', extend='max')
    ax.set_ylim(5500, 0)
    ax.set_title(lab, fontsize=19, fontweight='bold', pad=8)
    ax.text(0.03, 0.045, f'({letters[p]})', transform=ax.transAxes,
            fontsize=16, fontweight='bold', color='w')
    p += 1
    ax.set_xlabel('Latitude', fontsize=14)
    if c == 0:
        ax.set_ylabel('Depth (m)', fontsize=15)
    ax.tick_params(labelsize=12)

cax = fig.add_axes([0.30, 0.028, 0.42, 0.014])
cb = fig.colorbar(cf, cax=cax, orientation='horizontal',
                  ticks=np.arange(0, 2101, 400))
cb.set_label('Zonal-mean ideal age (years)', fontsize=14)
cb.ax.tick_params(labelsize=12)

fig.savefig('figures/figR3_age_global.pdf', dpi=400)
fig.savefig('figures/figR3_age_global.png', dpi=200)
print('saved figures/figR3_age_global.{pdf,png}')

# ---------------------------------------------------- summary numbers
print()
print('Basin-mean ideal age (years) below 2000 m')
print('%-10s' % 'BASIN' + ''.join('%9s' % l for l in LABELS))
for bname, bsel in BASINS.items():
    row = []
    for e in EXPS:
        a, dep, lat, lon = data[e]
        sel = a.sel(lat=slice(bsel['lat'][0], bsel['lat'][1]))
        if bname == 'Pacific':
            sel = sel.where((sel.lon >= 120) | (sel.lon <= -70))
        elif bname != 'Southern':
            sel = sel.sel(lon=slice(bsel['lon'][0], bsel['lon'][1]))
        prof = sel.mean(dim=['lat', 'lon'], skipna=True).values
        row.append(np.nanmean(prof[dep >= 2000]))
    print('%-10s' % bname + ''.join('%9.0f' % v for v in row))
