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
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

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

# ---- sizes are set for the printed reply letter (2026-11-27) -----------------
# The letter prints this figure 5.81 in wide (0.9 x A4 text width). All font
# sizes, line widths and panel positions below are given in points on that
# printed page; pt() converts them to the native figure size. Target: tick
# labels 7.5 pt, axis/colorbar labels and legend >= 8 pt, letters 9.5 pt bold.
PRINT_W = 5.81                    # printed width (in)
FIG_W = 7.0                       # native width (in); PNG at 600 dpi = 4200 px
S = PRINT_W / FIG_W               # printed size = native size * S
WP = PRINT_W * 72                 # printed width (pt)


def pt(v):
    return v / S


F_TICK, F_LAB, F_TITLE, F_LEG, F_LET = pt(7.5), pt(8.5), pt(9), pt(8), pt(9.5)
LW_AX = pt(0.6)
mpl.rcParams.update({
    'axes.linewidth': LW_AX, 'xtick.major.width': LW_AX,
    'ytick.major.width': LW_AX, 'xtick.major.size': pt(2.5),
    'ytick.major.size': pt(2.5), 'xtick.major.pad': pt(2),
    'ytick.major.pad': pt(2), 'axes.labelpad': pt(2.5),
    'text.color': 'k', 'axes.labelcolor': 'k',
    'xtick.color': 'k', 'ytick.color': 'k',
})

# panel grid in printed points (x from the left, y from the bottom)
L, R, GAP = 40, 15, 11             # margins and gap between panels
GAP_E = 48                         # room for the y axis of panel e
EW = 68                            # width of panel e (bars)
PW = (WP - L - R - 3 * GAP - GAP_E - EW) / 4   # profile width (~54 pt)
QW = (WP - L - 5 - 4 * GAP) / 5                # section width (~66 pt)
PH, QH = 95, 68                    # heights of row 1 (profiles) and row 2
Y2 = 65                            # bottom of row 2 (sections)
Y1 = Y2 + QH + 53                  # bottom of row 1
HP = Y1 + PH + 33                  # printed height (pt), incl. legend row
CB_W, CB_H = 240, 6


def box(x, y, w, h):
    return [x / WP, y / HP, w / WP, h / HP]


fig = plt.figure(figsize=(FIG_W, FIG_W * HP / WP))

letters = 'abcdefghij'
p = 0

# tick marks as before; fewer tick labels so they do not collide
XSTEP = {'Atlantic': 100, 'Indian': 200, 'Pacific': 500, 'Southern': 200}
XSHOW = {'Atlantic': (0, 300, 600), 'Indian': (0, 400, 800),
         'Pacific': (0, 1000), 'Southern': (0, 400, 800)}

# ---------------------------------------------------- row 1: basin profiles
for c, (bname, bsel) in enumerate(BASINS.items()):
    ax = fig.add_axes(box(L + c * (PW + GAP), Y1, PW, PH))
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
        ax.plot(prof, dep, color=COLS[e], lw=pt(1.0), label=lab,
                linestyle=LS[e], solid_capstyle='round')
    ax.set_ylim(5500, 0)
    ax.set_title(bname, fontsize=F_TITLE, fontweight='bold', pad=pt(3))
    ax.text(0.97, 0.98, f'({letters[p]})', transform=ax.transAxes,
            fontsize=F_LET, fontweight='bold', ha='right', va='top')
    p += 1
    ax.set_xlabel('Ideal age\n(years)', fontsize=F_LAB, fontweight='bold')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(XSTEP[bname]))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _, show=XSHOW[bname]: f'{v:.0f}' if round(v) in show else ''))
    ax.set_yticks(np.arange(0, 5001, 1000))
    if c == 0:
        ax.set_ylabel('Depth (m)', fontsize=F_LAB, fontweight='bold')
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=F_TICK)
    ax.grid(alpha=0.25, lw=pt(0.5))

# one legend for row 1 (profiles a-d and bars e use the same colours):
# each entry shows the profile line and the bar colour
handles = [(Line2D([], [], color=COLS[e], lw=pt(1.0), linestyle=LS[e]),
            Patch(facecolor=COLS[e], edgecolor='none')) for e in EXPS]
fig.legend(handles, LABELS, loc='upper center', ncol=5, frameon=False,
           bbox_to_anchor=((L + (WP - R)) / 2 / WP, 1 - 1 / HP),
           fontsize=F_LEG, handlelength=3.0, handletextpad=0.5,
           columnspacing=1.6, borderaxespad=0,
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)})

# ------------------------------------------- row 1, panel 5: deep-age bar summary
ax = fig.add_axes(box(WP - R - EW, Y1, EW, PH))
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
ax.set_xticklabels(list(BASINS.keys()), fontsize=F_TICK, rotation=45,
                   ha='right', rotation_mode='anchor')
ax.yaxis.set_major_locator(mticker.MultipleLocator(200))   # as before
ax.set_ylabel('Ideal age below\n2000 m (years)', fontsize=F_LAB,
              fontweight='bold')
ax.set_title('Deep-ocean mean', fontsize=F_TITLE, fontweight='bold',
             pad=pt(3))
ax.text(0.04, 0.98, f'({letters[p]})', transform=ax.transAxes,
        fontsize=F_LET, fontweight='bold', va='top')
p += 1
ax.tick_params(labelsize=F_TICK)
ax.grid(axis='y', alpha=0.25, lw=pt(0.5))

# ---------------------------------------------------- row 2: zonal-mean sections
lev = np.arange(0, 2101, 100)
for c, (e, lab) in enumerate(zip(EXPS, LABELS)):
    ax = fig.add_axes(box(L + c * (QW + GAP), Y2, QW, QH))
    a, dep, lat, lon = data[e]
    zm = a.mean(dim='lon', skipna=True).values
    cf = ax.contourf(lat, dep, zm, levels=lev, cmap='viridis', extend='max')
    ax.set_ylim(5500, 0)
    ax.set_title(lab, fontsize=F_TITLE, fontweight='bold', pad=pt(3))
    # letter above the panel, on the title baseline
    ax.annotate(f'({letters[p]})', xy=(0, 1), xycoords='axes fraction',
                xytext=(0, pt(3)), textcoords='offset points',
                fontsize=F_LET, fontweight='bold', ha='left', va='baseline')
    p += 1
    ax.set_xlabel('Latitude', fontsize=F_LAB, fontweight='bold')
    ax.set_xticks([-50, 0, 50])
    ax.set_yticks(np.arange(0, 5001, 1000))
    if c == 0:
        ax.set_ylabel('Depth (m)', fontsize=F_LAB, fontweight='bold')
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=F_TICK)

cax = fig.add_axes(box(L + (WP - L - 5 - CB_W) / 2, Y2 - 35, CB_W, CB_H))
cb = fig.colorbar(cf, cax=cax, orientation='horizontal',
                  ticks=np.arange(0, 2101, 400))
cb.set_label('Zonal-mean ideal age (years)', fontsize=F_LAB,
             fontweight='bold')
cb.ax.tick_params(labelsize=F_TICK)
cb.outline.set_linewidth(LW_AX)

fig.savefig('figures/figR3_age_global.pdf', dpi=400)
fig.savefig('figures/figR3_age_global.png', dpi=600)
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
