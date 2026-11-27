#!/usr/bin/env python
"""
Global meridional overturning streamfunction for the five climate states,
computed from the model's vertical velocity on the geographic mesh
(data_moc/calc_moc_5exps.py, climdiag.moc_depth), mean over the last 20 years of
each run's 100-year analysis period (PI/MH/LIG 2380-2399, LGM 2480-2499,
MIS3 2580-2599).

2026-11-08 (D6): no longer reads production/{exp}_age/diag_gmoc.nc. For PI, MH and
LIG that file was binned by rotated latitude, its row labels sit one bin north, and
the LIG file ends in 2371. See data_moc/CLAUDE.md.

Requested by Reviewer #3, who noted that the overturning can be computed
directly in the model and should be shown rather than inferred from ideal age.

Row 1: absolute streamfunction, all five states.
Row 2: anomalies relative to PI for MH, LIG, LGM, MIS3.
"""
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patheffects as pe

EXPS = ['pi', 'mh', 'lig', 'lgm', 'mis']
LABELS = ['PI', 'MH', 'LIG', 'LGM', 'MIS3']
MOC_FILE = 'data_moc/moc_5exps_geo.nc'

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'pdf.fonttype': 42,
})


_ds = xr.open_dataset(MOC_FILE)


def load(lab):
    """mean psi (depth, lat), latitudes crossed, interface depths (m, negative down
    as in the old diag_gmoc.nc convention so the index code below is unchanged)."""
    m = _ds['moc'].sel(exp=lab)
    return m.values, m['lat'].values, -m['depth'].values


data = {e: load(lab) for e, lab in zip(EXPS, LABELS)}

# ---- sizes are set for the printed reply letter (2026-11-27) -----------------
# The letter prints this figure 5.81 in wide (0.9 x A4 text width). All font
# sizes, line widths and panel positions below are given in points on that
# printed page; pt() converts them to the native figure size. Target: tick
# labels 7.5 pt, axis/colorbar labels and notes >= 8 pt, letters 9.5 pt bold.
PRINT_W = 5.81                    # printed width (in)
FIG_W = 7.0                       # native width (in); PNG at 600 dpi = 4200 px
S = PRINT_W / FIG_W               # printed size = native size * S
WP = PRINT_W * 72                 # printed width (pt)


def pt(v):
    return v / S


F_TICK, F_LAB, F_TITLE, F_NOTE, F_LET = pt(7.5), pt(8.5), pt(9), pt(8), pt(9.5)
LW_AX, LW_CONT = pt(0.6), pt(0.5)
# thin white halo so letters and notes stay legible over contour lines
HALO = [pe.withStroke(linewidth=pt(1.5), foreground='w')]
mpl.rcParams.update({
    'axes.linewidth': LW_AX, 'xtick.major.width': LW_AX,
    'ytick.major.width': LW_AX, 'xtick.major.size': pt(2.5),
    'ytick.major.size': pt(2.5), 'xtick.major.pad': pt(2),
    'ytick.major.pad': pt(2), 'axes.labelpad': pt(2.5),
    'text.color': 'k', 'axes.labelcolor': 'k',
    'xtick.color': 'k', 'ytick.color': 'k',
})

# panel grid in printed points (x from the left, y from the bottom)
L, R, GAP = 42, 5, 9                            # margins and gap between panels
PW = (WP - L - R - 4 * GAP) / 5                 # panel width (~67 pt)
PH = 68                                         # panel height
Y2 = 65                                         # bottom of row 2 (anomalies)
Y1 = Y2 + PH + 76                               # bottom of row 1 (absolute)
HP = Y1 + PH + 15                               # printed height (pt)
CB_W, CB_H = 240, 6                             # colorbar length and thickness
CB_X = L + (WP - L - R - CB_W) / 2


def box(x, y, w, h):
    return [x / WP, y / HP, w / WP, h / HP]


fig = plt.figure(figsize=(FIG_W, FIG_W * HP / WP))

lev_abs = np.arange(-24, 24.1, 2)
lev_anm = np.arange(-12, 12.1, 1)
letters = 'abcdefghi'
p = 0

pi_m, pi_lat, pi_dep = data['pi']


def style(ax, c):
    ax.set_ylim(6000, 0)
    ax.set_xlim(-78, 80)
    ax.set_xticks([-50, 0, 50])
    ax.set_yticks(np.arange(0, 6001, 1000))
    ax.set_xlabel('Latitude', fontsize=F_LAB, fontweight='bold')
    if c == 0:
        ax.set_ylabel('Depth (m)', fontsize=F_LAB, fontweight='bold')
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=F_TICK)


for c, (e, lab) in enumerate(zip(EXPS, LABELS)):
    m, lats, deps = data[e]
    ax = fig.add_axes(box(L + c * (PW + GAP), Y1, PW, PH))
    cf = ax.contourf(lats, -deps, m, levels=lev_abs, cmap='RdBu_r',
                     norm=TwoSlopeNorm(vcenter=0, vmin=-24, vmax=24),
                     extend='both')
    ax.contour(lats, -deps, m, levels=lev_abs, colors='k',
               linewidths=LW_CONT, alpha=0.35)
    style(ax, c)
    ax.set_title(lab, fontsize=F_TITLE, fontweight='bold', pad=pt(3))
    ax.text(0.03, 0.035, f'({letters[p]})', transform=ax.transAxes,
            fontsize=F_LET, fontweight='bold', path_effects=HALO)
    p += 1
    # AABW cell strength
    kz = deps <= -2000
    js = lats < -30
    aabw = np.nanmin(m[np.ix_(kz, js)])
    ax.text(0.97, 0.035, f'AABW\n{aabw:.1f} Sv', transform=ax.transAxes,
            ha='right', va='bottom', multialignment='right',
            fontsize=F_NOTE, color='k', linespacing=1.1, path_effects=HALO)

cax = fig.add_axes(box(CB_X, Y1 - 35, CB_W, CB_H))
cb = fig.colorbar(cf, cax=cax, orientation='horizontal',
                  ticks=np.arange(-24, 25, 8))
cb.set_label('Overturning streamfunction (Sv)', fontsize=F_LAB,
             fontweight='bold')
cb.ax.tick_params(labelsize=F_TICK)
cb.outline.set_linewidth(LW_AX)

for c, (e, lab) in enumerate(zip(EXPS[1:], LABELS[1:])):
    m, lats, deps = data[e]
    ax = fig.add_axes(box(L + (c + 1) * (PW + GAP), Y2, PW, PH))
    anm = m - pi_m
    cf2 = ax.contourf(lats, -deps, anm, levels=lev_anm, cmap='PuOr_r',
                      norm=TwoSlopeNorm(vcenter=0, vmin=-12, vmax=12),
                      extend='both')
    ax.contour(lats, -deps, anm, levels=lev_anm, colors='k',
               linewidths=LW_CONT, alpha=0.35)
    style(ax, c)
    ax.set_title(f'{lab} $-$ PI', fontsize=F_TITLE, fontweight='bold',
                 pad=pt(3))
    ax.text(0.03, 0.035, f'({letters[p]})', transform=ax.transAxes,
            fontsize=F_LET, fontweight='bold', path_effects=HALO)
    p += 1

cax2 = fig.add_axes(box(CB_X, Y2 - 35, CB_W, CB_H))
cb2 = fig.colorbar(cf2, cax=cax2, orientation='horizontal',
                   ticks=np.arange(-12, 13, 4))
cb2.set_label('Streamfunction anomaly (Sv)', fontsize=F_LAB,
              fontweight='bold')
cb2.ax.tick_params(labelsize=F_TICK)
cb2.outline.set_linewidth(LW_AX)

fig.savefig('figures/figR2_moc_5exps.pdf', dpi=400)
fig.savefig('figures/figR2_moc_5exps.png', dpi=600)
print('saved figures/figR2_moc_5exps.{pdf,png}')

print()
print('%-6s %10s %10s' % ('EXP', 'AABW(Sv)', 'upper(Sv)'))
for e, lab in zip(EXPS, LABELS):
    m, lats, deps = data[e]
    kz = deps <= -2000
    js = lats < -30
    aabw = np.nanmin(m[np.ix_(kz, js)])
    kn = (deps <= -500) & (deps >= -4000)
    jn = (lats > 20) & (lats < 70)   # 20-70N (was 0-70N until 2026-11-08, D8)
    upper_cell = np.nanmax(m[np.ix_(kn, jn)])
    print('%-6s %10.1f %10.1f' % (lab, aabw, upper_cell))
