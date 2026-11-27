#!/usr/bin/env python
"""
The transformation comparison on one page, three rows.

Row 1   our PI transformation in the sea ice zone, annual mean, beside the
        published estimate of Pellichero et al. (2018). Shaded by which term
        carries the total, since all terms are present.
Row 2   the same domain and season, our model against ORAS5. The model panel
        has the sea ice contribution removed so that the two sides carry the
        same two terms.
Row 3   the paper's own domain and season, JJA south of 60 S, same pairing.

Rows 2 and 3 are not shaded. ORAS5 has no sea ice thermodynamic term, so a
heat-versus-freshwater classification there would describe an incomplete
freshwater flux and read as a physical result rather than a missing field.

Neutral density on the x axis throughout, gamma = 1.0225 sigma_2 - 10.2445,
the fit used elsewhere in this project. Our simulated surface water is lighter
than observed, so equal gamma does not mean the same water mass.
"""
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})

GAMMA = lambda x: 1.0225 * x - 10.2445
XLIM = (26.3, 28.9)
SHADE = {0: '#F7D4D4', 1: '#D6E6F2'}
OBS = 'nc_paper/SUBMISSION/p3.png'
JJA = [5, 6, 7]

C_TOT, C_HEAT, C_FW, C_ICE = '#000000', '#E8000B', '#1F3FD4', '#1B7F3B'

# The letter prints these figures 5.81 in wide (0.9 x textwidth on A4) and caps the height
# at 5.93 in. So the figure is drawn 7 in wide (scale 0.83) and no taller than 1.02 x its
# width; the font sizes below then print at about 7.8 / 8.3 / 9.1 pt (2026-11-27).
W = 7.0
FS_TICK, FS_LAB, FS_LEG, FS_TITLE, FS_HEAD = 9.4, 10, 10, 11, 10.4


def model(prefix, months=None):
    """Model transformation; months=None means the annual mean."""
    d = xr.open_dataset('wmt_siz_pi.nc', decode_times=False)
    sig = d['sigma2_l_target'].values
    sel = (lambda a: a.mean(axis=0)) if months is None else (lambda a: a[months].mean(axis=0))
    V = lambda n: -sel(d[n].values) / 1e9
    P = prefix + '_surface_ocean_flux_advective_negative_rhs_%s_salt'
    heat = V(prefix + '_total_heat_heat')
    ice = V(P % 'sea_ice_melt') + V(prefix + '_surface_exchange_flux_nonadvective_salt')
    oth = sum(V(P % x) for x in ('evaporation', 'snow', 'rain_and_ice', 'rivers'))
    d.close()
    return GAMMA(sig), heat, ice, oth


def oras5(fn, season):
    d = xr.open_dataset(fn)
    sig = d['sigma2'].values
    h, f = d['heat_%s' % season].values, d['fresh_%s' % season].values
    d.close()
    return GAMMA(sig), h, f


def shade(ax, g, h, f):
    """Red where heat carries the total, blue where freshwater does.

    Classified along the drawn curve, not per bin: the bins are 0.2 wide in
    gamma and the sign changes fall between bin centres.
    """
    gf = np.linspace(g[0], g[-1], 2000)
    hf, ff = np.interp(gf, g, h), np.interp(gf, g, f)
    tf = hf + ff
    qh = np.where(np.sign(hf) == np.sign(tf), np.abs(hf), -1.0)
    qf = np.where(np.sign(ff) == np.sign(tf), np.abs(ff), -1.0)
    who = np.where(qh >= qf, 0, 1)
    e = np.r_[0, np.where(np.diff(who) != 0)[0] + 1, len(who)]
    for a, b in zip(e[:-1], e[1:]):
        ax.axvspan(gf[a], gf[min(b, len(gf) - 1)], color=SHADE[who[a]],
                   lw=0, zorder=0)


def curve(ax, g, h, f, ice=None, zeroline=True):
    m = (g >= XLIM[0]) & (g <= XLIM[1])
    ax.axhline(0, color='#999999', lw=0.8, zorder=1)
    ax.plot(g[m], (h + f)[m], color=C_TOT, lw=1.9, label='Total surface flux', zorder=3)
    ax.plot(g[m], h[m], color=C_HEAT, lw=1.5, label='Surface heat flux', zorder=3)
    ax.plot(g[m], f[m], color=C_FW, lw=1.5, label='Surface fresh water flux', zorder=3)
    if ice is not None:
        ax.plot(g[m], ice[m], color=C_ICE, lw=1.4, ls='--',
                label='Sea ice component', zorder=3)
    if zeroline:
        # Their dashed line marks where water stops becoming lighter and starts
        # becoming denser. That only means something when the curve has a real
        # negative lobe: in JJA the total is positive almost everywhere and the
        # first sign change is a 0.1 Sv wiggle at the light end, so skip it.
        gf = np.linspace(g[m][0], g[m][-1], 2000)
        tf = np.interp(gf, g[m], (h + f)[m])
        cr = np.where(np.diff(np.sign(tf)) > 0)[0]
        if len(cr) and np.nanmin(tf) < -0.05 * np.nanmax(np.abs(tf)):
            ax.axvline(gf[cr[0]], color=C_HEAT, lw=1.2, ls='--', zorder=2)
    ax.set_xlim(*XLIM)
    ax.set_xticks(np.arange(26.3, 28.95, 0.4))
    ax.set_xticklabels(['%.1f' % v for v in np.arange(26.3, 28.95, 0.4)], fontsize=FS_TICK)
    ax.tick_params(labelsize=FS_TICK, labelcolor='black')
    # short axes in the merged figure get every other y label, so the labels stay apart
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 4, 5, 10]) if AX_H < 1.3
                               else MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]))


# ------------------------------------------------------------------ the data
g_siz, h_siz, i_siz, o_siz = model('siz')                 # ice zone, annual
g_60, h_60, i_60, o_60 = model('so60', months=JJA)        # south of 60S, JJA
go_siz, ho_siz, fo_siz = oras5('wmt_oras5_siz.nc', 'annual')
go_60, ho_60, fo_60 = oras5('wmt_oras5_so60.nc', 'jja')

# MODE (first argument): 'all' draws the three rows on one page (figR20_wmt_merged);
# 'obs' draws row 1 only and 'rean' rows 2-3 only, for the reply to Reviewer 3, who
# asked for the observational and the reanalysis comparison separately (2026-11-22).
# Panel letters are (a)-(f) in 'all' (the reply to Reviewer 1 cites them); in 'rean' the two
# rows are relettered (a)-(d), as the user asked on 2026-11-25.
import sys
MODE = sys.argv[1] if len(sys.argv) > 1 else 'all'
ROWS = {'all': [0, 1, 2], 'obs': [0], 'rean': [1, 2]}[MODE]
OUT = {'all': 'figR20_wmt_merged', 'obs': 'figR20a_wmt_obs', 'rean': 'figR20b_wmt_oras5'}[MODE]

# Layout in inches from the top. Each row: header line, two-line panel titles, axes,
# then tick labels and x title. The text heights are fixed and the axes take the rest.
HEAD_H = 1.25 * FS_HEAD / 72 + 0.05
TITLE_H = 2 * 1.2 * FS_TITLE / 72 + 6 / 72
BELOW, GAP_ROW, TOP, BOT = 0.50, 0.10, 0.04, 0.02
LEG_H = 0.25 if MODE == "rean" else 0.44        # one legend line, or two
YLAB2 = MODE == 'all'                           # short axes: y title on two lines
YLABEL = 'Transformation\nrate (Sv)' if YLAB2 else 'Transformation rate (Sv)'
L = 0.80 if YLAB2 else 0.66
GAPC, R = 0.47, 0.06
COLW = (W - L - GAPC - R) / 2
HAS_HEAD = [not (MODE == 'obs' and r == 0) for r in ROWS]   # 2026-11-26 user: no heading on R13
fixed = TOP + BOT + LEG_H - GAP_ROW + sum((HEAD_H if hh else 0) + TITLE_H + BELOW + GAP_ROW
                                          for hh in HAS_HEAD)
AX_H = {'all': (1.02 * W - fixed) / 3, 'obs': 1.95, 'rean': 1.6}[MODE]
H = fixed + AX_H * len(ROWS)
fig = plt.figure(figsize=(W, H))

ROW_Y = []                     # (header top, axes top) in inches from the top
y = TOP
for hh in HAS_HEAD:
    hy = y if hh else None
    y += (HEAD_H if hh else 0) + TITLE_H
    ROW_Y.append((hy, y))
    y += AX_H + BELOW + GAP_ROW


def box(x, top, w, h):
    return [x / W, 1 - (top + h) / H, w / W, h / H]


def header(top, text):
    fig.text(0.5, 1 - top / H, text, ha='center', va='top',
             fontsize=FS_HEAD, fontweight='bold', color='black')


def labels(ax, ylab=True):
    ax.set_xlabel(r'Neutral density, $\gamma$ (kg m$^{-3}$)', fontsize=FS_LAB, fontweight='bold')
    if ylab:
        ax.set_ylabel(YLABEL, fontsize=FS_LAB, fontweight='bold')


def title(ax, text):
    ax.set_title(text, fontsize=FS_TITLE, fontweight='bold', loc='left', pad=6, color='black')


handles = None
if 0 in ROWS:
    # ---- row 1: ours against the published observational estimate
    hy, at = ROW_Y[ROWS.index(0)]
    if MODE == 'obs':          # one row only: give the published panel more width
        wa = 2.62
        xb = L + wa + 0.25
    else:
        wa = COLW
        xb = L + COLW + GAPC
    ax = fig.add_axes(box(L, at, wa, AX_H))
    shade(ax, g_siz, h_siz, o_siz + i_siz)
    curve(ax, g_siz, h_siz, o_siz + i_siz, ice=i_siz)
    ax.set_ylim(-10.5, 10.5)
    labels(ax)
    title(ax, '(a)  This study, PI,\nall terms including sea ice')
    if hy is not None:
        header(hy, 'Sea ice zone, annual mean  |  model against the published estimate')
    handles = ax.get_legend_handles_labels()

    # the published figure has no x title of ours below it, so it may use that height too
    ax = fig.add_axes(box(xb, at, W - xb - 0.02, AX_H + BELOW))
    img = mpimg.imread(OBS)
    hh, ww = img.shape[:2]
    ax.imshow(img[0:int(hh * 0.355), 0:int(ww * 0.60)])
    ax.set_anchor('NW')
    ax.axis('off')
    title(ax, '(b)  Pellichero et al. (2018),\ntheir Fig. 3a')

# ---- rows 2 and 3: like-for-like against ORAS5, no shading
rows = [
    (1, 'Sea ice zone, annual mean  |  model against ORAS5, sea ice term absent from both',
     ('(c)  This study, PI,\nsea ice contribution removed', g_siz, h_siz, o_siz),
     ('(d)  ORAS5 reanalysis\n1999-2018, heat and E$-$P$-$R', go_siz, ho_siz, fo_siz)),
    (2, 'South of 60$^\\circ$S, JJA  |  model against ORAS5, sea ice term absent from both',
     ('(e)  This study, PI,\nsea ice contribution removed', g_60, h_60, o_60),
     ('(f)  ORAS5 reanalysis\n1999-2018, heat and E$-$P$-$R', go_60, ho_60, fo_60)),
]

for r, rowlab, left, right in rows:
    if r not in ROWS:
        continue
    hy, at = ROW_Y[ROWS.index(r)]
    # scale from every curve, not just the total: in (d) the heat line rises
    # well above the total and was being clipped
    lim = 1.1 * max(max(np.nanmax(np.abs(p[2])), np.nanmax(np.abs(p[3])),
                        np.nanmax(np.abs(p[2] + p[3]))) for p in (left, right))
    for c, (ttl, g, h, f) in enumerate((left, right)):
        ax = fig.add_axes(box(L + c * (COLW + GAPC), at, COLW, AX_H))
        curve(ax, g, h, f)
        ax.set_ylim(-lim, lim)
        labels(ax, ylab=(c == 0))
        if c == 0:
            header(hy, rowlab)
            if handles is None:
                handles = ax.get_legend_handles_labels()
        if MODE == 'rean':
            ttl = '(%s)' % 'abcd'['cdef'.index(ttl[1])] + ttl[3:]
        title(ax, ttl)

hl, ll = handles
if 0 in ROWS:
    patches = [Patch(facecolor=SHADE[0], label='Heat driven (a only)'),
               Patch(facecolor=SHADE[1], label='Freshwater driven (a only)')]
    hl, ll = hl + patches, ll + [p.get_label() for p in patches]
fig.legend(hl, ll, frameon=False, fontsize=FS_LEG, ncol=3, loc='lower center',
           bbox_to_anchor=(0.5, BOT / H), columnspacing=1.2, handlelength=1.8,
           handletextpad=0.5, labelspacing=0.3, borderaxespad=0)

fig.savefig('figures/%s.pdf' % OUT, dpi=400)
fig.savefig('figures/%s.png' % OUT, dpi=400)
print('saved figures/%s.{pdf,png}  %.2f x %.2f in, axes height %.2f in' % (OUT, W, H, AX_H))
