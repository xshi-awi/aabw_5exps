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

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})

GAMMA = lambda x: 1.0225 * x - 10.2445
XLIM = (26.3, 28.9)
SHADE = {0: '#F7D4D4', 1: '#D6E6F2'}
OBS = 'nc_paper/SUBMISSION/p3.png'
JJA = [5, 6, 7]

C_TOT, C_HEAT, C_FW, C_ICE = '#000000', '#E8000B', '#1F3FD4', '#1B7F3B'


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
    ax.plot(g[m], (h + f)[m], color=C_TOT, lw=2.3, label='Total surface flux', zorder=3)
    ax.plot(g[m], h[m], color=C_HEAT, lw=1.9, label='Surface heat flux', zorder=3)
    ax.plot(g[m], f[m], color=C_FW, lw=1.9, label='Surface fresh water flux', zorder=3)
    if ice is not None:
        ax.plot(g[m], ice[m], color=C_ICE, lw=1.7, ls='--',
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
            ax.axvline(gf[cr[0]], color=C_HEAT, lw=1.5, ls='--', zorder=2)
    ax.set_xlim(*XLIM)
    ax.set_xticks(np.arange(26.3, 28.95, 0.4))
    ax.set_xticklabels(['%.1f' % v for v in np.arange(26.3, 28.95, 0.4)], fontsize=8.5)
    ax.tick_params(labelsize=9)


# ------------------------------------------------------------------ the data
g_siz, h_siz, i_siz, o_siz = model('siz')                 # ice zone, annual
g_60, h_60, i_60, o_60 = model('so60', months=JJA)        # south of 60S, JJA
go_siz, ho_siz, fo_siz = oras5('wmt_oras5_siz.nc', 'annual')
go_60, ho_60, fo_60 = oras5('wmt_oras5_so60.nc', 'jja')

fig = plt.figure(figsize=(14, 15.6))
gs = fig.add_gridspec(3, 2, hspace=0.52, wspace=0.18,
                      left=0.075, right=0.98, top=0.925, bottom=0.082)

# ---- row 1: ours against the published observational estimate
fig.text(0.5, 0.963,
         'Sea ice zone, annual mean  |  model against the published estimate',
         ha='center', fontsize=15, fontweight='bold', color='#111111')

ax = fig.add_subplot(gs[0, 0])
shade(ax, g_siz, h_siz, o_siz + i_siz)
curve(ax, g_siz, h_siz, o_siz + i_siz, ice=i_siz)
ax.set_ylim(-10.5, 10.5)
ax.set_ylabel('Transformation rate (Sv)', fontsize=11)
ax.set_title('(a)  This study, PI, all terms including sea ice',
             fontsize=12, fontweight='bold', loc='left', pad=8)
row1_handles, row1_labels = ax.get_legend_handles_labels()

ax = fig.add_subplot(gs[0, 1])
img = mpimg.imread(OBS)
hh, ww = img.shape[:2]
ax.imshow(img[0:int(hh * 0.355), 0:int(ww * 0.60)])
ax.axis('off')
ax.set_title('(b)  Pellichero et al. (2018), their Fig. 3a',
             fontsize=12, fontweight='bold', loc='left', pad=8)

# ---- rows 2 and 3: like-for-like against ORAS5, no shading
rows = [
    (1, 'Sea ice zone, annual mean  |  model against ORAS5, sea ice term absent from both',
     ('(c)  This study, PI, sea ice contribution removed', g_siz, h_siz, o_siz),
     ('(d)  ORAS5 reanalysis 1999-2018, heat and E$-$P$-$R', go_siz, ho_siz, fo_siz)),
    (2, 'South of 60$^\\circ$S, JJA  |  model against ORAS5, sea ice term absent from both',
     ('(e)  This study, PI, sea ice contribution removed', g_60, h_60, o_60),
     ('(f)  ORAS5 reanalysis 1999-2018, heat and E$-$P$-$R', go_60, ho_60, fo_60)),
]

for r, rowlab, left, right in rows:
    # scale from every curve, not just the total: in (d) the heat line rises
    # well above the total and was being clipped
    lim = 1.1 * max(max(np.nanmax(np.abs(p[2])), np.nanmax(np.abs(p[3])),
                        np.nanmax(np.abs(p[2] + p[3]))) for p in (left, right))
    for c, (title, g, h, f) in enumerate((left, right)):
        ax = fig.add_subplot(gs[r, c])
        curve(ax, g, h, f)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(r'Neutral density, $\gamma$ (kg m$^{-3}$)', fontsize=11)
        if c == 0:
            ax.set_ylabel('Transformation rate (Sv)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left', pad=8)
    fig.text(0.5, {1: 0.655, 2: 0.345}[r], rowlab, ha='center',
             fontsize=15, fontweight='bold', color='#111111')

fig.axes[0].set_xlabel(r'Neutral density, $\gamma$ (kg m$^{-3}$)', fontsize=11)

patches = [Patch(facecolor=SHADE[0], label='Heat driven (a only)'),
           Patch(facecolor=SHADE[1], label='Freshwater driven (a only)')]
fig.legend(row1_handles + patches,
           row1_labels + [p.get_label() for p in patches],
           frameon=False, fontsize=10, ncol=6, loc='lower center',
           bbox_to_anchor=(0.5, 0.012), columnspacing=1.5, handlelength=1.9)

fig.savefig('figures/figR20_wmt_merged.pdf', dpi=400)
fig.savefig('figures/figR20_wmt_merged.png', dpi=200)
print('saved figures/figR20_wmt_merged.{pdf,png}')

for lab, g, h, f in (('(c) model no ice, siz annual', g_siz, h_siz, o_siz),
                     ('(d) ORAS5 siz annual', go_siz, ho_siz, fo_siz),
                     ('(e) model no ice, 60S JJA', g_60, h_60, o_60),
                     ('(f) ORAS5 60S JJA', go_60, ho_60, fo_60)):
    m = (g >= XLIM[0]) & (g <= XLIM[1])
    t = (h + f)[m]
    print('%-30s peak %+7.2f Sv at gamma %.2f' % (lab, np.nanmax(t), g[m][np.nanargmax(t)]))
