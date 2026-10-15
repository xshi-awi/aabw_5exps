#!/usr/bin/env python
"""
ORAS5 transformation beside our model's, in the style of Fig. R1.1a.

Three curves, chosen so the missing term is visible rather than argued about:

  (a) ORAS5, heat and E-P-R only, because its water flux carries no sea ice
      thermodynamic term
  (b) our model with the same two terms, i.e. the ice contribution removed
  (c) our model complete

(a) against (b) is a like-for-like comparison of what the reanalysis can
actually supply. (b) against (c) shows what the missing term is worth.

Annual mean, September sea ice zone, neutral density on the x axis using the
same fit as elsewhere in this project, gamma = 1.0225 sigma_2 - 10.2445.
"""
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})

GAMMA = lambda x: 1.0225 * x - 10.2445
XLIM = (26.3, 28.9)
SHADE = {0: '#F7D4D4', 1: '#D6E6F2'}


def model():
    d = xr.open_dataset('wmt_siz_pi.nc', decode_times=False)
    sig = d['sigma2_l_target'].values
    V = lambda n: -d[n].values.mean(axis=0) / 1e9
    P = 'siz_surface_ocean_flux_advective_negative_rhs_%s_salt'
    heat = V('siz_total_heat_heat')
    ice = V(P % 'sea_ice_melt') + V('siz_surface_exchange_flux_nonadvective_salt')
    oth = sum(V(P % x) for x in ('evaporation', 'snow', 'rain_and_ice', 'rivers'))
    d.close()
    return GAMMA(sig), heat, ice, oth


def oras5():
    d = xr.open_dataset('wmt_oras5_siz.nc')
    sig = d['sigma2'].values
    h = d['heat_annual'].values
    f = d['fresh_annual'].values
    d.close()
    return GAMMA(sig), h, f


def shade(ax, g, h, f):
    """Red where heat carries the total, blue where freshwater does."""
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


gm, h_m, i_m, o_m = model()
go, h_o, f_o = oras5()

panels = [
    ('(a)  ORAS5 reanalysis, 1990-2009\n      heat and E$-$P$-$R only',
     go, h_o, f_o, None),
    ('(b)  This study, PI, same two terms\n      sea ice contribution removed',
     gm, h_m, o_m, None),
    ('(c)  This study, PI, complete',
     gm, h_m, o_m + i_m, i_m),
]

fig = plt.figure(figsize=(18, 5.6))
gs = fig.add_gridspec(1, 3, wspace=0.22, left=0.05, right=0.99,
                      top=0.80, bottom=0.26)

for k, (title, g, h, f, ice) in enumerate(panels):
    ax = fig.add_subplot(gs[0, k])
    m = (g >= XLIM[0]) & (g <= XLIM[1])
    shade(ax, g[m], h[m], f[m])
    ax.axhline(0, color='#999999', lw=0.8, zorder=1)
    ax.plot(g[m], (h + f)[m], color='#000000', lw=2.4, label='Total surface flux', zorder=3)
    ax.plot(g[m], h[m], color='#E8000B', lw=2.0, label='Surface heat flux', zorder=3)
    ax.plot(g[m], f[m], color='#1F3FD4', lw=2.0, label='Surface fresh water flux', zorder=3)
    if ice is not None:
        ax.plot(g[m], ice[m], color='#1B7F3B', lw=1.8, ls='--',
                label='Sea ice component', zorder=3)

    # the density where the total changes sign, as in their Fig. 3a
    gf = np.linspace(g[m][0], g[m][-1], 2000)
    tf = np.interp(gf, g[m], (h + f)[m])
    cr = np.where(np.diff(np.sign(tf)) > 0)[0]
    if len(cr):
        ax.axvline(gf[cr[0]], color='#E8000B', lw=1.6, ls='--', zorder=2)

    ax.set_xlim(*XLIM)
    # one y scale across the three panels, otherwise the magnitudes cannot be
    # compared by eye, which is the point of putting them side by side
    ax.set_ylim(-13.5, 15.5)
    ax.set_xticks(np.arange(26.3, 28.95, 0.4))
    ax.set_xticklabels(['%.1f' % v for v in np.arange(26.3, 28.95, 0.4)], fontsize=9)
    ax.set_xlabel(r'Neutral density, $\gamma$ (kg m$^{-3}$)', fontsize=12)
    if k == 0:
        ax.set_ylabel('Transformation rate (Sv)', fontsize=12)
    ax.set_title(title, fontsize=12.5, fontweight='bold', loc='left', pad=10)
    ax.tick_params(labelsize=10)

handles, labels = fig.axes[-1].get_legend_handles_labels()
patches = [Patch(facecolor=SHADE[0], label='Heat driven'),
           Patch(facecolor=SHADE[1], label='Freshwater driven')]
fig.legend(handles + patches, labels + [p.get_label() for p in patches],
           frameon=False, fontsize=10, ncol=6, loc='lower center',
           bbox_to_anchor=(0.5, 0.015), columnspacing=1.6, handlelength=1.9)

fig.savefig('figures/figR18_oras5_vs_model.pdf', dpi=400)
fig.savefig('figures/figR18_oras5_vs_model.png', dpi=200)
print('saved figures/figR18_oras5_vs_model.{pdf,png}')

# numbers for the caption
for lab, g, h, f in (('ORAS5', go, h_o, f_o),
                     ('model, no ice', gm, h_m, o_m),
                     ('model, complete', gm, h_m, o_m + i_m)):
    m = (g >= XLIM[0]) & (g <= XLIM[1])
    t = (h + f)[m]
    print('%-16s peak %+6.2f Sv at gamma %.2f, min %+6.2f Sv'
          % (lab, np.nanmax(t), g[m][np.nanargmax(t)], np.nanmin(t)))
    dense = m & (g >= 27.7)
    print('%-16s   gamma>=27.7: heat %+.2f, freshwater %+.2f Sv'
          % ('', np.nansum(h[dense]), np.nansum(f[dense])))
