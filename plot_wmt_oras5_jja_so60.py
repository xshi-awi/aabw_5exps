#!/usr/bin/env python
"""
ORAS5 winter transformation south of 60 S, beside our model on the same domain.

This is the paper's own domain and season, rather than the sea ice sector and
annual mean used for the Pellichero comparison, so it answers a different
question: how does the reanalysis compare with the model where the paper
actually makes its claims?

Three panels, as before:
  (a) ORAS5, heat and E-P-R only, because its water flux has no sea ice term
  (b) our model with the same two terms, ice contribution removed
  (c) our model complete

(a) against (b) is like-for-like; (b) against (c) shows what the missing sea
ice term is worth. Neutral density on the x axis using the project's fit,
gamma = 1.0225 sigma_2 - 10.2445.
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
JJA = [5, 6, 7]          # June, July, August as month indices


def model_jja():
    """Model JJA over the same domain the paper integrates, south of 60 S."""
    d = xr.open_dataset('wmt_siz_pi.nc', decode_times=False)
    sig = d['sigma2_l_target'].values
    V = lambda n: -d[n].values[JJA].mean(axis=0) / 1e9
    P = 'so60_surface_ocean_flux_advective_negative_rhs_%s_salt'
    heat = V('so60_total_heat_heat')
    ice = V(P % 'sea_ice_melt') + V('so60_surface_exchange_flux_nonadvective_salt')
    oth = sum(V(P % x) for x in ('evaporation', 'snow', 'rain_and_ice', 'rivers'))
    d.close()
    return GAMMA(sig), heat, ice, oth


def oras5_jja():
    d = xr.open_dataset('wmt_oras5_so60.nc')
    sig = d['sigma2'].values
    h, f = d['heat_jja'].values, d['fresh_jja'].values
    d.close()
    return GAMMA(sig), h, f


def shade(ax, g, h, f):
    """Red where heat carries the total, blue where freshwater does.

    Classified along the drawn curve rather than per bin: the bins are 0.2 wide
    in gamma and the sign changes fall between bin centres.
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


gm, h_m, i_m, o_m = model_jja()
go, h_o, f_o = oras5_jja()

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

ymax = max(np.nanmax(np.abs(h + f)) for _, _, h, f, _ in
           [(a, b, c, d, e) for a, b, c, d, e in panels]) * 1.12

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

    gf = np.linspace(g[m][0], g[m][-1], 2000)
    tf = np.interp(gf, g[m], (h + f)[m])
    cr = np.where(np.diff(np.sign(tf)) > 0)[0]
    if len(cr):
        ax.axvline(gf[cr[0]], color='#E8000B', lw=1.6, ls='--', zorder=2)

    ax.set_xlim(*XLIM)
    ax.set_ylim(-ymax, ymax)       # one scale across panels, so sizes compare
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
fig.suptitle('JJA, south of 60$^\\circ$S', fontsize=13, y=0.975)

fig.savefig('figures/figR19_oras5_jja_so60.pdf', dpi=400)
fig.savefig('figures/figR19_oras5_jja_so60.png', dpi=200)
print('saved figures/figR19_oras5_jja_so60.{pdf,png}')

for lab, g, h, f in (('ORAS5', go, h_o, f_o),
                     ('model, no ice', gm, h_m, o_m),
                     ('model, complete', gm, h_m, o_m + i_m)):
    m = (g >= XLIM[0]) & (g <= XLIM[1])
    t = (h + f)[m]
    dense = m & (g >= 27.7)
    print('%-16s peak %+7.2f Sv at gamma %.2f | gamma>=27.7: heat %+.2f fw %+.2f'
          % (lab, np.nanmax(t), g[m][np.nanargmax(t)],
             np.nansum(h[dense]), np.nansum(f[dense])))
