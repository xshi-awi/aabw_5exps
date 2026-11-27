#!/usr/bin/env python
"""
Reviewer-response figure: sensitivity of the thermal/haline partition to the
choice of integration domain.

Row 1: annual-mean WMT integrated south of 60 S (the domain used in the paper).
Row 2: annual-mean WMT integrated over the seasonal sea ice zone only
       (climatological max sea ice concentration > 15%), which is the domain
       comparable to Pellichero et al. (2018).

The point of the figure is that restricting the integration to the ice-covered
sector raises the haline share, as expected, but does not reverse the
interglacial thermal dominance, and leaves the glacial haline dominance
essentially untouched. The regime shift is therefore not an artefact of the
integration domain.

2026-11-27 (user, 12:04): sigma2 bins 0.1 kg m-3 wide (wmt_siz_<exp>_bin0p1.nc from
calc_wmt_siz_pellichero.py <exp> 0.1) in place of the 0.2 files; and text enlarged for
the manuscript, where the figure prints at \textwidth = 5.15 in. Font sizes are given
as printed points through pt(). At that width the thermal share and the peak no longer
fit inside the panels without covering curves, so they are printed under each panel
together with the domain area. x ticks are labelled every 1 kg m-3 (minor ticks every
0.5), and the panel letters sit above the panels. Previous (0.2 bins): figures/versions/figR1_wmt_domain_sensitivity_v20261127bin0p2.*
"""
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import offset_copy

WMT_FILE = 'wmt_siz100_{exp}_bin0p1.nc'  # 0.1 kg m-3 bins, 100 years of monthly output (user, 2026-11-27 12:44; calc_wmt_siz_100years.py)

P_W, P_H = 5.15, 4.59          # printed size (in) at \textwidth in the manuscript
FIG_W = 7.0
SCALE = P_W / FIG_W            # printed / saved width


def pt(p):
    """native size that prints as p points"""
    return p / SCALE

EXPS = ['pi', 'mh', 'lig', 'lgm', 'mis']
LABELS = ['PI', 'MH', 'LIG', 'LGM', 'MIS3']

# colour-vision-deficiency safe: black / vermillion / bluish-green / blue
C_TOT = '#000000'
C_HEAT = '#D55E00'
C_ICE = '#009E73'
C_OTH = '#0072B2'

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'pdf.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#333333',
})


def load(exp, dom):
    d = xr.open_dataset(WMT_FILE.format(exp=exp))
    sig = d['sigma2_l_target'].values
    heat = -d[f'{dom}_total_heat_heat'].mean('time').values / 1e9

    def g(n):
        k = f'{dom}_surface_ocean_flux_advective_negative_rhs_{n}_salt'
        return -d[k].mean('time').values / 1e9 if k in d else np.zeros_like(heat)

    ice = g('sea_ice_melt')
    oth = g('evaporation') + g('snow') + g('rain_and_ice') + g('rivers')
    area = d.attrs['so60_area_m2'] if dom == 'so60' else d.attrs['siz_area_m2']
    return sig, heat, ice, oth, area


ROWS = [('so60', 'South of 60°S\n(domain used here)'),
        ('siz', 'Seasonal ice zone\n(Pellichero-comparable)')]

fig, axes = plt.subplots(2, 5, figsize=(FIG_W, P_H / SCALE))

panel = 0
letters = 'abcdefghij'
for r, (dom, rowlab) in enumerate(ROWS):
    for c, (exp, lab) in enumerate(zip(EXPS, LABELS)):
        ax = axes[r, c]
        sig, heat, ice, oth, area = load(exp, dom)
        tot = heat + ice + oth

        ax.axhline(0, color='#999999', lw=pt(0.5), zorder=1)
        ax.plot(sig, tot, color=C_TOT, lw=pt(1.4), label='Total', zorder=6,
                solid_capstyle='round')
        ax.plot(sig, heat, color=C_HEAT, lw=pt(1.15), label='Heat', zorder=5)
        ax.plot(sig, ice, color=C_ICE, lw=pt(1.05), label='Sea ice FW', zorder=4)
        ax.plot(sig, oth, color=C_OTH, lw=pt(1.05), label='Other FW', zorder=3)

        if exp in ('pi', 'mh', 'lig'):
            ax.set_xlim(35.5, 37.6)
        else:
            ax.set_xlim(36.2, 38.3)

        # thermal share at the peak of total transformation
        m = (sig >= 35.0) & (sig <= 39.0)
        idx = np.where(m)[0]
        i = idx[int(np.nanargmax(tot[idx]))]
        den = abs(heat[i]) + abs(ice[i]) + abs(oth[i])
        tf = abs(heat[i]) / den * 100 if den > 0 else np.nan
        # under the panel (below the tick labels, and the x label in row 2)
        # 'A =' on its own line: in one line the area is wider than a panel step
        ax.annotate(f'thermal {tf:.0f}%\npeak {tot[i]:.1f} Sv\n'
                    f'A =\n{area/1e13:.2f}$\\times10^{{13}}$ m$^2$',
                    xy=(0.5, 0), xycoords='axes fraction',
                    xytext=(0, -pt(15 if r == 0 else 28)), textcoords='offset points',
                    ha='center', va='top', fontsize=pt(8), color='black',
                    annotation_clip=False)

        # letters above the panel, ending at its left edge (over the y tick labels),
        # on the title baseline: inside, the 0.1-bin peaks reach both top corners
        ax.text(0.0, 1.0, f'({letters[panel]})', fontsize=pt(9), fontweight='bold',
                va='baseline', ha='right',
                transform=offset_copy(ax.transAxes, fig=fig, x=-pt(1), y=pt(4),
                                      units='points'))
        panel += 1

        if r == 0:
            ax.set_title(lab, fontsize=pt(9), fontweight='bold', pad=pt(4))
        if r == 1:
            ax.set_xlabel(r'$\sigma_2$  (kg m$^{-3}$)', fontsize=pt(8.5),
                          fontweight='bold', labelpad=pt(2))
        if c == 0:
            ax.set_ylabel('WMT  (Sv)', fontsize=pt(8.5), fontweight='bold',
                          labelpad=pt(2))
            ax.annotate(rowlab, xy=(0, 0.5), xytext=(-pt(41), 0),
                        xycoords='axes fraction', textcoords='offset points',
                        rotation=90, ha='center', va='center',
                        fontsize=pt(8.5), fontweight='bold', annotation_clip=False)
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.tick_params(labelsize=pt(7.5), labelcolor='black', width=pt(0.6),
                       length=pt(3), pad=pt(2))
        ax.tick_params(which='minor', width=pt(0.5), length=pt(1.8))
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_linewidth(pt(0.6))

h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.998),
           ncol=4, frameon=False, fontsize=pt(8), handlelength=2.2,
           columnspacing=1.6)

# margins in printed inches: left 0.76 (row label, y label, ticks), right 0.10,
# top 0.46 (legend, titles), bottom 0.95 (ticks, x label, numbers); panels 1.10 in
# tall and 0.594 in wide, 0.98 in between rows, 0.33 in between columns
fig.subplots_adjust(left=0.76 / P_W, right=1 - 0.10 / P_W, top=1 - 0.46 / P_H,
                    bottom=0.95 / P_H, hspace=0.98 / 1.10, wspace=0.33 / 0.594)
fig.savefig('figures/figR1_wmt_domain_sensitivity.pdf', dpi=400)
fig.savefig('figures/figR1_wmt_domain_sensitivity.png', dpi=400)
print('saved figures/figR1_wmt_domain_sensitivity.{pdf,png}')

# ------------------------------------------------------------------ table
print()
print('%-5s %-6s %9s %8s %8s %8s %8s %7s' % ('EXP', 'DOM', 'area1e13', 'peak', 'sig2',
                                             'heat', 'seaice', 'therm%'))
for exp, lab in zip(EXPS, LABELS):
    for dom, _ in ROWS:
        sig, heat, ice, oth, area = load(exp, dom)
        tot = heat + ice + oth
        m = (sig >= 35.0) & (sig <= 39.0)
        idx = np.where(m)[0]
        i = idx[int(np.nanargmax(tot[idx]))]
        den = abs(heat[i]) + abs(ice[i]) + abs(oth[i])
        print('%-5s %-6s %9.2f %8.2f %8.2f %8.2f %8.2f %6.0f%%'
              % (lab, dom, area/1e13, tot[i], sig[i], heat[i], ice[i],
                 abs(heat[i])/den*100))
