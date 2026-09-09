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
"""
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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
    d = xr.open_dataset(f'wmt_siz_{exp}.nc')
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

fig, axes = plt.subplots(2, 5, figsize=(20, 8.2))

panel = 0
letters = 'abcdefghij'
for r, (dom, rowlab) in enumerate(ROWS):
    for c, (exp, lab) in enumerate(zip(EXPS, LABELS)):
        ax = axes[r, c]
        sig, heat, ice, oth, area = load(exp, dom)
        tot = heat + ice + oth

        ax.axhline(0, color='#999999', lw=0.8, zorder=1)
        ax.plot(sig, tot, color=C_TOT, lw=2.8, label='Total', zorder=6,
                solid_capstyle='round')
        ax.plot(sig, heat, color=C_HEAT, lw=2.3, label='Heat', zorder=5)
        ax.plot(sig, ice, color=C_ICE, lw=2.1, label='Sea ice FW', zorder=4)
        ax.plot(sig, oth, color=C_OTH, lw=2.1, label='Other FW', zorder=3)

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
        ax.annotate(f'thermal {tf:.0f}%\npeak {tot[i]:.1f} Sv',
                    xy=(0.97, 0.94), xycoords='axes fraction',
                    ha='right', va='top', fontsize=12, color='#333333')

        ax.text(0.02, 0.96, f'({letters[panel]})', transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')
        panel += 1

        if r == 0:
            ax.set_title(lab, fontsize=19, fontweight='bold', pad=10)
        if r == 1:
            ax.set_xlabel(r'$\sigma_2$  (kg m$^{-3}$)', fontsize=15)
        if c == 0:
            ax.set_ylabel('WMT  (Sv)', fontsize=15)
            ax.annotate(rowlab, xy=(0, 0.5), xytext=(-78, 0),
                        xycoords='axes fraction', textcoords='offset points',
                        rotation=90, ha='center', va='center',
                        fontsize=15, fontweight='bold', annotation_clip=False)
        ax.tick_params(labelsize=12)
        ax.annotate(f'A = {area/1e13:.2f}$\\times10^{{13}}$ m$^2$',
                    xy=(0.5, -0.30), xycoords='axes fraction',
                    ha='center', fontsize=11, color='#555555',
                    annotation_clip=False)

h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.995),
           ncol=4, frameon=False, fontsize=15)

fig.subplots_adjust(left=0.088, right=0.99, top=0.86, bottom=0.11,
                    hspace=0.42, wspace=0.26)
fig.savefig('figures/figR1_wmt_domain_sensitivity.pdf', dpi=400)
fig.savefig('figures/figR1_wmt_domain_sensitivity.png', dpi=200)
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
