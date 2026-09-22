#!/usr/bin/env python
"""
Variants of the PI density-class figure, for choosing which version to use.

Produces four panels-of-figures:

  figR11a_allSO_annual    whole Southern Ocean, annual mean
  figR11b_allSO_jja       whole Southern Ocean, winter
  figR11c_sectors_annual  Ross / Weddell / Adelie, annual mean
  figR11d_sectors_jja     Ross / Weddell / Adelie, winter

The three sectors partition the full longitude circle south of 60 S, so their
sum reproduces the Southern Ocean domain and the comparison between the two
rows is exact rather than indicative.

Axis is our own sigma2 over the full populated range. Nothing is truncated: an
earlier version started at the Pellichero 27.1 boundary and silently dropped the
light classes, which is where the largest feature sits.

Outputs figures/figR11{a,b,c,d}_*.{pdf,png}.
"""
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

C_HEAT = '#D55E00'
C_ICE = '#009E73'
C_OTH = '#0072B2'
C_TOT = '#000000'

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#333333'})

CLASSES = [(None, 36.3, 'Light\n$\\sigma_2<$36.3'),
           (36.3, 36.7, 'Intermediate\n36.3$-$36.7'),
           (36.7, 37.0, 'Peak\n36.7$-$37.0'),
           (37.0, None, 'Densest\n$\\sigma_2\\geq$37.0')]

REGIONS = {'southern_ocean': 'Southern Ocean (south of 60$^\\circ$S)',
           'siz': 'Seasonal sea ice zone (Pellichero-comparable)',
           'ross_sea': 'Ross sector (180$^\\circ$W$-$60$^\\circ$W)',
           'weddell_sea': 'Weddell sector (60$^\\circ$W$-$79$^\\circ$E)',
           'adelie': 'Adélie sector (79$^\\circ$E$-$180$^\\circ$E)'}


def load(region, months=None):
    """region is a sector file, or 'siz'/'so60'/'open' inside wmt_siz_pi.nc."""
    if region in ('siz', 'so60', 'open'):
        d = xr.open_dataset('wmt_siz_pi.nc', decode_times=False)
        sig = d['sigma2_l_target'].values
        sel = slice(None) if months is None else [m - 1 for m in months]
        V = lambda n: -d[n].values[sel].mean(axis=0) / 1e9
        h = V(f'{region}_total_heat_heat')
        g = lambda n: (V(f'{region}_surface_ocean_flux_advective_negative_rhs_{n}_salt')
                       if f'{region}_surface_ocean_flux_advective_negative_rhs_{n}_salt'
                       in d else np.zeros_like(h))
        i = g('sea_ice_melt')
        o = sum(g(k) for k in ['evaporation', 'snow', 'rain_and_ice', 'rivers'])
        return sig, h, i, o

    d = xr.open_dataset(
        f'pi/wmt_echam_heat_fesom_freshwater_{region}_pi.nc', decode_times=False)
    sig = d['sigma2_l_target'].values
    sel = slice(None) if months is None else [m - 1 for m in months]
    V = lambda n: -d[n].values[sel].mean(axis=0) / 1e9
    h = V('total_heat_surface_exchange_flux_nonadvective_heat')
    i = V('surface_ocean_flux_advective_negative_rhs_sea_ice_melt_salt')
    o = sum(V(f'surface_ocean_flux_advective_negative_rhs_{k}_salt')
            for k in ['evaporation', 'snow', 'rain_and_ice', 'rivers'])
    return sig, h, i, o


def by_class(region, months=None):
    sig, h, i, o = load(region, months)
    rows = []
    for lo, hi, _ in CLASSES:
        m = np.ones_like(sig, bool)
        if lo is not None:
            m &= sig >= lo
        if hi is not None:
            m &= sig < hi
        H, I, O = np.nansum(h[m]), np.nansum(i[m]), np.nansum(o[m])
        den = abs(H) + abs(I) + abs(O)
        rows.append(dict(heat=H, ice=I, oth=O, total=H + I + O,
                         thermal=abs(H) / den * 100 if den else np.nan))
    return rows


x = np.arange(len(CLASSES))
xlab = [c[2] for c in CLASSES]


def panel_components(ax, rows, title, unit):
    w = 0.26
    ax.bar(x - w, [r['heat'] for r in rows], w, color=C_HEAT, label='Heat')
    ax.bar(x, [r['ice'] for r in rows], w, color=C_ICE, label='Sea ice FW')
    ax.bar(x + w, [r['oth'] for r in rows], w, color=C_OTH, label='Other FW')
    ax.plot(x, [r['total'] for r in rows], 'o-', color=C_TOT, lw=2.3, ms=8,
            label='Total', zorder=6)
    ax.axhline(0, color='#444444', lw=0.9)
    for k, r in enumerate(rows):
        ax.annotate('%+.1f' % r['total'], xy=(k, r['total']),
                    xytext=(0, 13 if r['total'] >= 0 else -21),
                    textcoords='offset points', ha='center',
                    fontsize=10, fontweight='bold', zorder=7)
    ax.set_xticks(x)
    ax.set_xticklabels(xlab, fontsize=10.5)
    ax.set_ylabel('%s transformation (Sv)' % unit, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=10.5)


def panel_share(ax, rows, title):
    ax.axhspan(0, 33, color=C_ICE, alpha=0.11, zorder=0)
    ax.plot(x, [r['thermal'] for r in rows], '-o', color=C_TOT, lw=2.6, ms=9)
    ax.axhline(50, color='#999999', lw=1.0, ls=':')
    for k, r in enumerate(rows):
        if np.isfinite(r['thermal']):
            dy = 14 if r['thermal'] < 20 else -21
            ax.annotate('%.0f%%' % r['thermal'], xy=(k, r['thermal']),
                        xytext=(0, dy), textcoords='offset points', ha='center',
                        fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(xlab, fontsize=10.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Thermal share (%)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=10.5)


# ------------------------------------------------------------------ whole SO
for tag, months, unit, sname in [('a', None, 'Annual-mean', 'annual'),
                                 ('b', [6, 7, 8], 'Winter (JJA)', 'jja')]:
    rows = by_class('southern_ocean', months)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0))
    panel_components(axes[0], rows, '(a)  Components by class', unit)
    axes[0].legend(frameon=False, fontsize=11, loc='lower right')
    panel_share(axes[1], rows, '(b)  Thermal share')
    axes[1].text(0.03, 26.5, 'haline dominated', fontsize=11.5, color='#00654a',
                 transform=axes[1].get_yaxis_transform())
    for a in axes:
        a.set_xlabel('Density class, $\\sigma_2$ (kg m$^{-3}$)', fontsize=12)
    fig.suptitle('PI, %s, %s' % (REGIONS['southern_ocean'], unit.lower()),
                 fontsize=15, fontweight='bold')
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.16, wspace=0.22)
    out = 'figures/figR11%s_allSO_%s' % (tag, sname)
    fig.savefig(out + '.pdf', dpi=400)
    fig.savefig(out + '.png', dpi=200)
    plt.close(fig)
    print('saved ' + out + '.{pdf,png}')

# ------------------------------------------------------------ sea ice zone
# The Pellichero-comparable domain. No sector breakdown is possible here: the
# wmt_siz files carry only the three domains, not a longitude split.
for tag, months, unit, sname in [('e', None, 'Annual-mean', 'annual'),
                                 ('f', [6, 7, 8], 'Winter (JJA)', 'jja')]:
    rows = by_class('siz', months)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0))
    panel_components(axes[0], rows, '(a)  Components by class', unit)
    axes[0].legend(frameon=False, fontsize=11, loc='lower right')
    panel_share(axes[1], rows, '(b)  Thermal share')
    axes[1].text(0.03, 26.5, 'haline dominated', fontsize=11.5, color='#00654a',
                 transform=axes[1].get_yaxis_transform())
    for a in axes:
        a.set_xlabel('Density class, $\\sigma_2$ (kg m$^{-3}$)', fontsize=12)
    fig.suptitle('PI, %s, %s' % (REGIONS['siz'], unit.lower()),
                 fontsize=15, fontweight='bold')
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.16, wspace=0.22)
    out = 'figures/figR11%s_siz_%s' % (tag, sname)
    fig.savefig(out + '.pdf', dpi=400)
    fig.savefig(out + '.png', dpi=200)
    plt.close(fig)
    print('saved ' + out + '.{pdf,png}')

# ------------------------------------------------------------------ sectors
SECT = ['ross_sea', 'weddell_sea', 'adelie']
for tag, months, unit, sname in [('c', None, 'Annual-mean', 'annual'),
                                 ('d', [6, 7, 8], 'Winter (JJA)', 'jja')]:
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.2))
    for j, reg in enumerate(SECT):
        rows = by_class(reg, months)
        panel_components(axes[0, j], rows, '(%s)  %s' % ('abc'[j], REGIONS[reg]), unit)
        panel_share(axes[1, j], rows, '(%s)  Thermal share' % 'def'[j])
        if j == 0:
            axes[0, j].legend(frameon=False, fontsize=10.5, loc='lower right')
            axes[1, j].text(0.03, 26.5, 'haline dominated', fontsize=11,
                            color='#00654a',
                            transform=axes[1, j].get_yaxis_transform())
        axes[1, j].set_xlabel('Density class, $\\sigma_2$ (kg m$^{-3}$)', fontsize=12)
    fig.suptitle('PI by sector, %s' % unit.lower(), fontsize=16, fontweight='bold')
    fig.subplots_adjust(left=0.055, right=0.99, top=0.91, bottom=0.085,
                        wspace=0.26, hspace=0.38)
    out = 'figures/figR11%s_sectors_%s' % (tag, sname)
    fig.savefig(out + '.pdf', dpi=400)
    fig.savefig(out + '.png', dpi=200)
    plt.close(fig)
    print('saved ' + out + '.{pdf,png}')

# ------------------------------------------------------------------ table
print()
hdr = '%-16s %-8s %-14s %9s %9s %9s %9s %7s'
print(hdr % ('region', 'season', 'class', 'total', 'heat', 'seaice', 'othFW', 'therm%'))
for reg in ['southern_ocean', 'siz'] + SECT:
    for sname, months in [('annual', None), ('JJA', [6, 7, 8])]:
        for (lo, hi, _), r in zip(CLASSES, by_class(reg, months)):
            lab = '%s-%s' % (lo if lo else 'min', hi if hi else 'max')
            print(hdr % (reg, sname, lab, '%.2f' % r['total'], '%.2f' % r['heat'],
                         '%.2f' % r['ice'], '%.2f' % r['oth'],
                         '%.0f%%' % r['thermal'] if np.isfinite(r['thermal']) else 'n/a'))
