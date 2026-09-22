#!/usr/bin/env python
"""
PI transformation in the seasonal sea ice zone on a neutral density axis, drawn
in the same form as Fig. 3a of Pellichero et al. (2018), with their published
values overlaid.

WHAT IS PLOTTED AS A POINT, AND WHAT IS ONLY ANNOTATED

Two of their numbers are transformation rates, the same quantity as our curve,
so they are drawn as points with error bars:

    -22 +/- 4 Sv  peak toward lighter water, at 27.3 gamma
      5 +/- 5 Sv  peak toward denser water,  at 27.9 gamma

Their 9 +/- 4 Sv at 27.1 gamma is a SUBDUCTION rate, the divergence of the
transformation rather than the transformation itself. It has the same units but
is not the same quantity, so it appears as an annotation only. Drawing it as a
point on this axis would compare unlike things. The same applies to their
27 +/- 7 Sv net upwelling and the 4 Sv Weddell subduction.

All observed values come from the text of their paper, not from reading their
figures. Their data availability statement reads only "All relevant data are
available from the authors", so nothing is downloadable and every number quoted
here can be cited to a sentence.

ON THE SHARED VERTICAL AXIS

Their lighter-water peak is -22 Sv against our 6.8 Sv maximum, so a shared axis
compresses our curve. It is kept shared regardless: the magnitude difference is
one of the things we report, and a second axis would disguise it.

Outputs figures/figR15_pi_gamma_vs_obs.{pdf,png}.
"""
import numpy as np
import xarray as xr
import seawater as sw
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

MESH = '/home/a/a270064/bb1029/inputs/mesh_core2/fesom.mesh.diag.nc'

C_TOT = '#000000'
C_HEAT = '#D55E00'
C_ICE = '#009E73'
C_OTH = '#0072B2'
C_OBS = '#7B3294'

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#333333'})


def gamma_fit():
    """sigma2 -> gamma_n from PI winter surface water inside the ice zone."""
    T = xr.open_dataset('pi/temp.clim.nc', decode_times=False)['temp'][:, :, 0].values
    S = xr.open_dataset('pi/salt.clim.nc', decode_times=False)['salt'][:, :, 0].values
    lat = np.degrees(xr.open_dataset(MESH)['nodes'].isel(n2=1).values)
    a = xr.open_dataset('pi/a_ice_clim.nc', decode_times=False)['a_ice'].values
    siz = (a.max(axis=0) > 0.15) & (lat < 0)
    t = T[[5, 6, 7]].mean(axis=0)[siz]
    s = S[[5, 6, 7]].mean(axis=0)[siz]
    ok = np.isfinite(t) & np.isfinite(s) & (s > 20)
    t, s = t[ok], s[ok]
    sig2 = sw.dens(s, t, 2000) - 1000
    sig0 = sw.dens(s, t, 0) - 1000
    fit = np.polyfit(sig2, sig0, 1)
    # the range the fit is actually constrained over
    return fit, float(np.percentile(sig2, 0.5)), float(np.percentile(sig2, 99.5))


FIT, S_LO, S_HI = gamma_fit()
G = lambda x: np.polyval(FIT, x)
print('gamma = %.4f * sigma2 + %.4f, constrained over sigma2 %.1f-%.1f'
      % (FIT[0], FIT[1], S_LO, S_HI))


def load():
    d = xr.open_dataset('wmt_siz_pi.nc', decode_times=False)
    sig = d['sigma2_l_target'].values
    V = lambda n: -d[n].values.mean(axis=0) / 1e9
    h = V('siz_total_heat_heat')

    def g(n):
        k = f'siz_surface_ocean_flux_advective_negative_rhs_{n}_salt'
        return V(k) if k in d else np.zeros_like(h)

    i = g('sea_ice_melt')
    o = sum(g(k) for k in ['evaporation', 'snow', 'rain_and_ice', 'rivers'])
    return sig, h, i, o


SIG, HEAT, ICE, OTH = load()
TOT = HEAT + ICE + OTH

# keep only classes the model actually populates, and only where the gamma fit
# is constrained by data rather than extrapolating past the end of the ocean
live = (np.abs(TOT) > 1e-6) & (SIG >= S_LO - 0.3) & (SIG <= S_HI + 0.3)
i0, i1 = np.where(live)[0][[0, -1]]
sl = slice(i0, i1 + 1)
sig, heat, ice, oth, tot = SIG[sl], HEAT[sl], ICE[sl], OTH[sl], TOT[sl]
gam = G(sig)

fig, ax = plt.subplots(figsize=(13.8, 7.4))

ax.axhline(0, color='#888888', lw=0.9, zorder=1)
ax.plot(gam, tot, color=C_TOT, lw=3.0, label='Total', zorder=6,
        solid_capstyle='round')
ax.plot(gam, heat, color=C_HEAT, lw=2.4, label='Heat flux', zorder=5)
ax.plot(gam, ice, color=C_ICE, lw=2.2, label='Sea ice FW', zorder=4)
ax.plot(gam, oth, color=C_OTH, lw=2.2, label='Other FW', zorder=3)

# their 27.6 divide between lightening and densification
ax.axvline(27.6, color=C_OBS, ls='--', lw=1.6, zorder=2)

# the two published TRANSFORMATION rates
ax.errorbar([27.3], [-22], yerr=[4], fmt='D', color=C_OBS, ms=12, capsize=7,
            lw=2.4, zorder=8, label='Observed transformation\n(Pellichero et al. 2018)')
ax.errorbar([27.9], [5], yerr=[5], fmt='D', color=C_OBS, ms=12, capsize=7,
            lw=2.4, zorder=8)

lo, hi = gam.min() - 0.05, min(gam.max(), 28.9) + 0.05
ax.set_xlim(lo, hi)
ax.set_ylim(-30, 22)

ax.annotate('observed peak toward lighter water\n$-$22 $\\pm$ 4 Sv at 27.3 $\\gamma$',
            xy=(27.3, -22), xytext=(0.055, 0.115), textcoords='axes fraction',
            fontsize=10.5, color=C_OBS, ha='left', linespacing=1.4,
            arrowprops=dict(arrowstyle='->', color=C_OBS, lw=1.3))
ax.annotate('observed peak toward\ndenser water\n5 $\\pm$ 5 Sv at 27.9 $\\gamma$',
            xy=(27.88, 11.0), xytext=(27.86, 17.5), textcoords='data',
            fontsize=10.5, color=C_OBS, ha='left', va='center', linespacing=1.4,
            arrowprops=dict(arrowstyle='->', color=C_OBS, lw=1.3,
                            connectionstyle='arc3,rad=0.2'))
ax.text(27.585, 20.4, '27.6 $\\gamma$: observed divide between\n'
        'lightening and densification',
        fontsize=10, color=C_OBS, va='top', ha='right', linespacing=1.4)

# the subduction number is a different quantity, so it is stated, not plotted
ax.text(0.015, 0.975,
        'Also reported, but a different quantity and therefore not plotted:\n'
        'peak subduction 9 $\\pm$ 4 Sv at 27.1 $\\gamma$, "dominated by '
        'freshwater-driven\nwater-mass transformation"; net upwelling 27 $\\pm$ 7 Sv '
        'over 27.3$-$27.9 $\\gamma$.\nSubduction is the divergence of the '
        'transformation, not the transformation.',
        transform=ax.transAxes, va='top', ha='left', fontsize=9.5,
        color='#444444', linespacing=1.55,
        bbox=dict(boxstyle='round,pad=0.55', fc='#f6f4f8', ec='#d4cbdd'))

k = int(np.nanargmax(tot))
ax.annotate('our maximum\n%.1f Sv at %.2f $\\gamma$' % (tot[k], gam[k]),
            xy=(gam[k] + 0.012, tot[k] + 0.4), xytext=(27.36, 12.4),
            textcoords='data', fontsize=10.5, color='#222222', ha='center',
            va='bottom', linespacing=1.4,
            arrowprops=dict(arrowstyle='->', color='#222222', lw=1.3,
                            connectionstyle='arc3,rad=-0.2'))

ax.set_xlabel('Neutral density, $\\gamma$ (kg m$^{-3}$), converted from $\\sigma_2$',
              fontsize=13.5)
ax.set_ylabel('Annual-mean transformation (Sv)', fontsize=13.5)
ax.set_title('PI transformation in the seasonal sea ice zone, against the '
             'observed rates', fontsize=15, fontweight='bold')
ax.legend(frameon=False, fontsize=11.5, loc='lower right')
ax.grid(alpha=0.2, lw=0.6)
ax.tick_params(labelsize=12)

fig.subplots_adjust(left=0.072, right=0.985, top=0.925, bottom=0.115)
fig.savefig('figures/figR15_pi_gamma_vs_obs.pdf', dpi=400)
fig.savefig('figures/figR15_pi_gamma_vs_obs.png', dpi=200)
print('saved figures/figR15_pi_gamma_vs_obs.{pdf,png}')

print()
print('%8s %8s %9s %9s %9s %9s' % ('sigma2', 'gamma', 'total', 'heat', 'seaice', 'othFW'))
for k in range(len(sig)):
    if abs(tot[k]) > 0.02:
        print('%8.1f %8.2f %9.2f %9.2f %9.2f %9.2f'
              % (sig[k], gam[k], tot[k], heat[k], ice[k], oth[k]))
k = int(np.nanargmax(tot))
print('\nour maximum %.2f Sv at gamma %.2f (sigma2 %.1f)' % (tot[k], gam[k], sig[k]))
print('their densifying peak 5 +/- 5 Sv at gamma 27.9')
m = (gam >= 27.6) & (gam < 27.9)
print('our total over gamma 27.6-27.9: %.2f Sv' % np.nansum(tot[m]))
m = gam >= 27.9
print('our total over gamma >= 27.9  : %.2f Sv' % np.nansum(tot[m]))
