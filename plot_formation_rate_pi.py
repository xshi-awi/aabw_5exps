#!/usr/bin/env python
"""
Water mass formation rate for PI, drawn the way Pellichero et al. (2018) draw
their Fig. 3b, so the two can be read side by side.

Formation is the convergence of the transformation, F = -dT/dsigma, integrated
over each density bin. Positive means water accumulates in the class and must
subduct; negative means it is removed and must be supplied by upwelling. That is
a different quantity from the transformation itself, which is what our Fig. 8
shows, and it is the quantity their panel b plots.

Domain is the seasonal sea ice zone, the Pellichero-comparable domain from
calc_wmt_siz_pellichero.py, and the average is annual, matching their
annual-mean framework. Their published integrals are drawn as horizontal
brackets, on the same convention as their figure.

Two differences from their panel are unavoidable and are stated on the figure
rather than hidden. Our density bins are 0.2 kg/m3 wide against their 0.1, since
that is the resolution of the transformation we computed. And our axis is
sigma2, mapped onto their neutral density axis using the model's own winter
surface water, so the tick labels are approximate above 27.9 where that density
barely outcrops.

Outputs figures/figR12_formation_rate_pi.{pdf,png}.
"""
import numpy as np
import xarray as xr
import seawater as sw
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

MESH = '/home/a/a270064/bb1029/inputs/mesh_core2/fesom.mesh.diag.nc'

C_TOT = '#7F7F7F'
C_HEAT = '#D62728'
C_FW = '#1F4EF5'

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#333333'})


# ------------------------------------------------------------ sigma2 -> gamma_n
def sigma2_to_gamma():
    """Fit gamma_n(sigma2) from PI winter surface water inside the ice zone."""
    T = xr.open_dataset('pi/temp.clim.nc', decode_times=False)['temp'][:, :, 0].values
    S = xr.open_dataset('pi/salt.clim.nc', decode_times=False)['salt'][:, :, 0].values
    lat = np.degrees(xr.open_dataset(MESH)['nodes'].isel(n2=1).values)
    aice = xr.open_dataset('pi/a_ice_clim.nc', decode_times=False)['a_ice'].values
    siz = (aice.max(axis=0) > 0.15) & (lat < 0)
    t = T[[5, 6, 7]].mean(axis=0)[siz]
    s = S[[5, 6, 7]].mean(axis=0)[siz]
    ok = np.isfinite(t) & np.isfinite(s) & (s > 20)
    t, s = t[ok], s[ok]
    sig0 = sw.dens(s, t, 0) - 1000
    sig2 = sw.dens(s, t, 2000) - 1000
    fit = np.polyfit(sig2, sig0, 1)
    print('gamma_n ~ %.4f * sigma2 + %.4f  (from %d points)' % (fit[0], fit[1], len(t)))
    return lambda x: np.polyval(fit, x)


G = sigma2_to_gamma()


# ------------------------------------------------------------------ transformation
def load():
    d = xr.open_dataset('wmt_siz_pi.nc')
    sig = d['sigma2_l_target'].values
    heat = -d['siz_total_heat_heat'].mean('time').values / 1e9

    def g(n):
        k = f'siz_surface_ocean_flux_advective_negative_rhs_{n}_salt'
        return -d[k].mean('time').values / 1e9 if k in d else np.zeros_like(heat)

    fw = (g('sea_ice_melt') + g('evaporation') + g('snow')
          + g('rain_and_ice') + g('rivers'))
    return sig, heat, fw


SIG, HEAT, FW = load()
TOT = HEAT + FW

# keep the populated range only
live = np.abs(TOT) + np.abs(HEAT) + np.abs(FW) > 1e-6
i0, i1 = np.where(live)[0][[0, -1]]
sl = slice(max(i0 - 1, 0), min(i1 + 2, len(SIG)))
sig, heat, fw, tot = SIG[sl], HEAT[sl], FW[sl], TOT[sl]

# formation = -d(transformation)/d(density), per bin
dsig = np.diff(sig).mean()


def formation(x):
    return -np.gradient(x, dsig) * dsig


# Sign check against their own text: with F = -dT/dgamma, classes lighter than
# 27.3 come out positive (subduction) and 27.3-27.9 negative (upwelling), which
# is what they describe. The opposite sign would reverse both.
f_tot, f_heat, f_fw = formation(tot), formation(heat), formation(fw)
gam = G(sig)

fig, ax = plt.subplots(figsize=(13.5, 6.4))

w = dsig * 0.26
ax.bar(gam - w, f_tot, w * 0.95, color=C_TOT, label='Total', zorder=4)
ax.bar(gam, f_heat, w * 0.95, color=C_HEAT, label='Heat flux', zorder=4)
ax.bar(gam + w, f_fw, w * 0.95, color=C_FW, label='FW flux', zorder=4)
ax.axhline(0, color='#333333', lw=1.0, zorder=5)

# their 27.6 divide between lightening and densification
g276 = 27.6
ax.axvline(g276, color='#D62728', ls='--', lw=1.6, zorder=3)
ax.text(g276 + 0.012, ax.get_ylim()[1], ' 27.6 $\\gamma$', fontsize=11,
        color='#D62728', va='top')

lo, hi = gam.min() - 0.05, gam.max() + 0.05
ax.set_xlim(lo, hi)

ymax = max(abs(f_tot).max(), abs(f_heat).max(), abs(f_fw).max()) * 1.45
ax.set_ylim(-ymax, ymax)

# sense-of-sign arrows, as on their panel
ax.annotate('', xy=(lo + 0.06, ymax * 0.62), xytext=(lo + 0.06, ymax * 0.25),
            arrowprops=dict(arrowstyle='->', lw=1.6, color='#333333'))
ax.text(lo + 0.09, ymax * 0.44, 'Subduction', fontsize=12, va='center')
ax.annotate('', xy=(lo + 0.06, -ymax * 0.62), xytext=(lo + 0.06, -ymax * 0.25),
            arrowprops=dict(arrowstyle='->', lw=1.6, color='#333333'))
ax.text(lo + 0.09, -ymax * 0.44, 'Upwelling', fontsize=12, va='center')

# their published integrals, for comparison with ours over the same ranges
def integral(g0, g1):
    m = (gam >= g0) & (gam < g1)
    return f_tot[m].sum()


BRACK = [(gam.min(), 27.3, 'Subduction: 22 $\\pm$ 4 Sv', '#3B5BC0'),
         (27.3, 27.9, 'Upwelling: 27 $\\pm$ 7 Sv', '#C04040'),
         (27.9, gam.max(), 'Subduction: 5 $\\pm$ 5 Sv', '#3B5BC0')]
yb = ymax * 0.90
for g0, g1, lab, c in BRACK:
    a, b = max(g0, lo), min(g1, hi)
    if b - a < 0.02:
        continue
    ax.annotate('', xy=(a, yb), xytext=(b, yb),
                arrowprops=dict(arrowstyle='<->', color=c, lw=1.4))
    ours = integral(g0, g1)
    ax.text((a + b) / 2, yb + ymax * 0.045,
            '%s\n(observed)\nours, net: %+.1f Sv' % (lab, ours),
            ha='center', va='bottom', fontsize=9.8, color=c, linespacing=1.35)
    ax.axvspan(a, b, color=c, alpha=0.055, zorder=0)

ax.set_xlabel('Neutral density, $\\gamma$ (kg m$^{-3}$), converted from $\\sigma_2$',
              fontsize=13.5)
ax.set_ylabel('Water mass formation rate (Sv)', fontsize=13.5)
ax.set_title('PI formation rate in the seasonal sea ice zone, annual mean',
             fontsize=15, fontweight='bold', pad=42)
ax.legend(frameon=False, fontsize=12, loc='lower left')
ax.grid(axis='y', alpha=0.22, lw=0.6)
ax.tick_params(labelsize=12)

ax.text(0.995, 0.015,
        'bins 0.2 kg m$^{-3}$ (theirs 0.1); $\\gamma$ axis converted from $\\sigma_2$',
        transform=ax.transAxes, ha='right', fontsize=9.5,
        color='#777777', style='italic')

fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.135)
fig.savefig('figures/figR12_formation_rate_pi.pdf', dpi=400)
fig.savefig('figures/figR12_formation_rate_pi.png', dpi=200)
print('saved figures/figR12_formation_rate_pi.{pdf,png}')

print()
print('%8s %8s %10s %10s %10s' % ('sigma2', 'gamma', 'F_total', 'F_heat', 'F_fw'))
for k in range(len(sig)):
    if abs(f_tot[k]) > 0.01 or abs(f_heat[k]) > 0.01:
        print('%8.1f %8.2f %10.2f %10.2f %10.2f'
              % (sig[k], gam[k], f_tot[k], f_heat[k], f_fw[k]))
print()
for g0, g1, lab, _ in BRACK:
    print('%-32s ours %+7.2f Sv' % (lab.replace('$\\pm$', '+/-'), integral(g0, g1)))
