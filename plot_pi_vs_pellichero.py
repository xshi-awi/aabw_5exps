#!/usr/bin/env python
"""
Our PI transformation next to the observational estimate of Pellichero et al. (2018).

Reviewer 1 asked for a direct comparison against an observational WMT, and the
submitted version showed only model curves. We cannot recompute their estimate:
they state that reanalysis fluxes are unusable in the sea-ice sector and solve a
mixed-layer heat and salt budget instead, using a seal-tag hydrographic database
we do not have. So we reproduce their published Fig. 3a beside ours.

Matching their analysis as closely as our output allows:
  domain  the seasonal sea ice zone (September concentration > 15%), which is
          their sector definition, not the full region south of 60S
  season  annual mean, as in their Fig. 3a, not the JJA mean used in the paper
  sign    transformation, positive = densification

Panel (a) is drawn to match their Fig. 3a as closely as our output allows:
neutral density on the x axis, the same line colours, the same lightening and
densification arrows, the buoyancy gain and loss annotations, and a red dashed
line at the density where the total transformation changes sign. Shading marks
where heat or freshwater dominates, which their panel does not have.

Neutral density comes from a linear fit of sigma_0 on sigma_2 over PI winter
surface water inside the ice zone, gamma = 1.0225 sigma_2 - 10.2445, the same
conversion used elsewhere in this project. Our surface water is lighter than
observed, so equal gamma does not mean the same water mass; the offset is itself
one of our findings and is stated in the caption.
"""
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})

SRC = 'wmt_siz_pi.nc'
OBS = 'nc_paper/SUBMISSION/p3.png'
PRE = 'siz_'          # the Pellichero-comparable domain

HEAT = 'total_heat_heat'
FW = ['surface_ocean_flux_advective_negative_rhs_evaporation_salt',
      'surface_ocean_flux_advective_negative_rhs_snow_salt',
      'surface_ocean_flux_advective_negative_rhs_sea_ice_melt_salt',
      'surface_ocean_flux_advective_negative_rhs_rain_and_ice_salt',
      'surface_ocean_flux_advective_negative_rhs_rivers_salt',
      'surface_exchange_flux_nonadvective_salt']

d = xr.open_dataset(SRC)
sig = d['sigma2_l_target'].values

# Annual mean, to match their Fig. 3a. The sign flip and the 1e9 scaling are
# the convention used throughout this project (see plot_siz_gamma_5exps.py):
# the stored field is the negative of the transformation, in kg/s.
V = lambda n: -d[n].values.mean(axis=0) / 1e9
heat = V(PRE + HEAT)
fw = sum(V(PRE + v) for v in FW)
tot = heat + fw

# split the freshwater flux for the shading, as Fig. R2.2 does: the sea ice
# thermodynamic term behaves differently from precipitation, evaporation and
# runoff, and merging them hides which one is driving the transformation
ICE = ['surface_ocean_flux_advective_negative_rhs_sea_ice_melt_salt',
       'surface_exchange_flux_nonadvective_salt']
OTH = ['surface_ocean_flux_advective_negative_rhs_evaporation_salt',
       'surface_ocean_flux_advective_negative_rhs_snow_salt',
       'surface_ocean_flux_advective_negative_rhs_rain_and_ice_salt',
       'surface_ocean_flux_advective_negative_rhs_rivers_salt']
ice = sum(V(PRE + v) for v in ICE)
oth = sum(V(PRE + v) for v in OTH)
total = tot
d.close()

# their figure spans 26.3-28.9 neutral density; ours is shown over the range
# the simulated fluxes actually populate
m = (sig >= 35.8) & (sig <= 37.6)

fig = plt.figure(figsize=(15.5, 5.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.18], wspace=0.18,
                      left=0.055, right=0.985, top=0.86, bottom=0.26)

# ---------------------------------------------------------------- our result
# gamma_n from a linear fit on PI winter surface water in the ice zone
GAMMA = lambda x: 1.0225 * x - 10.2445
gam = GAMMA(sig)
mg = (gam >= 26.3) & (gam <= 28.9)

# Their red dashed line marks the density separating water that becomes lighter
# from water that becomes denser. Ours is the same thing: the density at which
# the total transformation changes sign, found by interpolation rather than set
# by hand.
_g, _t = gam[mg], tot[mg]
_i = np.where(np.diff(np.sign(_t)) > 0)[0]
ZERO_X = (_g[_i[0]] + (_g[_i[0] + 1] - _g[_i[0]])
          * (0 - _t[_i[0]]) / (_t[_i[0] + 1] - _t[_i[0]])) if len(_i) else 27.6
print('total changes sign at gamma %.3f' % ZERO_X)

ax = fig.add_subplot(gs[0, 0])

# Shading by which term actually drives the transformation. Two rules matter.
# The freshwater flux is split into the sea ice contribution and the rest, as
# in Fig. R2.2, because they behave differently. And only a term whose sign
# AGREES with the total can be called its driver: at gamma 27.1 the heat term
# is +7.5 Sv while the total is negative, so heat is opposing the total there,
# not carrying it. Comparing magnitudes alone called that bin thermal, which is
# wrong.
# A term can only be the driver if it pushes the same way as the total. At
# gamma 27.1 the heat term is +7.5 Sv while the total is negative, so heat is
# opposing the total, not carrying it; comparing magnitudes alone called that
# bin thermal, which is wrong. Where both qualify, the larger one wins.
SHADE = {0: '#F7D4D4', 1: '#D6E6F2'}                 # red, blue

# The bins are coarse, 0.2 in gamma, while the curves are drawn as straight
# segments between bin centres. Shading by bin would put the boundaries in the
# wrong place: the total crosses zero at 27.10, part-way through a bin, and the
# heat and freshwater lines cross each other between samples too. Resample the
# three curves finely along the drawn segments and classify there, so the
# shading changes exactly where the lines say it should.
gf = np.linspace(gam[mg][0], gam[mg][-1], 2000)
hf = np.interp(gf, gam[mg], heat[mg])
ff = np.interp(gf, gam[mg], fw[mg])
tf = hf + ff

# a term drives the total only if it pushes the same way; where both do, the
# larger one wins
q_heat = np.where(np.sign(hf) == np.sign(tf), np.abs(hf), -1.0)
q_fw = np.where(np.sign(ff) == np.sign(tf), np.abs(ff), -1.0)
who = np.where(q_heat >= q_fw, 0, 1)                 # 0 heat, 1 freshwater

edges = np.r_[0, np.where(np.diff(who) != 0)[0] + 1, len(who)]
for a, b in zip(edges[:-1], edges[1:]):
    ax.axvspan(gf[a], gf[min(b, len(gf) - 1)], color=SHADE[who[a]],
               lw=0, zorder=0)

ax.axhline(0, color='#999999', lw=0.8, zorder=1)
ax.plot(gam[mg], tot[mg], color='#000000', lw=2.4, label='Total surface flux', zorder=3)
ax.plot(gam[mg], heat[mg], color='#E8000B', lw=2.0, label='Surface heat flux', zorder=3)
ax.plot(gam[mg], fw[mg], color='#1F3FD4', lw=2.0, label='Surface fresh water flux', zorder=3)
# the sea ice term is shaded green but was never drawn; show the line too, since
# it is the component the dense classes depend on
ax.plot(gam[mg], ice[mg], color='#1B7F3B', lw=1.8, ls='--',
        label='Sea ice component', zorder=3)

# red dashed divider at the sign change of the total, as in their panel
ax.axvline(ZERO_X, color='#E8000B', lw=1.6, ls='--', zorder=2)

ax.set_xlabel(r'Neutral density, $\gamma$ (kg m$^{-3}$)', fontsize=13)
ax.set_ylabel('Transformation rate (Sv)', fontsize=13)
ax.set_title('(a)  This study, PI, sea ice zone, annual mean',
             fontsize=14, fontweight='bold', loc='left', pad=34)
# the shading needs its own key, otherwise the colours are unexplained
from matplotlib.patches import Patch
lines = ax.get_legend_handles_labels()
patches = [Patch(facecolor=SHADE[0], label='Heat driven'),
           Patch(facecolor=SHADE[1], label='Freshwater driven')]
ax.legend(lines[0] + patches, lines[1] + [p.get_label() for p in patches],
          frameon=False, fontsize=9, ncol=4, columnspacing=1.3,
          handlelength=1.8, loc='upper center', bbox_to_anchor=(0.5, -0.17))
ax.set_xlim(26.3, 28.9)
ax.set_xticks(np.arange(26.3, 28.95, 0.2))
ax.set_xticklabels(['%.1f' % v for v in np.arange(26.3, 28.95, 0.2)], fontsize=9)
ax.tick_params(labelsize=11)

# Lightening / densification arrows ABOVE the axes, as in (b), where they sit
# outside the frame rather than over the data. Axes coordinates keep them clear
# of both the curves and the panel title.
ax.set_ylim(-10.5, 10.5)
yl = ax.get_ylim()
xr_ = (26.3, 28.9)
xf = lambda v: (v - xr_[0]) / (xr_[1] - xr_[0])
zf = xf(ZERO_X)
ax.annotate('', xy=(zf - 0.26, 1.045), xytext=(zf - 0.015, 1.045),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', lw=1.6, color='#111111'),
            annotation_clip=False)
ax.text(zf - 0.14, 1.075, 'Lightening', ha='center', fontsize=10,
        transform=ax.transAxes, clip_on=False)
ax.annotate('', xy=(zf + 0.26, 1.045), xytext=(zf + 0.015, 1.045),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', lw=1.6, color='#111111'),
            annotation_clip=False)
ax.text(zf + 0.14, 1.075, 'Densification', ha='center', fontsize=10,
        transform=ax.transAxes, clip_on=False)

# buoyancy gain / loss at the left edge, where the curves have not started
ax.annotate('', xy=(0.045, 0.78), xytext=(0.045, 0.58),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', lw=1.4, color='#111111'))
ax.text(0.075, 0.70, 'Buoyancy loss', fontsize=9.5, va='center',
        transform=ax.transAxes)
ax.annotate('', xy=(0.045, 0.22), xytext=(0.045, 0.42),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', lw=1.4, color='#111111'))
ax.text(0.075, 0.30, 'Buoyancy gain', fontsize=9.5, va='center',
        transform=ax.transAxes)

# ------------------------------------------------------- their published panel
ax2 = fig.add_subplot(gs[0, 1])
img = mpimg.imread(OBS)
# top-left panel (a) of their figure 3
h, w = img.shape[:2]
ax2.imshow(img[0:int(h*0.355), 0:int(w*0.60)])
ax2.axis('off')
ax2.set_title('(b)  Pellichero et al. (2018), their Fig. 3a',
              fontsize=14, fontweight='bold', loc='left')

fig.savefig('figures/figR16_pi_vs_pellichero.pdf', dpi=400)
fig.savefig('figures/figR16_pi_vs_pellichero.png', dpi=200)
print('saved figures/figR16_pi_vs_pellichero.{pdf,png}')

# numbers quoted in the letter
pk = np.nanargmax(tot)
print('\nPI, sea ice zone, annual mean:')
print('  peak total transformation %.1f Sv at sigma2 %.2f' % (tot[pk], sig[pk]))
print('  heat %.1f Sv, freshwater %.1f Sv at that class' % (heat[pk], fw[pk]))
dense = sig >= 37.0
print('  sigma2 >= 37.0 : heat %.2f Sv, freshwater %.2f Sv'
      % (heat[dense].sum(), fw[dense].sum()))
