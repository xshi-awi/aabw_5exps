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

The density axis is deliberately NOT converted to neutral density. Our surface
water is lighter than observed, so a shared axis would force a point-by-point
correspondence that does not exist; the offset is itself one of our findings.
Each panel therefore carries its own axis and the comparison is of shape.
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
d.close()

# their figure spans 26.3-28.9 neutral density; ours is shown over the range
# the simulated fluxes actually populate
m = (sig >= 35.8) & (sig <= 37.6)

fig = plt.figure(figsize=(15.5, 5.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.18], wspace=0.18,
                      left=0.055, right=0.985, top=0.90, bottom=0.13)

# ---------------------------------------------------------------- our result
ax = fig.add_subplot(gs[0, 0])
ax.axhline(0, color='#999999', lw=0.8)
ax.plot(sig[m], tot[m], color='#111111', lw=2.8, label='Total surface flux')
ax.plot(sig[m], heat[m], color='#D7263D', lw=2.2, label='Surface heat flux')
ax.plot(sig[m], fw[m], color='#2E86AB', lw=2.2, label='Surface freshwater flux')
ax.set_xlabel(r'Potential density $\sigma_2$ (kg m$^{-3}$)', fontsize=13)
ax.set_ylabel('Transformation rate (Sv)', fontsize=13)
ax.set_title('(a)  This study, PI, sea ice zone, annual mean',
             fontsize=14, fontweight='bold', loc='left')
ax.legend(frameon=False, fontsize=11, loc='upper left')
ax.tick_params(labelsize=11)
ax.grid(axis='y', alpha=0.25, lw=0.6)

# arrows matching their lightening/densification convention
yl = ax.get_ylim()
ax.annotate('', xy=(37.45, yl[1]*0.93), xytext=(36.9, yl[1]*0.93),
            arrowprops=dict(arrowstyle='->', lw=1.4, color='#444444'))
ax.text(37.17, yl[1]*0.97, 'Densification', ha='center', fontsize=10, color='#444444')

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
