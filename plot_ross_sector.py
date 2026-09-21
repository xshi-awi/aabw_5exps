#!/usr/bin/env python
"""
Why the Ross sector behaves differently, answering Reviewer 1 comment 13.

The reviewer asked why intensified wind stress does not produce more Ross Sea
ice, and suggested that relatively warm water might be the reason. Checking the
output shows the mechanism is real but belongs to the last interglacial, not to
the glacial states the comment was aimed at.

(a) Winter surface temperature above the local freezing point, by sector and state.
(b) Winter sea ice concentration, by sector and state.
(c) Anomalies relative to PI for the two quantities, which isolates the LIG Ross
    signal: the sector warms by about 1 K and loses 0.19 of ice concentration,
    an order of magnitude more than the other sectors, while in the glacial states
    all three sectors sit within about 0.1 K of freezing and gain ice comparably.

All quantities are JJA means, area weighted between 60 and 75 S.
"""
import numpy as np
import xarray as xr
import gsw
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

EXPS = ['pi', 'mh', 'lig', 'lgm', 'mis']
LABELS = ['PI', 'MH', 'LIG', 'LGM', 'MIS3']
MESH = {'pi': 'mesh_core2', 'mh': 'mesh_core2', 'lig': 'mesh_core2',
        'lgm': 'mesh_glac1d', 'mis': 'mesh_glac1d_38k'}
SECTORS = [('Ross', -180, -60), ('Weddell', -60, 79), ('Adélie', 79, 180)]
SCOL = {'Ross': '#D55E00', 'Weddell': '#0072B2', 'Adélie': '#009E73'}

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#333333'})


def sector_means(exp):
    m = xr.open_dataset(f'/home/a/a270064/bb1029/inputs/{MESH[exp]}/fesom.mesh.diag.nc')
    lat = np.degrees(m['nodes'].isel(n2=1).values)
    lon = np.degrees(m['nodes'].isel(n2=0).values)
    area = (m['nod_area'].isel(nz=0).values if 'nz' in m['nod_area'].dims
            else m['nod_area'].isel(nl=0).values)
    sst = xr.open_dataset(f'{exp}/sst_clim.nc')['sst'].values[[5, 6, 7]].mean(axis=0)
    sss = xr.open_dataset(f'{exp}/sss_clim.nc')['sss'].values[[5, 6, 7]].mean(axis=0)
    ai = xr.open_dataset(f'{exp}/a_ice_clim.nc')['a_ice'].values[[5, 6, 7]].mean(axis=0)
    m.close()
    out = {}
    for nm, lo, hi in SECTORS:
        s = (lat < -60) & (lat > -75) & (lon >= lo) & (lon < hi)
        w = area[s] / area[s].sum()
        SA = gsw.SA_from_SP(sss[s], 0, lon[s], lat[s])
        CT = gsw.CT_from_pt(SA, sst[s])
        tf = gsw.CT_freezing(SA, 0, 0)
        out[nm] = dict(dTf=float(((CT - tf) * w).sum()),
                       ice=float((ai[s] * w).sum()))
    return out


D = {e: sector_means(e) for e in EXPS}

fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
x = np.arange(len(EXPS))
bw = 0.26

# ---------------------------------------------------------------- (a)
ax = axes[0]
for k, (nm, _, _) in enumerate(SECTORS):
    v = [D[e][nm]['dTf'] for e in EXPS]
    ax.bar(x + (k - 1) * bw, v, width=bw, color=SCOL[nm], label=nm)
ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=13)
ax.set_ylabel('Surface temperature above freezing (K)', fontsize=13)
ax.set_title('(a)  How far from freezing', fontsize=15, fontweight='bold')
ax.legend(frameon=False, fontsize=12)
ax.grid(axis='y', alpha=0.25, lw=0.6)
ax.annotate('LIG Ross is 2.7 K\nabove freezing', xy=(2 - bw, 2.72),
            xytext=(1.15, 3.15), fontsize=11.5, color='#D55E00',
            arrowprops=dict(arrowstyle='->', color='#D55E00', lw=1.4))
ax.set_ylim(0, 3.8)

# ---------------------------------------------------------------- (b)
ax = axes[1]
for k, (nm, _, _) in enumerate(SECTORS):
    v = [D[e][nm]['ice'] for e in EXPS]
    ax.bar(x + (k - 1) * bw, v, width=bw, color=SCOL[nm], label=nm)
ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=13)
ax.set_ylabel('Winter sea ice concentration', fontsize=13)
ax.set_title('(b)  Winter ice cover', fontsize=15, fontweight='bold')
ax.grid(axis='y', alpha=0.25, lw=0.6)
ax.set_ylim(0, 1.12)

# ---------------------------------------------------------------- (c)
ax = axes[2]
ax.axhline(0, color='#444444', lw=0.9)
ax.axvline(0, color='#444444', lw=0.9)
for nm, _, _ in SECTORS:
    for e, lab in zip(EXPS[1:], LABELS[1:]):
        dT = D[e][nm]['dTf'] - D['pi'][nm]['dTf']
        di = D[e][nm]['ice'] - D['pi'][nm]['ice']
        mk = 'o' if e in ('mh', 'lig') else 's'
        ax.scatter(dT, di, s=170, color=SCOL[nm], marker=mk,
                   edgecolor='white', linewidth=1.3, zorder=5)
        if nm == 'Ross' and e == 'lig':
            ax.annotate('LIG Ross', xy=(dT, di), xytext=(dT - 1.35, di + 0.20),
                        fontsize=12, fontweight='bold', color='#D55E00',
                        arrowprops=dict(arrowstyle='->', color='#D55E00', lw=1.4))
ax.set_xlabel('Change in distance from freezing (K)', fontsize=13)
ax.set_ylabel('Change in ice concentration', fontsize=13)
ax.set_title('(c)  Anomalies relative to PI', fontsize=15, fontweight='bold')
ax.grid(alpha=0.25, lw=0.6)
h = [plt.Line2D([], [], marker='o', ls='', color=SCOL[n], label=n, markersize=11)
     for n, _, _ in SECTORS]
h += [plt.Line2D([], [], marker='o', ls='', color='#888888', label='interglacial', markersize=11),
      plt.Line2D([], [], marker='s', ls='', color='#888888', label='glacial', markersize=11)]
ax.legend(handles=h, frameon=False, fontsize=11, loc='lower left', ncol=2)

for a in axes:
    a.tick_params(labelsize=12)

axes[2].set_xlim(-2.3, 1.6)
axes[2].set_ylim(-0.33, 0.85)
fig.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.13, wspace=0.28)
fig.savefig('figures/figR9_ross_sector.pdf', dpi=400)
fig.savefig('figures/figR9_ross_sector.png', dpi=200)
print('saved figures/figR9_ross_sector.{pdf,png}')

print()
print('%-8s %-9s %9s %9s %10s %10s' % ('exp', 'sector', 'T-Tf', 'ice', 'd(T-Tf)', 'd ice'))
for e, lab in zip(EXPS, LABELS):
    for nm, _, _ in SECTORS:
        print('%-8s %-9s %9.2f %9.3f %10.2f %10.3f'
              % (lab, nm, D[e][nm]['dTf'], D[e][nm]['ice'],
                 D[e][nm]['dTf'] - D['pi'][nm]['dTf'],
                 D[e][nm]['ice'] - D['pi'][nm]['ice']))
