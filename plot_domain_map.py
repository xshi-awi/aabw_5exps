#!/usr/bin/env python
"""
The analysis domain, drawn explicitly.

Reviewer 2 wrote that our domain and the domain of Pellichero et al. (2018)
"look similar", which is the premise of their argument that the disagreement must
be model bias. The domains are in fact not similar, and this figure shows why.

(a) The published domain, everything south of 60 S.
(b) The seasonal sea ice zone of the model, inside the September 15% contour,
    which is the definition Pellichero et al. use.
(c) The two overlaid: the orange area is inside our domain but outside the ice
    zone, and is 42.7% of the total. It is open water all year, so thermal
    forcing dominates there by construction.
"""
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42})

MESHDIR = '/home/a/a270064/bb1029/inputs/mesh_core2'


def circular(ax):
    t = np.linspace(0, 2 * np.pi, 200)
    c = np.vstack([np.sin(t), np.cos(t)]).T
    ax.set_boundary(mpath.Path(c * 0.5 + 0.5, closed=True), transform=ax.transAxes)


m = xr.open_dataset(f'{MESHDIR}/fesom.mesh.diag.nc')
lat = np.degrees(m['nodes'].isel(n2=1).values)
lon = np.degrees(m['nodes'].isel(n2=0).values)
area = (m['nod_area'].isel(nz=0).values if 'nz' in m['nod_area'].dims
        else m['nod_area'].isel(nl=0).values)
sep = xr.open_dataset('pi/a_ice_clim.nc')['a_ice'].values[8]   # September

so60 = lat < -60
siz = (sep > 0.15) & (lat < 0)
a_so60 = area[so60].sum()
a_siz = area[siz].sum()
a_open = area[so60 & ~siz].sum()
frac = a_open / a_so60 * 100

# regrid the three masks for plotting
sel = lat < -40
gy = np.arange(-78, -39, 0.4)
gx = np.arange(-180, 180.5, 0.8)
GX, GY = np.meshgrid(gx, gy)


def grid_of(mask):
    f = np.where(mask, 1.0, 0.0)
    return griddata(np.column_stack([lon[sel], lat[sel]]), f[sel], (GX, GY),
                    method='linear')


g_so60 = grid_of(so60)
g_siz = grid_of(siz)
g_sepconc = griddata(np.column_stack([lon[sel], lat[sel]]), sep[sel], (GX, GY),
                     method='linear')

fig = plt.figure(figsize=(17.5, 6.6))
gs = fig.add_gridspec(1, 3, wspace=0.06, left=0.03, right=0.985,
                      top=0.86, bottom=0.15)
proj = ccrs.SouthPolarStereo()

PANELS = [
    ('(a)  Domain used in the paper\nsouth of 60$^\\circ$S',
     g_so60, '#0072B2', f'A = {a_so60/1e13:.2f}$\\times10^{{13}}$ m$^2$'),
    ('(b)  Seasonal sea ice zone\nSeptember ice concentration $>$ 15%',
     g_siz, '#009E73', f'A = {a_siz/1e13:.2f}$\\times10^{{13}}$ m$^2$'),
]

for c, (title, g, col, note) in enumerate(PANELS):
    ax = fig.add_subplot(gs[0, c], projection=proj)
    ax.set_extent([-180, 180, -90, -42], crs=ccrs.PlateCarree())
    circular(ax)
    ax.contourf(gx, gy, g, levels=[0.5, 1.5], colors=[col], alpha=0.55,
                transform=ccrs.PlateCarree())
    ax.contour(gx, gy, g, levels=[0.5], colors=[col], linewidths=2.0,
               transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#999999', zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5,
                      linestyle='--', x_inline=False, y_inline=True)
    gl.top_labels = gl.right_labels = gl.bottom_labels = False
    gl.ylabel_style = {'size': 9, 'color': '#333333'}
    ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
    ax.text(0.5, -0.07, note, transform=ax.transAxes, ha='center', fontsize=12.5)

# ------------------------------------------------- (c) the difference
ax = fig.add_subplot(gs[0, 2], projection=proj)
ax.set_extent([-180, 180, -90, -42], crs=ccrs.PlateCarree())
circular(ax)
diff = np.where((g_so60 > 0.5) & (g_siz < 0.5), 1.0, 0.0)
both = np.where((g_so60 > 0.5) & (g_siz > 0.5), 1.0, 0.0)
ax.contourf(gx, gy, both, levels=[0.5, 1.5], colors=['#009E73'], alpha=0.5,
            transform=ccrs.PlateCarree())
ax.contourf(gx, gy, diff, levels=[0.5, 1.5], colors=['#D55E00'], alpha=0.75,
            transform=ccrs.PlateCarree())
ax.contour(gx, gy, g_sepconc, levels=[0.15], colors=['k'], linewidths=1.8,
           transform=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor='#999999', zorder=3)
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5,
                  linestyle='--', x_inline=False, y_inline=True)
gl.top_labels = gl.right_labels = gl.bottom_labels = False
gl.ylabel_style = {'size': 9, 'color': '#333333'}
ax.set_title('(c)  The difference\nblack line = September ice edge',
             fontsize=15, fontweight='bold', pad=10)
ax.text(0.5, -0.07,
        f'orange: in our domain but outside the ice zone,\n{frac:.1f}% of the total area',
        transform=ax.transAxes, ha='center', fontsize=12)

handles = [mpatches.Patch(color='#009E73', alpha=0.5, label='ice covered in winter'),
           mpatches.Patch(color='#D55E00', alpha=0.75, label='open water all year')]
ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(-0.12, -0.02),
          frameon=False, fontsize=11.5)

fig.savefig('figures/figR8_domain_map.pdf', dpi=400)
fig.savefig('figures/figR8_domain_map.png', dpi=200)
print('saved figures/figR8_domain_map.{pdf,png}')
print(f'  <60S area          {a_so60:.4e} m2')
print(f'  Sept ice zone area {a_siz:.4e} m2')
print(f'  open-water part    {a_open:.4e} m2  = {frac:.1f}% of the <60S domain')
