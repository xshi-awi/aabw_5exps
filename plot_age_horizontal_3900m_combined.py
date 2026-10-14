"""
Combined plot: Age at 3900m depth
Row 1: Absolute values for PI, MH, LIG, LGM, MIS3 (5 panels)
Row 2: Anomalies MH-PI, LIG-PI, LGM-PI, MIS3-PI (4 panels)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings('ignore')


def set_circular_boundary(ax):
    """给极地投影子图设置圆形边界(避免方形角落空白)"""
    theta = np.linspace(0, 2 * np.pi, 200)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

# ---- Settings ----
DATA_DIR = '/work/ba1066/a270064/cc_projects/aabw_5exps'
DEPTH_TARGET = -3900.0
LAT_CUTOFF = -50.0  # only south of 50S

exps      = ['pi', 'mh', 'lig', 'lgm', 'mis']
exp_names = ['PI', 'MH', 'LIG', 'LGM', 'MIS3']
paleo     = ['mh', 'lig', 'lgm', 'mis']
paleo_names = ['MH\u2212PI', 'LIG\u2212PI', 'LGM\u2212PI', 'MIS3\u2212PI']

# ---- Load data ----
def load_age(exp, depth_idx):
    f = nc.Dataset(f'{DATA_DIR}/{exp}/age_reg.nc')
    lat = f.variables['lat'][:]
    lon = f.variables['lon'][:]
    age = f.variables['age'][0, depth_idx, :, :].astype(float)

    # apply ocean mask
    fm = nc.Dataset(f'{DATA_DIR}/{exp}/mask.nc')
    salt = fm.variables['salt'][0, depth_idx, :, :].astype(float)
    fill_s = fm.variables['salt']._FillValue if hasattr(fm.variables['salt'], '_FillValue') else 1e20
    mask = (salt > fill_s * 0.9) | np.isnan(salt)

    fill_a = f.variables['age']._FillValue if hasattr(f.variables['age'], '_FillValue') else 1e20
    age = np.ma.masked_where((age > fill_a * 0.9) | np.isnan(age) | mask, age)

    # mask north of 50S
    for i, la in enumerate(lat):
        if la >= LAT_CUTOFF:
            age[i, :] = np.ma.masked
    return lat, lon, age

# get depth index from PI file
f0 = nc.Dataset(f'{DATA_DIR}/pi/age_reg.nc')
depth = f0.variables['depth_coord'][:]
depth_idx = int(np.argmin(np.abs(depth - DEPTH_TARGET)))
print(f'Using depth level {depth[depth_idx]} m (index {depth_idx})')

lat, lon, _ = load_age('pi', depth_idx)
ages = {}
for exp in exps:
    _, _, ages[exp] = load_age(exp, depth_idx)

# ---- Colormap for absolute values ----
cmap_abs = plt.cm.get_cmap('plasma_r', 25)
levels_abs = np.linspace(0, 2500, 26)
norm_abs = mcolors.BoundaryNorm(levels_abs, cmap_abs.N)

# ---- Colormap for anomalies (diverging, white near zero) ----
levels_anm = np.array([-1500, -1000, -800, -600, -400, -300, -200, -150,
                        -100, -75, -50, -25, -5, 5, 25, 50, 75, 100,
                        150, 200, 300, 400, 600, 800, 1000, 1500])
cmap_anm = plt.cm.get_cmap('RdBu_r', len(levels_anm) - 1)
norm_anm = mcolors.BoundaryNorm(levels_anm, cmap_anm.N)

# ---- Figure layout ----
# 2 rows: row1=5 panels, row2=4 panels (centered)
proj = ccrs.SouthPolarStereo()
fig = plt.figure(figsize=(18, 13))

import matplotlib.gridspec as gridspec
# 3 rows: row0=panels, row1=abs colorbar, row2=anomaly panels, row3=anm colorbar
gs = gridspec.GridSpec(4, 10, figure=fig,
                       height_ratios=[4, 0.22, 4, 0.22],
                       hspace=0.55, wspace=0.05,
                       left=0.02, right=0.98, top=0.95, bottom=0.07)

axes_row1 = [fig.add_subplot(gs[0, i*2:(i*2+2)], projection=proj) for i in range(5)]
axes_row2 = [fig.add_subplot(gs[2, 1+i*2:(1+i*2+2)], projection=proj) for i in range(4)]

panel_labels = list('abcdefghi')

def plot_polar(ax, data, lat, lon, norm, cmap, label, title, first_col=True):
    ax.set_extent([-180, 180, -90, LAT_CUTOFF], crs=ccrs.PlateCarree())
    set_circular_boundary(ax)
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=3)
    ax.coastlines(linewidth=0.4, color='k', zorder=4)

    # pcolormesh needs 2D lon/lat
    lon2d, lat2d = np.meshgrid(lon, lat)
    # mask north of 50S before plotting
    data_plot = np.ma.masked_where(lat2d >= LAT_CUTOFF, data)

    im = ax.pcolormesh(lon2d, lat2d, data_plot,
                       norm=norm, cmap=cmap,
                       transform=ccrs.PlateCarree(),
                       shading='auto', zorder=2)

    # Labels outside the circular boundary, rotated to follow it, ran into each
    # other between neighbouring panels and ate the space between them. Both
    # sets go inside instead: longitudes upright at the rim, latitudes along the
    # 180 deg meridian, which is where Xiaoxu asked for them.
    # Outside labels only where there is white space: the top (0 deg), the
    # bottom (180 deg), and the west side of the leftmost panel. Anything on the
    # east side lands on the next panel, since the columns sit close together.
    LON_LABELS = ((-120, -60, 180) if first_col else (180,))

    gl = ax.gridlines(draw_labels=False, linewidth=0.6, alpha=0.8,
                      linestyle='--', color='gray', zorder=5)
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = mticker.FixedLocator(np.arange(-80, -49, 10))

    # Longitudes sit just outside the circular boundary, in axes coordinates so
    # they cannot drift onto the data and need no background box. Placed by
    # angle around the circle: the projection puts 0 deg at the top and runs
    # clockwise. The radius is a touch beyond 0.5 so the labels clear the rim
    # without reaching the neighbouring panel.
    import numpy as _np
    R = 0.545
    for lon in LON_LABELS:
        lab = '0°' if lon == 0 else ('180°' if abs(lon) == 180
                                     else '%d°%s' % (abs(lon), 'W' if lon < 0 else 'E'))
        th = _np.radians(lon)
        x = 0.5 + R * _np.sin(th)
        y = 0.5 + R * _np.cos(th)
        # anchor away from the circle so the text grows outwards, not inwards
        ha = 'center' if abs(_np.sin(th)) < 0.3 else ('left' if _np.sin(th) > 0 else 'right')
        va = 'center' if abs(_np.cos(th)) < 0.3 else ('bottom' if _np.cos(th) > 0 else 'top')
        ax.text(x, y, lab, transform=ax.transAxes, ha=ha, va=va,
                fontsize=8.5, color='#333333', zorder=6, clip_on=False)
    for la in (-60, -70, -80):
        ax.text(150, la, '%d°S' % abs(la), transform=ccrs.PlateCarree(),
                ha='center', va='center', fontsize=8.5, color='#333333',
                zorder=6)

    # panel label
    ax.text(0.03, 0.97, f'({label})', transform=ax.transAxes,
            fontsize=19, fontweight='bold', va='top', ha='left',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
    # title
    ax.set_title(title, fontsize=21, fontweight='bold', pad=6)
    return im

# Row 1: absolute values
for i, (exp, name) in enumerate(zip(exps, exp_names)):
    im1 = plot_polar(axes_row1[i], ages[exp], lat, lon,
                     norm_abs, cmap_abs, panel_labels[i], name, first_col=(i == 0))

# Row 2: anomalies
for i, (exp, name) in enumerate(zip(paleo, paleo_names)):
    anm = ages[exp] - ages['pi']
    # set |anm| < 5 to masked (white gap between -5 and 5)
    anm = np.ma.masked_where(np.abs(anm) < 5, anm)
    im2 = plot_polar(axes_row2[i], anm, lat, lon,
                     norm_anm, cmap_anm, panel_labels[5+i], name, first_col=(i == 0))

# ---- Colorbars ----
# Absolute colorbar in gs row 1
cax1 = fig.add_subplot(gs[1, 1:9])
cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm_abs, cmap=cmap_abs),
                   cax=cax1, orientation='horizontal')
cb1.set_label('Age (years)', fontsize=19)
cb1.ax.tick_params(labelsize=15)
cb1.set_ticks(np.arange(0, 2501, 500))

# Anomaly colorbar in gs row 3
cax2 = fig.add_subplot(gs[3, 1:9])
cb2 = plt.colorbar(plt.cm.ScalarMappable(norm=norm_anm, cmap=cmap_anm),
                   cax=cax2, orientation='horizontal', extend='both')
cb2.set_label('Age anomaly (years)', fontsize=19)
cb2.ax.tick_params(labelsize=14)
tick_vals = [-1500, -800, -400, -200, -100, -50, -25,
              25, 50, 100, 200, 400, 800, 1500]
cb2.set_ticks(tick_vals)
cb2.ax.set_xticklabels([str(v) for v in tick_vals], rotation=45, ha='right')

outfile = 'figures/age_horizontal_3900m_combined.pdf'
plt.savefig(outfile, dpi=150, bbox_inches='tight')
print(f'Saved: {outfile}')
plt.savefig(outfile.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f'Saved: {outfile.replace(".pdf", ".png")}')
