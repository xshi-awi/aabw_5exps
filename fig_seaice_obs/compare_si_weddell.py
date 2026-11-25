"""Side-by-side for the user (2026-11-25): why the Weddell Sea is white in Figure R3 but
coloured in the SI sea-ice figure (fig02_climate_mld_seaice_jja, panel f).
All panels use the SI drawing: 1-degree field, cells below 1% masked, Gaussian smoothing
sigma=1.5 with mask normalisation (smooth_data of plot_climate_patterns_jja.py), contourf
levels 0.1-1, NCL BlAqGrYeOrRe, under colour #0000FF. Only the data, the months and the
smoothing switch change from panel to panel."""
import numpy as np, xarray as xr, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, cartopy.crs as ccrs, cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
from cartopy.util import add_cyclic_point
B = '/work/ba1066/a270064/cc_projects/aabw_5exps'
OBS = f'{B}/fig_seaice_obs/osisaf_sh_monthly_1979_2008.nc'
blaq = ['#0000FF', '#0040FF', '#0080FF', '#00BFFF', '#00FFFF', '#00FFB0', '#00FF60', '#00FF00',
        '#60FF00', '#BFFF00', '#FFFF00', '#FFD000', '#FFA000', '#FF6000', '#FF2000', '#FF0000', '#CC0000']
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('BlAqGrYeOrRe', blaq, N=256)
cmap.set_under('#0000FF')

def smooth_data(data, sigma=1.5):
    f = np.nan_to_num(data, nan=0.0); m = (~np.isnan(data)).astype(float)
    s = gaussian_filter(f, sigma); sm = gaussian_filter(m, sigma)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = s / sm; r[sm < 0.1] = np.nan
    return r

def model(fn, months):
    ds = xr.open_dataset(fn, decode_times=False)
    a = ds.a_ice.isel(depth_coord=0).isel(time=months).mean('time').values
    return a, ds.lon.values, ds.lat.values

def obs_1deg(month):
    """OSI SAF 1979-2008 climatology of one month (1-12), averaged into the 1-degree cells
    of the model's regular grid (centres -89.5..89.5, -179.5..179.5)."""
    ds = xr.open_dataset(OBS, decode_times=False)
    c = np.nanmean(ds.conc.sel(month=month).values, axis=0)
    la, lo = ds.lat.values, ds.lon.values
    ok = np.isfinite(c)
    i = np.floor(la[ok] + 90).astype(int); j = np.floor(((lo[ok] + 180) % 360)).astype(int)
    s_ = np.zeros((180, 360)); n = np.zeros((180, 360))
    np.add.at(s_, (i, j), c[ok]); np.add.at(n, (i, j), 1)
    with np.errstate(invalid='ignore'):
        out = s_ / n
    out[n == 0] = np.nan
    return out, np.arange(-179.5, 180), np.arange(-89.5, 90)

def si_panel(ax, a, lon, lat, title, smooth=True):
    a = np.where(a < 0.01, np.nan, a)
    if smooth:
        a = smooth_data(a, sigma=1.5)
    ac, lc = add_cyclic_point(a, coord=lon)
    L, La = np.meshgrid(lc, lat)
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    im = ax.contourf(L, La, ac, levels=np.linspace(0.1, 1, 19), cmap=cmap, extend='both',
                     transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5, zorder=3)
    ax.gridlines(crs=ccrs.PlateCarree(), xlocs=np.arange(-180, 181, 60), ylocs=[-70, -60],
                 linewidth=0.4, color='gray', alpha=0.6, linestyle='--')
    ax.set_title(title, fontsize=9.5, loc='left')
    # Weddell box used for the numbers (30W-30E, 60-68S)
    bl = np.r_[np.linspace(-30, 30, 30), np.full(10, 30), np.linspace(30, -30, 30), np.full(10, -30)]
    bt = np.r_[np.full(30, -60), np.linspace(-60, -68, 10), np.full(30, -68), np.linspace(-68, -60, 10)]
    ax.plot(bl, bt, color='k', lw=0.8, ls='--', transform=ccrs.PlateCarree(), zorder=4)
    return im

if __name__ == '__main__':
    fig = plt.figure(figsize=(17, 4.6))
    P = [(f'{B}/pi/a_ice_reg.nc', [5, 6, 7, 8], True,
          '(a) SI figure as published:\noverwritten PI file (2090-2099),\nJun-Sep mean, smoothed'),
         (f'{B}/pi/a_ice_reg_beta.nc', [5, 6, 7, 8], True,
          '(b) same drawing, correct PI\n(model years 2300-2399),\nJun-Sep mean, smoothed'),
         (f'{B}/pi/a_ice_reg_beta.nc', [8], True,
          '(c) same drawing, correct PI,\nSeptember, smoothed\n(candidate for Figure R3a)'),
         (f'{B}/pi/a_ice_reg_beta.nc', [8], False,
          '(d) correct PI, September,\nnot smoothed\n(as Figure R3a now)')]
    for k, (fn, mo, sm, t) in enumerate(P):
        ax = fig.add_subplot(1, 5, k + 1, projection=ccrs.SouthPolarStereo())
        a, lon, lat = model(fn, mo)
        im = si_panel(ax, a, lon, lat, t, smooth=sm)
    ax = fig.add_subplot(1, 5, 5, projection=ccrs.SouthPolarStereo())
    a, lon, lat = obs_1deg(9)
    si_panel(ax, a, lon, lat, '(e) OSI SAF 1979-2008,\nSeptember, same drawing,\nsmoothed')
    cax = fig.add_axes([0.25, 0.13, 0.5, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=np.arange(0.1, 1.01, 0.1))
    cb.set_label('Sea ice concentration (fraction); deepest blue = 0.01-0.1, white = below 0.01 before smoothing', fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.22, wspace=0.05)
    fig.savefig('/work/ba1066/a270064/cc_projects/aabw_5exps/fig_seaice_obs/compare_si_weddell.png', dpi=130)
    print('saved')
