#!/usr/bin/env python
"""
Southern Ocean sea ice in AWI-ESM2 PI against OSI SAF CDR v3.0 (1979-2008),
plus the five-state numbers. Writes figR21_seaice_obs.{png,pdf} and
seaice_numbers.txt in this folder. Needs osisaf_sh_monthly_1979_2008.nc from
prep_osisaf_clim.py.

PI is model years 2300-2399 of pi_age (pi/100years/a_ice_mergetime.nc), the
same 100 years as every other PI field in the paper. pi/a_ice_clim.nc is a
different, 10-year (2090-2099) mean of unknown origin; it is only reported
for comparison. a_ice is a fraction although its units attribute says "%".
"""
import os
import warnings
import numpy as np
import netCDF4
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.tri as mtri
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

BASE = '/work/ba1066/a270064/cc_projects/aabw_5exps'
HERE = os.path.join(BASE, 'fig_seaice_obs')
OBS = os.path.join(HERE, 'osisaf_sh_monthly_1979_2008.nc')
MESHROOT = '/home/a/a270064/bb1029/inputs'
EXPS = ['pi', 'mh', 'lig', 'lgm', 'mis']
LABEL = {'pi': 'PI', 'mh': 'MH', 'lig': 'LIG', 'lgm': 'LGM', 'mis': 'MIS3'}
MESH = {'pi': 'mesh_core2', 'mh': 'mesh_core2', 'lig': 'mesh_core2',
        'lgm': 'mesh_glac1d', 'mis': 'mesh_glac1d_38k'}
YEARS = {'pi': '2300-2399', 'mh': '2300-2399', 'lig': '2300-2399 (98 yr, no 2318, 2372)',
         'lgm': '2400-2499', 'mis': '2500-2599'}
SECTORS = [('Weddell', -60, 79), ('Ross', -180, -60), ('Adelie', 79, 180)]
MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
       'Nov', 'Dec']
THR = 0.15
R_KM = 6371.0
LAEA = '+proj=laea +lon_0=0 +datum=WGS84 +ellps=WGS84 +lat_0=-90.0'
TO_LAEA = Transformer.from_crs('EPSG:4326', LAEA, always_xy=True)

C_MOD = '#D55E00'   # Okabe-Ito vermillion: model
C_OBS = '#000000'   # black: observations

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42})

warnings.filterwarnings('ignore', category=RuntimeWarning)
out_lines = []
# Default since 2026-11-25 (user button FT): panels a, b, d, e drawn exactly as the sea ice row
# of SI Fig. A6 (plot_climate_patterns_jja.py, plot_figure2): 1-degree fields, cells below 1%
# masked, Gaussian smoothing sigma=1.5 normalised by the mask (which fills masked cells from
# icy neighbours), contourf; but with the correct PI years 2300-2399, never pi/a_ice_reg.nc.
# --plain gives the 2026-11-22 drawing (native field on EASE2, 1% and 15% lines) as *_plain.*
import sys
SI_STYLE = '--plain' not in sys.argv
# Default since 2026-11-27 (user note in the letter): two panels, JJA only, (a) PI sea ice drawn
# as in the SI figure, (b) PI minus OSI SAF. --six gives the earlier six-panel figure as *_six.*
SIX = '--six' in sys.argv
# 15% lines of the JJA figure. Default since 2026-11-27 05:48 (user button GH "both lines off"):
# no lines, written as figR21_seaice_obs.*. Variants under their own suffix: --with-edges (both
# lines in both panels), --no-obs-edge (model line only), --edges-b-only (both lines in (b) only).
EDGE_MODE = ('withedges' if '--with-edges' in sys.argv else 'noobsedge' if '--no-obs-edge' in sys.argv
             else 'edgesb' if '--edges-b-only' in sys.argv else 'noedges')
PI_REG = f'{BASE}/pi/a_ice_reg_beta.nc'   # 1-degree PI, model years 2300-2399 (not pi/a_ice_reg.nc)
# --pi-other (user note 2026-11-27 09:xx "先替换成另一个海冰文件，我看看效果如何"): the JJA figure
# drawn from the files that SI Fig. A2 reads, pi/a_ice_reg.nc (1 degree) and pi/a_ice_clim.nc
# (mesh). Both are the 2090-2099 mean of an unidentified run, NOT the paper's PI years.
# Written as figR21_seaice_obs_pi9099.*. Numbers in seaice_numbers.txt are not affected.
PI_OTHER = '--pi-other' in sys.argv
if PI_OTHER:
    PI_REG = f'{BASE}/pi/a_ice_reg.nc'


def say(s=''):
    print(s)
    out_lines.append(s)


def laea_km(lon, lat):
    x, y = TO_LAEA.transform(lon, lat)
    return np.asarray(x) / 1e3, np.asarray(y) / 1e3


# ---------------------------------------------------------------- model side
class Mesh:
    """FESOM2 mesh, Southern Hemisphere part, with barycentric sampling."""

    def __init__(self, name):
        d = netCDF4.Dataset(f'{MESHROOT}/{name}/fesom.mesh.diag.nc')
        nodes = d.variables['nodes'][:].data          # radians
        self.lon = np.degrees(nodes[0])
        self.lat = np.degrees(nodes[1])
        self.area = d.variables['nod_area'][0, :].data  # m2, surface level
        elem = d.variables['elem'][:].data.T - 1       # 1-based in file
        d.close()
        self.sh = self.lat < 0
        keep = np.all(self.lat[elem] < -25, axis=1)
        self.idx = np.where(self.lat < -25)[0]
        remap = -np.ones(self.lat.size, int)
        remap[self.idx] = np.arange(self.idx.size)
        x, y = laea_km(self.lon[self.idx], self.lat[self.idx])
        self.tri = mtri.Triangulation(x, y, remap[elem[keep]])
        self.trifinder = self.tri.get_trifinder()

    def weights(self, lon, lat):
        """Triangle vertex indices and barycentric weights for points."""
        x, y = laea_km(lon.ravel(), lat.ravel())
        return self.weights_xy(x, y, lon.shape)

    def weights_xy(self, x, y, shape):
        t = self.trifinder(x, y)
        ok = t >= 0
        v = self.tri.triangles[np.where(ok, t, 0)]
        xv, yv = self.tri.x[v], self.tri.y[v]
        det = ((yv[:, 1] - yv[:, 2]) * (xv[:, 0] - xv[:, 2]) +
               (xv[:, 2] - xv[:, 1]) * (yv[:, 0] - yv[:, 2]))
        w0 = ((yv[:, 1] - yv[:, 2]) * (x - xv[:, 2]) +
              (xv[:, 2] - xv[:, 1]) * (y - yv[:, 2])) / det
        w1 = ((yv[:, 2] - yv[:, 0]) * (x - xv[:, 2]) +
              (xv[:, 0] - xv[:, 2]) * (y - yv[:, 2])) / det
        w = np.stack([w0, w1, 1 - w0 - w1], axis=1)
        return dict(v=self.idx[v], w=w, ok=ok, shape=shape)

    @staticmethod
    def sample(W, z):
        """Linear interpolation of node field z (..., nod) at sampled points."""
        f = np.sum(z[..., W['v']] * W['w'], axis=-1)
        f = np.where(W['ok'], f, np.nan)
        return f.reshape(z.shape[:-1] + W['shape'])


def model_monthly(exp):
    d = netCDF4.Dataset(f'{BASE}/{exp}/100years/a_ice_mergetime.nc')
    a = d.variables['a_ice'][:].data.astype(np.float32)
    d.close()
    nt = a.shape[0]
    return a.reshape(nt // 12, 12, -1)                 # (year, month, nod)


def ext_area_native(a, m):
    """Extent and area in 1e6 km2 on the native mesh (Southern Hemisphere)."""
    w = m.area * m.sh
    flat = a.reshape(-1, a.shape[-1])
    ext = np.array([np.sum((r >= THR) * w) for r in flat]) / 1e12
    are = np.array([np.sum(r.astype(np.float64) * w) for r in flat]) / 1e12
    return ext.reshape(a.shape[:-1]), are.reshape(a.shape[:-1])


# ---------------------------------------------------------------- obs side
def load_obs():
    d = netCDF4.Dataset(OBS)
    c = d.variables['conc'][:].data
    o = dict(conc=c, nd=d.variables['n_days'][:].data,
             years=d.variables['year'][:].data,
             xc=d.variables['xc'][:].data, yc=d.variables['yc'][:].data,
             lat=d.variables['lat'][:].data, lon=d.variables['lon'][:].data,
             dext=d.variables['daily_extent_mean'][:].data,
             cell=float(d.cell_area_km2),
             attrs={k: d.getncattr(k) for k in d.ncattrs()})
    d.close()
    with np.errstate(invalid='ignore'):
        o['clim'] = np.nanmean(c, axis=0)              # (12, y, x)
    o['ocean'] = np.isfinite(o['clim'][8])
    # sanity: the grid's own lat/lon must map onto xc/yc
    gx, gy = laea_km(o['lon'], o['lat'])
    X, Y = np.meshgrid(o['xc'], o['yc'])
    assert np.nanmax(np.abs(gx - X)) < 1.0 and np.nanmax(np.abs(gy - Y)) < 1.0
    return o


def obs_sampler(o, field):
    """Bilinear sampler of an obs field on the EASE2 grid (NaN on land)."""
    f = RegularGridInterpolator((o['yc'][::-1], o['xc']), field[::-1, :],
                                bounds_error=False, fill_value=np.nan)

    def at(lon, lat):
        x, y = laea_km(lon.ravel(), lat.ravel())
        return f(np.column_stack([y, x])).reshape(lon.shape)
    return at


def obs_ext_area(c, cell):
    with np.errstate(invalid='ignore'):
        ext = np.nansum(c >= THR, axis=(-2, -1)) * cell / 1e6
    are = np.nansum(c, axis=(-2, -1)) * cell / 1e6
    return ext, are


# ---------------------------------------------------------------- ice edge
MLON = np.arange(-179.75, 180, 0.5)
MLAT = np.arange(-80, -44.999, 0.05)
GLON, GLAT = np.meshgrid(MLON, MLAT)                    # (lat, lon)


def edge_lat(F, lats=MLAT):
    """Northernmost 15% crossing along each meridian. F is (lat, lon)."""
    ice = np.isfinite(F) & (F >= THR)
    nlat, nlon = F.shape
    edge = np.full(nlon, np.nan)
    for j in range(nlon):
        k = np.where(ice[:, j])[0]
        if k.size == 0:
            continue
        i = k[-1]
        if i + 1 < nlat and np.isfinite(F[i + 1, j]):
            f0, f1 = F[i, j], F[i + 1, j]
            edge[j] = lats[i] + (f0 - THR) / (f0 - f1) * (lats[i + 1] - lats[i])
        else:
            edge[j] = lats[i]
    return edge


def sector_mask(lon, lo, hi):
    if hi >= 180:
        return (lon >= lo) & (lon <= 180)
    return (lon >= lo) & (lon < hi)


def edge_by_sector(edge, lons=MLON):
    r = {}
    for name, lo, hi in SECTORS:
        s = sector_mask(lons, lo, hi)
        r[name] = (np.nanmean(edge[s]), np.sum(np.isfinite(edge[s])), s.sum())
    r['Circumpolar'] = (np.nanmean(edge), np.sum(np.isfinite(edge)), edge.size)
    return r


def fmt_edge(r):
    return '  '.join(f'{k} {v[0]:6.2f}' + ('' if v[1] == v[2] else f' ({v[1]}/{v[2]} meridians)')
                     for k, v in r.items())


# ================================================================= main
def main():
    o = load_obs()
    m = Mesh('mesh_core2')
    pi_y = model_monthly('pi')                       # (100, 12, nod)
    pi_clim = pi_y.mean(axis=0, dtype=np.float64)
    d = netCDF4.Dataset(f'{BASE}/pi/a_ice_clim.nc')
    pi_9099 = d.variables['a_ice'][:].data.astype(np.float64)
    d.close()

    say('Southern Hemisphere sea ice: AWI-ESM2 vs OSI SAF CDR v3.0')
    say('Written by fig_seaice_obs/plot_seaice_obs.py. Extent = area of cells/nodes with')
    say('concentration >= 15%; area = sum of concentration x cell area. Units 1e6 km2.')
    say('Model: native FESOM2 mesh, nod_area level 0 (m2; mesh_core2 ocean total %.1f e6 km2).'
        % (m.area.sum() / 1e12))
    say('Obs: %s, %s v%s, doi:%s.' % (o['attrs']['source_title'], o['attrs']['source_product_id'],
                                       o['attrs']['source_product_version'], o['attrs']['source_doi']))
    say('Obs grid: EASE2 25 km LAEA 432x432, cell area %.1f km2 (from xc/yc spacing).' % o['cell'])
    say('Model PI: pi_age model years 2300-2399 (100 yr, pi/100years/a_ice_mergetime.nc).')
    say('Monthly values are computed from monthly-mean concentration fields for both model')
    say('and obs. "mean" = mean over years of each year\'s monthly extent; "clim field" =')
    say('extent of the multi-year mean concentration field.')
    nd = o['nd']
    say('Obs days per month: before Aug 1987 every other day; Apr-May 1986 have no data;')
    say('Jun 1986 has %d days, Dec 1987 %d days, Jan 1988 %d days. All available days are used.'
        % (nd[7, 5], nd[8, 11], nd[9, 0]))
    say()

    # ------------------------------------------------ 1. seasonal cycle
    oext, oare = obs_ext_area(o['conc'], o['cell'])  # (30, 12)
    have = nd > 0
    oext = np.where(have, oext, np.nan)
    oare = np.where(have, oare, np.nan)
    oext_c, oare_c = obs_ext_area(o['clim'], o['cell'])
    pext, pare = ext_area_native(pi_y, m)              # (100, 12)
    pext_c, pare_c = ext_area_native(pi_clim, m)
    p9_ext, p9_are = ext_area_native(pi_9099, m)

    say('1. MONTHLY SH EXTENT AND AREA (1e6 km2)')
    say('%-4s | %-31s | %-25s | %-31s | %s' % ('', 'OSI SAF 1979-2008 extent', 'OSI SAF area',
                                              'PI 2300-2399 extent', 'PI area'))
    say('%-4s | %6s %6s %6s %6s %5s | %6s %6s %6s %5s | %6s %6s %6s %6s %5s | %6s %6s'
        % ('mon', 'mean', 'min', 'max', 'daily', 'fld', 'mean', 'min', 'max', 'fld',
           'mean', 'min', 'max', 'fld', 'ratio', 'mean', 'fld'))
    for k in range(12):
        say('%-4s | %6.2f %6.2f %6.2f %6.2f %5.2f | %6.2f %6.2f %6.2f %5.2f | %6.2f %6.2f %6.2f %6.2f %5.2f | %6.2f %6.2f'
            % (MON[k], np.nanmean(oext[:, k]), np.nanmin(oext[:, k]), np.nanmax(oext[:, k]),
               np.nanmean(o['dext'][:, k]), oext_c[k],
               np.nanmean(oare[:, k]), np.nanmin(oare[:, k]), np.nanmax(oare[:, k]), oare_c[k],
               pext[:, k].mean(), pext[:, k].min(), pext[:, k].max(), pext_c[k],
               pext[:, k].mean() / np.nanmean(oext[:, k]),
               pare[:, k].mean(), pare_c[k]))
    say('  daily = OSI SAF monthly mean of daily extents (as in the OSI SAF index); fld = clim field.')
    say('  ratio = PI mean extent / OSI SAF mean extent.')
    i87 = list(o['years']).index(1987)
    dec_wo = np.delete(oext[:, 11], i87)
    say('  Dec 1987 (2 days) extent %.2f is the Dec maximum; without it Dec max = %.2f.'
        % (oext[i87, 11], np.nanmax(dec_wo)))
    say('  Annual mean extent: OSI SAF %.2f, PI %.2f. Seasonal range (max-min of mean cycle): OSI SAF %.2f, PI %.2f.'
        % (np.nanmean(oext), pext.mean(), np.nanmax(np.nanmean(oext, 0)) - np.nanmin(np.nanmean(oext, 0)),
           pext.mean(0).max() - pext.mean(0).min()))
    say('  Month of max / min: OSI SAF %s / %s; PI %s / %s.'
        % (MON[np.nanargmax(np.nanmean(oext, 0))], MON[np.nanargmin(np.nanmean(oext, 0))],
           MON[np.argmax(pext.mean(0))], MON[np.argmin(pext.mean(0))]))
    say()
    say('  For comparison, pi/a_ice_clim.nc (10-yr mean of model years 2090-2099, run not')
    say('  identified; NOT the paper\'s PI period), clim-field extent / area:')
    say('  ' + ' '.join(f'{MON[k]} {p9_ext[k]:.2f}/{p9_are[k]:.2f}' for k in range(12)))
    say()

    # ------------------------------------------------ 2. maps, sectors, edges
    W_grid = m.weights(*laea_lonlat_grid(o))
    W_mer = m.weights(GLON, GLAT)
    obs_on_mer = {k: obs_sampler(o, o['clim'][k])(GLON, GLAT) for k in (1, 8)}
    mod_on_mer = {k: Mesh.sample(W_mer, pi_clim[k]) for k in (1, 8)}
    m9_on_mer = {k: Mesh.sample(W_mer, pi_9099[k]) for k in (1, 8)}

    say('2. SECTOR EXTENT/AREA AND 15% ICE EDGE LATITUDE (Sep, Feb)')
    say('Sectors: Weddell 60W-79E, Ross 180-60W, Adelie 79E-180. Edge = northernmost 15%')
    say('crossing along each meridian (0.5 deg spacing, 0.05 deg sampling, 80S-45S),')
    say('averaged over meridians with ice; clim fields. Model sampled by linear interpolation')
    say('on the mesh triangles, obs bilinearly on the EASE2 grid.')
    edges = {}
    for k in (8, 1):
        say(f'  {MON[k]}:')
        for name, lo, hi in SECTORS + [('Circumpolar', -180, 180)]:
            sm = m.sh & sector_mask(m.lon, lo, hi)
            so = o['ocean'] & sector_mask(o['lon'], lo, hi)
            me = np.sum((pi_clim[k] >= THR) * m.area * sm) / 1e12
            ma = np.sum(pi_clim[k] * m.area * sm) / 1e12
            oe = np.sum((o['clim'][k] >= THR) & so) * o['cell'] / 1e6
            oa = np.nansum(np.where(so, o['clim'][k], 0)) * o['cell'] / 1e6
            say('    %-11s extent PI %5.2f obs %5.2f (PI-obs %+5.2f)   area PI %5.2f obs %5.2f (PI-obs %+5.2f)'
                % (name, me, oe, me - oe, ma, oa, ma - oa))
        eo = edge_lat(obs_on_mer[k])
        em = edge_lat(mod_on_mer[k])
        e9 = edge_lat(m9_on_mer[k])
        edges[k] = (em, eo)
        say('    edge lat OSI SAF        : ' + fmt_edge(edge_by_sector(eo)))
        say('    edge lat PI 2300-2399   : ' + fmt_edge(edge_by_sector(em)))
        say('    edge lat PI 2090-2099   : ' + fmt_edge(edge_by_sector(e9)))
        both = np.isfinite(eo) & np.isfinite(em)
        dd = em - eo
        say('    PI-obs edge (meridians with ice in both): ' + '  '.join(
            f'{n} {np.nanmean(dd[both & sector_mask(MLON, lo, hi)]):+.2f}' for n, lo, hi in SECTORS)
            + f'  Circumpolar {np.nanmean(dd[both]):+.2f} deg')
        if k == 8:
            j = np.nanargmax(dd)
            jn = np.nanargmin(dd)
            say('    largest edge deficit (PI south of obs): %.2f deg at %.1fE; smallest: %+.2f deg at %.1fE'
                % (-np.nanmin(dd), MLON[jn], -np.nanmax(dd), MLON[j]))
            for name, lo, hi in SECTORS:
                s = both & sector_mask(MLON, lo, hi)
                say('      %-8s PI edge north of obs on %d of %d meridians; max deficit %.1f deg at %.1fE'
                    % (name, np.sum(dd[s] > 0), s.sum(), -np.nanmin(dd[s]),
                       MLON[s][np.nanargmin(dd[s])]))
            say('      30-degree bands (lon range: PI edge, obs edge, PI-obs):')
            for lo in range(-180, 180, 30):
                s = both & (MLON >= lo) & (MLON < lo + 30)
                say('        %4d..%4dE  %6.2f  %6.2f  %+6.2f'
                    % (lo, lo + 30, np.nanmean(em[s]), np.nanmean(eo[s]), np.nanmean(dd[s])))
    say()

    # ------------------------------------------------ 3. inside observed Sep ice zone
    say('3. INSIDE THE OBSERVED SEPTEMBER 15% CONTOUR')
    obs_raw = obs_on_nodes(o, o['clim'][8], m)
    inside = m.sh & (obs_raw >= THR)
    obs_nodes = np.where(inside, obs_raw, 0.0)
    A = np.sum(m.area * inside)
    miss = np.sum(m.area * inside * (pi_clim[8] < THR))
    miss9 = np.sum(m.area * inside * (pi_9099[8] < THR))
    say('  Native mesh (obs interpolated to model nodes):')
    say('    observed ice zone area on model nodes %.2f e6 km2' % (A / 1e12))
    say('    fraction where PI < 15%%: %.1f%% (PI 2090-2099 file: %.1f%%)' % (100 * miss / A, 100 * miss9 / A))
    say('    mean concentration there: PI %.1f%%, obs %.1f%% (PI 2090-2099 file: %.1f%%)'
        % (100 * np.sum(pi_clim[8] * m.area * inside) / A,
           100 * np.sum(obs_nodes * m.area * inside) / A,
           100 * np.sum(pi_9099[8] * m.area * inside) / A))
    for name, lo, hi in SECTORS:
        s = inside & sector_mask(m.lon, lo, hi)
        As = np.sum(m.area * s)
        say('    %-8s obs zone %.2f e6 km2, PI<15%% on %.1f%%, mean conc PI %.1f%% obs %.1f%%'
            % (name, As / 1e12, 100 * np.sum(m.area * s * (pi_clim[8] < THR)) / As,
               100 * np.sum(pi_clim[8] * m.area * s) / As, 100 * np.sum(obs_nodes * m.area * s) / As))
    outside = m.sh & (obs_raw < THR) & (pi_clim[8] >= THR)
    say('    PI ice (>=15%%) outside the observed zone: %.3f e6 km2' % (np.sum(m.area * outside) / 1e12))
    mod_grid = Mesh.sample(W_grid, pi_clim[8])
    og = o['clim'][8]
    ins = (og >= THR) & np.isfinite(mod_grid)
    lost = np.sum((og >= THR) & ~np.isfinite(mod_grid))
    say('  Cross-check on the EASE2 grid (model interpolated to the obs grid):')
    say('    fraction where PI < 15%%: %.1f%%; mean conc PI %.1f%% obs %.1f%%; %d of %d obs-ice cells'
        ' fall on model land and are skipped'
        % (100 * np.mean(mod_grid[ins] < THR), 100 * np.mean(mod_grid[ins]), 100 * np.mean(og[ins]),
           lost, np.sum(og >= THR)))
    say()

    # Weddell gyre interior, where the model convects
    Fm, Fo = mod_on_mer[8], obs_on_mer[8]
    box = (GLON >= -30) & (GLON <= 30) & (GLAT >= -70) & (GLAT <= -60) & np.isfinite(Fm) & np.isfinite(Fo)
    wb = np.cos(np.radians(GLAT))[box]
    say('  Sep mean conc in the Weddell gyre box 30W-30E, 70S-60S: PI %.1f%%, obs %.1f%%'
        % (100 * np.average(Fm[box], weights=wb), 100 * np.average(Fo[box], weights=wb)))
    for la in (-64, -66, -68, -70):
        row = np.argmin(np.abs(MLAT - la))
        s = (MLON >= -30) & (MLON <= 30)
        say('    along %dS, 30W-30E: PI mean %.1f%% (min %.1f%%, max %.1f%%), obs mean %.1f%%'
            % (-la, 100 * np.nanmean(Fm[row, s]), 100 * np.nanmin(Fm[row, s]),
               100 * np.nanmax(Fm[row, s]), 100 * np.nanmean(Fo[row, s])))
    # Enclosed polynyas: patches of conc < 15% south of 50S that are not
    # connected to the open ocean at 50S (8-connected, EASE2 grid). "coastal"
    # = the patch touches land or ice shelf, "open-ocean" = it does not.
    say('  Enclosed open-water patches (polynyas), Weddell sector 60W-79E, e6 km2:')
    say('    (open water = conc < 15%; a patch is enclosed if it does not connect to 50S)')
    for k in (5, 6, 7, 8, 9):
        F = Mesh.sample(W_grid, pi_clim[k])
        oc, co = polynyas(o, F)
        ob_oc, ob_co = polynyas(o, o['clim'][k])
        say('    %s clim: PI open-ocean %.3f, coastal %.3f | OSI SAF open-ocean %.3f, coastal %.3f'
            % (MON[k], oc, co, ob_oc, ob_co))
    yr_oc = np.zeros((pi_y.shape[0], 5))
    yr_co = np.zeros((pi_y.shape[0], 5))
    for yi in range(pi_y.shape[0]):
        F = Mesh.sample(W_grid, pi_y[yi, 5:10])
        for kk in range(5):
            yr_oc[yi, kk], yr_co[yi, kk] = polynyas(o, F[kk])
    say('  Individual PI years 2300-2399, Weddell sector, enclosed open-ocean polynyas:')
    for kk, k in enumerate(range(5, 10)):
        a = yr_oc[:, kk]
        say('    %s: years with > 0.01 e6 km2: %d of %d; > 0.1: %d; max %.3f e6 km2 (coastal: median %.3f, max %.3f)'
            % (MON[k], np.sum(a > 0.01), a.size, np.sum(a > 0.1), a.max(),
               np.median(yr_co[:, kk]), yr_co[:, kk].max()))
    say()

    # ------------------------------------------------ 4. five states
    say('4. FIVE STATES (native meshes; years: ' + ', '.join(f'{LABEL[e]} {YEARS[e]}' for e in EXPS) + ')')
    jja = [5, 6, 7]
    five = {}
    for e in EXPS:
        me = m if MESH[e] == 'mesh_core2' else Mesh(MESH[e])
        ay = pi_y if e == 'pi' else model_monthly(e)
        clim = ay.mean(axis=0, dtype=np.float64)
        ext, are = ext_area_native(ay, me)
        ext_c, are_c = ext_area_native(clim, me)
        jfield = clim[jja].mean(axis=0)
        je_f, ja_f = ext_area_native(jfield, me)
        Wm = m.weights(GLON, GLAT) if me is m else me.weights(GLON, GLAT)
        ej = edge_by_sector(edge_lat(Mesh.sample(Wm, jfield)))
        es = edge_by_sector(edge_lat(Mesh.sample(Wm, clim[8])))
        five[e] = dict(ext=ext.mean(0), are=are.mean(0), ext_c=ext_c, are_c=are_c,
                       jext=ext[:, jja].mean(), jare=are[:, jja].mean(),
                       jext_f=je_f, jare_f=ja_f, ej=ej, es=es)
        if me is not m:
            del me
    say('  Monthly extent (mean over years), 1e6 km2:')
    say('    %-5s ' % '' + ' '.join(f'{x:>6s}' for x in MON))
    for e in EXPS:
        say('    %-5s ' % LABEL[e] + ' '.join(f'{x:6.2f}' for x in five[e]['ext']))
    say('    %-5s ' % 'OSI' + ' '.join(f'{x:6.2f}' for x in np.nanmean(oext, 0)))
    say('  Monthly area (mean over years), 1e6 km2:')
    for e in EXPS:
        say('    %-5s ' % LABEL[e] + ' '.join(f'{x:6.2f}' for x in five[e]['are']))
    say('    %-5s ' % 'OSI' + ' '.join(f'{x:6.2f}' for x in np.nanmean(oare, 0)))
    say('  JJA (mean of Jun-Aug monthly values over all years; in brackets the extent/area of the')
    say('  JJA-mean concentration field, which is what Fig. 1i-l maps); anomaly vs PI:')
    p = five['pi']
    for e in EXPS:
        f = five[e]
        say('    %-5s extent %5.2f [%5.2f] (%+5.2f, %+4.0f%%)   area %5.2f [%5.2f] (%+5.2f, %+4.0f%%)'
            % (LABEL[e], f['jext'], f['jext_f'], f['jext'] - p['jext'], 100 * (f['jext'] / p['jext'] - 1),
               f['jare'], f['jare_f'], f['jare'] - p['jare'], 100 * (f['jare'] / p['jare'] - 1)))
    e9, a9 = ext_area_native(pi_9099[jja], m)
    j9 = pi_9099[jja].mean(axis=0)
    say('    %-5s extent  n/a  [%5.2f]                  area %5.2f [%5.2f]   <- pi/a_ice_clim.nc, 2090-2099'
        % ('PI*', ext_area_native(j9, m)[0], a9.mean(), ext_area_native(j9, m)[1]))
    say('    MH-PI* area %+5.2f, LIG-PI* area %+5.2f, LGM-PI* %+5.2f, MIS3-PI* %+5.2f'
        % tuple(five[e]['jare'] - a9.mean() for e in ['mh', 'lig', 'lgm', 'mis']))
    ej9 = edge_by_sector(edge_lat(Mesh.sample(W_mer, j9)))
    ojext = np.nanmean(oext[:, jja])
    ojare = np.nanmean(oare[:, jja])
    say('    %-5s extent %5.2f [%5.2f]           area %5.2f [%5.2f]'
        % ('OSI', ojext, obs_ext_area(o['clim'][jja].mean(0), o['cell'])[0], ojare,
           obs_ext_area(o['clim'][jja].mean(0), o['cell'])[1]))
    oj_edge = edge_by_sector(edge_lat(obs_sampler(o, np.nanmean(o['clim'][jja], 0))(GLON, GLAT)))
    say('  JJA 15% edge latitude (JJA-mean field), deg; anomaly vs PI in brackets:')
    names = [s[0] for s in SECTORS] + ['Circumpolar']
    say('    %-5s ' % '' + ' '.join(f'{n:>16s}' for n in names))
    for e in EXPS:
        say('    %-5s ' % LABEL[e] + ' '.join(
            f'{five[e]["ej"][n][0]:7.2f} ({five[e]["ej"][n][0] - p["ej"][n][0]:+5.2f})' for n in names))
    say('    %-5s ' % 'OSI' + ' '.join(f'{oj_edge[n][0]:7.2f}         ' for n in names))
    say('    %-5s ' % 'PI*' + ' '.join(f'{ej9[n][0]:7.2f}         ' for n in names)
        + '  <- 2090-2099 file')
    say('  September 15% edge latitude, deg:')
    for e in EXPS:
        say('    %-5s ' % LABEL[e] + ' '.join(f'{five[e]["es"][n][0]:7.2f}' for n in names))
    say()

    with open(os.path.join(HERE, 'seaice_numbers.txt'), 'w') as fh:
        fh.write('\n'.join(out_lines) + '\n')
    print('wrote seaice_numbers.txt')

    # ------------------------------------------------ figure
    if SIX:
        make_figure(o, m, W_grid, pi_clim, oext, oare, pext, pare)
    else:
        make_figure_jja(o, W_grid, pi_9099 if PI_OTHER else pi_clim)


def polynyas(o, F):
    """Enclosed open-water area (e6 km2) in the Weddell sector on the EASE2 grid.

    Returns (open-ocean patches, coastal patches). Land/ice shelf = NaN in F.
    """
    from scipy import ndimage
    dom = o['lat'] < -50
    sea = dom & np.isfinite(F)
    water = sea & (F < THR)
    st = np.ones((3, 3), bool)
    lab, n = ndimage.label(water, structure=st)
    rim = dom & ~ndimage.binary_erosion(dom)
    open_ids = set(np.unique(lab[rim & water])) | {0}
    landish = dom & ~np.isfinite(F)
    wed = sector_mask(o['lon'], -60, 79)
    oc = co = 0.0
    for i in range(1, n + 1):
        if i in open_ids:
            continue
        c = lab == i
        if not np.any(c & wed):
            continue
        a = c.sum() * o['cell'] / 1e6
        if np.any(ndimage.binary_dilation(c, structure=st) & landish):
            co += a
        else:
            oc += a
    return oc, co


def laea_lonlat_grid(o):
    return o['lon'].astype(np.float64), o['lat'].astype(np.float64)


def obs_on_nodes(o, field, m):
    """Obs field at model nodes: bilinear, nearest valid cell where that fails."""
    f = np.full(m.lat.size, np.nan)
    s = m.lat < -40
    x, y = laea_km(m.lon[s], m.lat[s])
    rgi = RegularGridInterpolator((o['yc'][::-1], o['xc']), field[::-1, :],
                                  bounds_error=False, fill_value=np.nan)
    v = rgi(np.column_stack([y, x]))
    bad = ~np.isfinite(v)
    X, Y = np.meshgrid(o['xc'], o['yc'])
    ok = np.isfinite(field)
    tree = cKDTree(np.column_stack([X[ok], Y[ok]]))
    dist, j = tree.query(np.column_stack([x[bad], y[bad]]))
    v[bad] = np.where(dist < 50.0, field[ok][j], np.nan)
    f[s] = v
    return f


# ---------------------------------------------------------------- figure
def circular(ax):
    t = np.linspace(0, 2 * np.pi, 200)
    c = np.vstack([np.sin(t), np.cos(t)]).T
    ax.set_boundary(mpath.Path(c * 0.5 + 0.5, closed=True), transform=ax.transAxes)


def map_axes(fig, spec, laea):
    ax = fig.add_subplot(spec, projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    circular(ax)
    ax.set_facecolor('#d9d9d9')
    return ax


def decorate(ax, fs=8.5, col='#222222', lw_coast=0.4, lw_grid=0.5, lon_lat=-46.6):
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#9e9e9e',
                   edgecolor='none', zorder=3)
    ax.coastlines('50m', linewidth=lw_coast, color='#444444', zorder=4)
    ax.gridlines(crs=ccrs.PlateCarree(), xlocs=np.arange(-180, 181, 60),
                 ylocs=[-80, -70, -60], linewidth=lw_grid, color='#555555',
                 alpha=0.6, linestyle='--', zorder=5)
    for la in (-60, -70):
        ax.text(150, la, f'{-la}°S', transform=ccrs.PlateCarree(), fontsize=fs,
                ha='center', va='center', zorder=6, color=col,
                bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.75))
    for lo, lab in [(0, '0°'), (60, '60°E'), (120, '120°E'), (180, '180°'),
                    (-120, '120°W'), (-60, '60°W')]:
        ax.text(lo, lon_lat, lab, transform=ccrs.PlateCarree(), fontsize=fs,
                ha='center', va='center', clip_on=False, color=col)


def smooth_si(data, sigma=1.5):
    """smooth_data of plot_climate_patterns_jja.py, unchanged."""
    from scipy.ndimage import gaussian_filter
    f = np.nan_to_num(data, nan=0.0)
    w = (~np.isnan(data)).astype(float)
    sm, sw = gaussian_filter(f, sigma), gaussian_filter(w, sigma)
    r = sm / sw
    r[sw < 0.1] = np.nan
    return r


def si_fields(o):
    """1-degree PI and OSI SAF fields for September and February (fractions)."""
    d = netCDF4.Dataset(PI_REG)
    a = d.variables['a_ice'][:, 0].data.astype(np.float64)       # (12, 180, 360)
    lon, lat = d.variables['lon'][:].data, d.variables['lat'][:].data
    d.close()
    out = {('mod', k): a[k] for k in (1, 8)}
    ok = o['ocean']
    i = np.floor(o['lat'][ok] + 90).astype(int)
    j = np.floor((o['lon'][ok] + 180) % 360).astype(int)
    for k in (1, 8):
        c = o['clim'][k][ok]
        g = np.isfinite(c)
        s_, n = np.zeros((180, 360)), np.zeros((180, 360))
        np.add.at(s_, (i[g], j[g]), c[g])
        np.add.at(n, (i[g], j[g]), 1)
        f = np.full((180, 360), np.nan)
        f[n > 0] = s_[n > 0] / n[n > 0]
        out[('obs', k)] = f
    return out, lon, lat


def make_figure(o, m, W_grid, pi_clim, oext, oare, pext, pare):
    laea = ccrs.LambertAzimuthalEqualArea(central_longitude=0, central_latitude=-90,
                                          globe=ccrs.Globe(ellipse='WGS84'))
    X, Y = o['xc'] * 1e3, o['yc'] * 1e3
    fig = plt.figure(figsize=(15, 11.0))
    gs = fig.add_gridspec(2, 3, left=0.03, right=0.985, top=0.94, bottom=0.15,
                          wspace=0.16, hspace=0.30)
    # Concentration panels drawn as sea ice is drawn in manuscript Fig. 2
    # (plot_climate_patterns_jja.py, plot_figure2): NCL BlAqGrYeOrRe colours, levels
    # 10-100% in 5% steps, values from 1 to 10% in the under colour (deepest blue), and
    # only cells with less than 1% ice left white (2026-11-22, user request). Fig. 2 also
    # smooths its 1-degree field; that smoothing is not applied here.
    blaq = ['#0000FF', '#0040FF', '#0080FF', '#00BFFF', '#00FFFF', '#00FFB0', '#00FF60',
            '#00FF00', '#60FF00', '#BFFF00', '#FFFF00', '#FFD000', '#FFA000', '#FF6000',
            '#FF2000', '#FF0000', '#CC0000']
    cmap = mpl.colors.LinearSegmentedColormap.from_list('BlAqGrYeOrRe', blaq, N=256)
    cmap.set_under('#0000FF')
    cmap.set_bad('white')
    cmap_d = mpl.cm.get_cmap('RdBu')
    norm = mpl.colors.BoundaryNorm(np.arange(10, 101, 5), cmap.N, extend='min')
    norm_d = mpl.colors.BoundaryNorm(np.arange(-90, 91, 20), cmap_d.N, extend='both')

    if SI_STYLE:
        sif, slon, slat = si_fields(o)
        from cartopy.util import add_cyclic_point

    def conc_panel(spec, field, title, edge_col, ls='-', si=None):
        ax = map_axes(fig, spec, laea)
        ax.set_facecolor('white')
        f = np.where(o['ocean'], field, np.nan) * 100
        f0 = np.where(np.isfinite(f), f, 0)
        if si is not None:
            # SI-figure drawing; the 15% line stays the unsmoothed field of the numbers
            g = smooth_si(np.where(si < 0.01, np.nan, si)) * 100
            gc, lc = add_cyclic_point(g, coord=slon)
            L, La = np.meshgrid(lc, slat)
            pm = ax.contourf(L, La, gc, levels=np.arange(10, 101, 5), cmap=cmap,
                             extend='min', transform=ccrs.PlateCarree())
        else:
            f = np.where(f < 1, np.nan, f)                  # less than 1% ice: white
            pm = ax.pcolormesh(X, Y, f, cmap=cmap, norm=norm, transform=laea,
                               shading='nearest', rasterized=True)
            ax.contour(X, Y, f0, levels=[1], colors=edge_col, linewidths=1.0,
                       linestyles=':', transform=laea, zorder=4)
        ax.contour(X, Y, f0, levels=[15], colors=edge_col,
                   linewidths=1.3, linestyles=ls, transform=laea, zorder=4)
        decorate(ax)
        ax.set_title(title, fontsize=12.5, pad=26, loc='left')
        return ax, pm

    mod = {k: Mesh.sample(W_grid, pi_clim[k]) for k in (1, 8)}
    S = (lambda w, k: sif[(w, k)]) if SI_STYLE else (lambda w, k: None)
    ax_a, pm = conc_panel(gs[0, 0], mod[8], '(a)  AWI-ESM2 PI, September', C_MOD, '--', S('mod', 8))
    ax_b, _ = conc_panel(gs[0, 1], o['clim'][8], '(b)  OSI SAF 1979–2008, September', C_OBS,
                         si=S('obs', 8))
    ax_d, _ = conc_panel(gs[1, 0], mod[1], '(d)  AWI-ESM2 PI, February', C_MOD, '--', S('mod', 1))
    ax_e, _ = conc_panel(gs[1, 1], o['clim'][1], '(e)  OSI SAF 1979–2008, February', C_OBS,
                         si=S('obs', 1))

    # (c) difference
    ax_c = map_axes(fig, gs[0, 2], laea)
    diff = (mod[8] - o['clim'][8]) * 100
    pd = ax_c.pcolormesh(X, Y, diff, cmap=cmap_d, norm=norm_d, transform=laea,
                         shading='nearest', rasterized=True)
    ax_c.contour(X, Y, np.nan_to_num(o['clim'][8]), levels=[THR], colors=C_OBS,
                 linewidths=1.4, transform=laea, zorder=4)
    ax_c.contour(X, Y, np.nan_to_num(mod[8]), levels=[THR], colors=C_MOD,
                 linewidths=1.6, linestyles='--', transform=laea, zorder=4)
    decorate(ax_c)
    for lo in (-60, 79, 180):
        ax_c.plot([lo, lo], [-72, -50], transform=ccrs.PlateCarree(), color='#7a7a7a',
                  lw=0.9, ls=':', zorder=5)
    ax_c.set_title('(c)  PI minus OSI SAF, September', fontsize=12.5, pad=26, loc='left')
    h1 = mpl.lines.Line2D([], [], color=C_OBS, lw=1.4, label='OSI SAF 15%')
    h2 = mpl.lines.Line2D([], [], color=C_MOD, lw=1.6, ls='--', label='PI 15%')
    ax_c.legend(handles=[h1, h2], loc='lower right', bbox_to_anchor=(1.12, -0.06),
                fontsize=9, frameon=False)

    # colour bars
    p_d = ax_d.get_position()
    p_e = ax_e.get_position()
    cax = fig.add_axes([p_d.x0 + 0.06, 0.085, p_e.x1 - p_d.x0 - 0.12, 0.017])
    cb = fig.colorbar(pm, cax=cax, orientation='horizontal', ticks=np.arange(10, 101, 10),
                      extend='min')
    if SI_STYLE:
        cb.set_label('Sea ice concentration (%), panels a, b, d, e, smoothed as in the SI figure',
                     fontsize=11)
        hl = [mpl.lines.Line2D([], [], color=C_OBS, lw=1.3, label='OSI SAF 15% (not smoothed)'),
              mpl.lines.Line2D([], [], color=C_MOD, lw=1.3, ls='--', label='PI 15% (not smoothed)')]
    else:
        cb.set_label('Sea ice concentration (%), panels a, b, d, e. Arrow: 1-10%; white: below 1%',
                     fontsize=11)
        hl = [mpl.lines.Line2D([], [], color=C_OBS, lw=1.3, label='OSI SAF 15%'),
              mpl.lines.Line2D([], [], color=C_OBS, lw=1.0, ls=':', label='OSI SAF 1%'),
              mpl.lines.Line2D([], [], color=C_MOD, lw=1.3, ls='--', label='PI 15%'),
              mpl.lines.Line2D([], [], color=C_MOD, lw=1.0, ls=':', label='PI 1%')]
    fig.legend(handles=hl, loc='lower center', bbox_to_anchor=(0.5 * (p_d.x0 + p_e.x1), 0.0),
               ncol=len(hl), fontsize=9.5, frameon=False)
    cb.ax.tick_params(labelsize=9.5)
    p_c = ax_c.get_position()
    cax2 = fig.add_axes([p_c.x0 + 0.03, p_c.y0 - 0.045, p_c.width - 0.06, 0.014])
    cb2 = fig.colorbar(pd, cax=cax2, orientation='horizontal', ticks=np.arange(-90, 91, 30),
                       extend='both')
    cb2.set_label('Concentration difference (%)', fontsize=10.5)
    cb2.ax.tick_params(labelsize=9)

    # (f) seasonal cycle
    ax_f = fig.add_subplot(gs[1, 2])
    pos = ax_f.get_position()
    ax_f.set_position([pos.x0 + 0.035, pos.y0 + 0.01, pos.width - 0.035, pos.height - 0.06])
    mon = np.arange(1, 13)
    ax_f.fill_between(mon, np.nanmin(oext, 0), np.nanmax(oext, 0), color='#bdbdbd',
                      alpha=0.7, lw=0, label='OSI SAF extent, min–max')
    ax_f.fill_between(mon, pext.min(0), pext.max(0), color=C_MOD, alpha=0.18, lw=0,
                      label='PI extent, min–max')
    ax_f.plot(mon, np.nanmean(oext, 0), color=C_OBS, lw=2.0, marker='o', ms=3.5,
              label='OSI SAF extent')
    ax_f.plot(mon, pext.mean(0), color=C_MOD, lw=2.0, marker='o', ms=3.5,
              label='PI extent')
    ax_f.plot(mon, np.nanmean(oare, 0), color=C_OBS, lw=1.5, ls='--', label='OSI SAF area')
    ax_f.plot(mon, pare.mean(0), color=C_MOD, lw=1.5, ls='--', label='PI area')
    ax_f.set_xticks(mon)
    ax_f.set_xticklabels([s[0] for s in MON], fontsize=10)
    ax_f.set_xlim(0.6, 12.4)
    ax_f.set_ylim(0, 25)
    ax_f.set_yticks(np.arange(0, 26, 5))
    ax_f.set_ylabel('Sea ice extent or area (10$^6$ km$^2$)', fontsize=11)
    ax_f.tick_params(labelsize=10)
    ax_f.spines['top'].set_visible(False)
    ax_f.spines['right'].set_visible(False)
    ax_f.grid(axis='y', color='#e0e0e0', lw=0.6)
    ax_f.legend(loc='upper left', fontsize=8.6, frameon=False, ncol=1)
    ax_f.set_title('(f)  Southern Hemisphere seasonal cycle', fontsize=12.5, pad=26, loc='left')

    for ext in ('png', 'pdf'):
        fn = os.path.join(HERE, f'figR21_seaice_obs_six{"" if SI_STYLE else "_plain"}.{ext}')
        fig.savefig(fn, dpi=200 if ext == 'png' else 300)
        print('saved', fn)


def make_figure_jja(o, W_grid, pi_clim):
    """Two panels, JJA (June-August): (a) PI drawn as the sea ice row of the SI figure,
    (b) PI minus OSI SAF on the EASE2 grid. The 15% lines are the unsmoothed fields."""
    from cartopy.util import add_cyclic_point
    laea = ccrs.LambertAzimuthalEqualArea(central_longitude=0, central_latitude=-90,
                                          globe=ccrs.Globe(ellipse='WGS84'))
    X, Y = o['xc'] * 1e3, o['yc'] * 1e3
    J = (5, 6, 7)
    mod = np.mean([Mesh.sample(W_grid, pi_clim[k]) for k in J], axis=0)
    obs = np.mean([o['clim'][k] for k in J], axis=0)
    d = netCDF4.Dataset(PI_REG)
    a = d.variables['a_ice'][:, 0].data.astype(np.float64)
    slon, slat = d.variables['lon'][:].data, d.variables['lat'][:].data
    d.close()
    a = a[list(J)].mean(axis=0)
    blaq = ['#0000FF', '#0040FF', '#0080FF', '#00BFFF', '#00FFFF', '#00FFB0', '#00FF60',
            '#00FF00', '#60FF00', '#BFFF00', '#FFFF00', '#FFD000', '#FFA000', '#FF6000',
            '#FF2000', '#FF0000', '#CC0000']
    cmap = mpl.colors.LinearSegmentedColormap.from_list('BlAqGrYeOrRe', blaq, N=256)
    cmap.set_under('#0000FF')
    cmap_d = mpl.cm.get_cmap('RdBu')
    norm_d = mpl.colors.BoundaryNorm(np.arange(-90, 91, 20), cmap_d.N, extend='both')
    # 2026-11-27: printed 5.8 in wide in the letter. Figure 7.4 in wide, so every size below
    # prints at 0.79 of its value (titles 9 pt, labels and ticks >= 7.7 pt, all black).
    fig = plt.figure(figsize=(7.4, 4.5))
    gs = fig.add_gridspec(1, 2, left=0.035, right=0.965, top=0.86, bottom=0.235, wspace=0.12)
    FS_T, FS_L, FS_TK = 11.5, 10.5, 9.8

    def lines(ax, panel):
        if EDGE_MODE == 'noedges' or (EDGE_MODE == 'edgesb' and panel == 'a'):
            return
        # land stays NaN so that the 15% lines do not trace the coastline
        if EDGE_MODE != 'noobsedge':
            ax.contour(X, Y, np.where(o['ocean'], obs, np.nan), levels=[THR], colors=C_OBS,
                       linewidths=1.4, transform=laea, zorder=4)
        ax.contour(X, Y, np.where(o['ocean'], mod, np.nan), levels=[THR],
                   colors=C_MOD, linewidths=1.6, linestyles='--', transform=laea, zorder=4)

    def deco(ax):
        decorate(ax, fs=FS_TK, col='black', lw_coast=0.7, lw_grid=0.7, lon_lat=-46.0)

    ax_a = map_axes(fig, gs[0, 0], laea)
    ax_a.set_facecolor('white')
    g = smooth_si(np.where(a < 0.01, np.nan, a)) * 100
    gc, lc = add_cyclic_point(g, coord=slon)
    L, La = np.meshgrid(lc, slat)
    pm = ax_a.contourf(L, La, gc, levels=np.arange(10, 101, 5), cmap=cmap, extend='min',
                       transform=ccrs.PlateCarree())
    lines(ax_a, 'a')
    deco(ax_a)
    ax_a.set_title('(a)  AWI-ESM2 PI, JJA', fontsize=FS_T, fontweight='bold', pad=25)

    ax_b = map_axes(fig, gs[0, 1], laea)
    pd = ax_b.pcolormesh(X, Y, (mod - obs) * 100, cmap=cmap_d, norm=norm_d, transform=laea,
                         shading='nearest', rasterized=True)
    lines(ax_b, 'b')
    deco(ax_b)
    ax_b.set_title('(b)  PI minus OSI SAF 1979–2008, JJA', fontsize=FS_T, fontweight='bold', pad=25)

    pa, pb = ax_a.get_position(), ax_b.get_position()
    cax = fig.add_axes([pa.x0 + 0.03, 0.135, pa.width - 0.06, 0.03])
    cb = fig.colorbar(pm, cax=cax, orientation='horizontal', ticks=np.arange(10, 101, 10),
                      extend='min')
    cb.set_label('Sea ice concentration (%)', fontsize=FS_L, fontweight='bold')
    cb.ax.tick_params(labelsize=FS_TK, labelcolor='black', width=0.8)
    cb.outline.set_linewidth(0.8)
    cax2 = fig.add_axes([pb.x0 + 0.03, 0.135, pb.width - 0.06, 0.03])
    cb2 = fig.colorbar(pd, cax=cax2, orientation='horizontal', ticks=np.arange(-90, 91, 30),
                       extend='both')
    cb2.set_label('Concentration difference (%)', fontsize=FS_L, fontweight='bold')
    cb2.ax.tick_params(labelsize=FS_TK, labelcolor='black', width=0.8)
    cb2.outline.set_linewidth(0.8)
    hl = [mpl.lines.Line2D([], [], color=C_OBS, lw=1.4, label='OSI SAF 15%'),
          mpl.lines.Line2D([], [], color=C_MOD, lw=1.6, ls='--', label='PI 15%')]
    if EDGE_MODE == 'noobsedge':
        hl = hl[1:]
    if EDGE_MODE != 'noedges':
        fig.legend(handles=hl, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=len(hl),
                   fontsize=FS_L, frameon=False)
    suf = '' if EDGE_MODE == 'noedges' else f'_{EDGE_MODE}'
    if PI_OTHER:
        suf += '_pi9099'
    for ext in ('png', 'pdf'):
        fn = os.path.join(HERE, f'figR21_seaice_obs{suf}.{ext}')
        fig.savefig(fn, dpi=290 if ext == 'png' else 300)
        print('saved', fn)


if __name__ == '__main__':
    main()
