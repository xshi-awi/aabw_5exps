#!/usr/bin/env python
"""Split the paper's surface WMT (south of 60S, sigma2, JJA and annual) by
region and density class, from data/acc_<exp>.nc (calc_formation_location.py).

Usage: python3 -s analyze_formation_location.py [pi mh lig lgm mis]
  default = all five states; numbers.txt is written only for all five.
Writes data/G_<exp>.nc (regional curves), data/regions_<exp>.nc (masks,
fields and class-restricted JJA density-flux maps) and numbers.txt.

G(s) is the transformation across isopycnal s (Sv, positive = towards higher
density); G_r(s) is the part forced inside region r, i.e. the rate at which
region r turns lighter water into water denser than s. Formation in a class
range [a, b) is G(a) - G(b).
"""
import sys
import os
import json
import numpy as np
import netCDF4 as nc
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

try:
    import gsw
except ImportError:
    import tempfile
    _d = tempfile.mkdtemp()
    os.symlink('/home/a/a270064/.local/lib/python3.9/site-packages/gsw', f'{_d}/gsw')
    sys.path.insert(0, _d)
    import gsw

ROOT = '/work/ba1066/a270064/cc_projects/aabw_5exps'
HERE = f'{ROOT}/fig_formation_location'
INP = '/home/a/a270064/bb1029/inputs'
EXPS = ['pi', 'mh', 'lig', 'lgm', 'mis']
SURF_MESH = {'pi': 'mesh_core2', 'mh': 'mesh_core2', 'lig': 'mesh_core2',
             'lgm': 'mesh_glac1d', 'mis': 'mesh_glac1d_38k'}
# the PI/MH/LIG runs used awicm2_final384: same nodes, different bottom depth
RUN_MESH = {'pi': 'awicm2_final384', 'mh': 'awicm2_final384', 'lig': 'awicm2_final384',
            'lgm': 'mesh_glac1d', 'mis': 'mesh_glac1d_38k'}
LABEL = {'pi': 'PI', 'mh': 'MH', 'lig': 'LIG', 'lgm': 'LGM', 'mis': 'MIS3'}
RHO_REF = 1035.0
R_EARTH = 6371.0
JJA = [5, 6, 7]            # clim files: index 0 = January (month-end stamps)
SHELF_DEPTHS = [1000.0, 1500.0]
COAST_KM = [200.0, 100.0, 300.0]
MLD_CONV = 400.0
ICE_EDGE = 0.15
LOWICE = 0.7
DEEP_FOR_BOTTOM = 3000.0
FIXED_S2 = [36.8, 37.0, 37.2]
FIXED_S0 = [26.5, 27.0]
# sectors: the paper's three boxes, with its Ross box split at 140W
SECTORS = [('Weddell 60W-79E', -60, 79), ('Adelie/East 79E-180', 79, 180),
           ('Ross 180-140W', -180, -140), ('Amundsen-Bellingshausen 140W-60W', -140, -60)]


def read_mesh(name):
    d = nc.Dataset(f'{INP}/{name}/fesom.mesh.diag.nc')
    d.set_auto_mask(False)
    out = dict(lon=np.degrees(d['nodes'][0]), lat=np.degrees(d['nodes'][1]),
               zb=np.abs(d['zbar_n_bottom'][:]), nlev=d['nlevels_nod2D'][:],
               edges=d['edges'][:] - 1, edge_tri=d['edge_tri'][:])
    d.close()
    return out


def read_clim(exp, name, var):
    """Climatology on nodes; refuses files that are not from the paper run
    (pi/a_ice_clim.nc was overwritten by another run, see root CLAUDE.md)."""
    fn = f'{ROOT}/{exp}/{name}'
    d = nc.Dataset(fn)
    d.set_auto_mask(False)
    assert f'production/{exp}_age/' in getattr(d, 'history', ''), f'{fn} not from production/{exp}_age'
    v = d[var][:]
    d.close()
    return v


def coast_distance(mesh, sub):
    """Great-circle distance (km) to the Antarctic coastline = the largest
    closed mesh boundary whose nodes lie on average south of 60S."""
    bd = mesh['edge_tri'][1] <= 0
    e = mesh['edges'][:, bd]
    n = mesh['lat'].size
    g = coo_matrix((np.ones(e.shape[1]), (e[0], e[1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    bnodes = np.unique(e)
    best = max((bnodes[lab[bnodes] == c] for c in np.unique(lab[bnodes])
                if mesh['lat'][bnodes[lab[bnodes] == c]].mean() < -60), key=len)
    other = np.setdiff1d(bnodes[mesh['lat'][bnodes] < -60], best).size

    def xyz(i):
        la, lo = np.radians(mesh['lat'][i]), np.radians(mesh['lon'][i])
        return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    chord, _ = cKDTree(xyz(best)).query(xyz(sub))
    return 2 * R_EARTH * np.arcsin(np.clip(chord / 2, 0, 1)), best.size, other


def bottom_sigma2(exp, rmesh):
    """Annual-mean sigma2 of the deepest wet layer (salt as SA, temp as CT,
    the same convention as the WMT)."""
    T = read_clim(exp, 'temp.clim.nc', 'temp').mean(axis=0)     # (nod2, nz1)
    S = read_clim(exp, 'salt.clim.nc', 'salt').mean(axis=0)
    kb = rmesh['nlev'] - 2
    ii = np.arange(kb.size)
    deep = kb + 1 < T.shape[1]
    empty_below = np.abs(T[ii[deep], kb[deep] + 1]).max()
    return gsw.sigma2(S[ii, kb], T[ii, kb]), empty_below, (T[ii, kb], S[ii, kb])


def wq(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w[o].sum()
    return np.interp(q, c, x[o])


def kbin(edges, s):
    return int(np.searchsorted(edges, s, side='right') - 1)


def pct(a, b):
    return 100 * a / b if abs(b) > 1e-9 else np.nan


def main(exp):
    out = []
    P = out.append
    key = {'exp': exp}
    A = nc.Dataset(f'{HERE}/data/acc_{exp}.nc')
    A.set_auto_mask(False)
    sub = A['node_index'][:]
    lat, lon, area = A['lat'][:], A['lon'][:], A['area'][:]
    E = {b: A[f'edges_{b}'][:] for b in ['sig2c', 'sig2f', 'sig0f']}
    acc = {k: A[k][:].astype(np.float64) for k in A.variables if k.startswith('acc_')}
    cntf = A['cnt_jja_sig2f'][:].astype(np.float64)
    Fm = {k: A[k][:] for k in A.variables if k.startswith('F_')}
    sig_jja = A['sig2_mean_jja'][:]
    A.close()
    cen = {b: 0.5 * (e[1:] + e[:-1]) for b, e in E.items()}

    smesh, rmesh = read_mesh(SURF_MESH[exp]), read_mesh(RUN_MESH[exp])
    assert np.allclose(smesh['lat'], rmesh['lat'], atol=1e-3)
    assert np.allclose(np.cos(np.radians(smesh['lon'] - rmesh['lon'])), 1, atol=1e-6)
    zb = rmesh['zb'][sub]
    dist, ncoast, nisl = coast_distance(smesh, sub)
    aice = read_clim(exp, 'a_ice_clim_beta.nc' if exp == 'pi' else 'a_ice_clim.nc', 'a_ice')
    assert np.nanmax(aice) <= 1.0001
    aice_jja = aice[JJA].mean(axis=0)[sub]
    mld_jja = np.abs(read_clim(exp, 'MLD1_clim.nc', 'MLD1'))[JJA].mean(axis=0)[sub]
    mld2_jja = np.abs(read_clim(exp, 'MLD2_clim.nc', 'MLD2'))[JJA].mean(axis=0)[sub]
    sigb_all, empty_below, (tb_all, sb_all) = bottom_sigma2(exp, rmesh)
    sigb, tb, sb = sigb_all[sub], tb_all[sub], sb_all[sub]

    s60 = (lat <= -60) & (lat >= -90)     # the published Southern_Ocean box
    dp = s60 & (zb >= DEEP_FOR_BOTTOM)
    bq = wq(sigb[dp], area[dp], [0.1, 0.5, 0.9])
    key['sig_bot'] = list(bq)

    def G(mask, seas='jja', b='sig2f', comps=('heat', 'ice', 'other')):
        m = mask & s60
        return sum(acc[f'acc_{seas}_{b}_{c}'][m].sum(axis=0) for c in comps) * RHO_REF / 1e9 / np.diff(E[b])

    ALL = np.ones(lat.size, bool)
    P(f'==== {LABEL[exp]} ({exp}) ====')
    P(f'area south of 60S {area[s60].sum():.4e} m2 ({s60.sum()} nodes); Antarctic coastline {ncoast} nodes, '
      f'other boundary nodes south of 60S (islands, not counted as coast) {nisl}')
    P(f'bottom layer check: |T| one layer below the bottom max {empty_below:.1e}')

    # --------------------------------------------- reproduction (0.2 bins)
    pub = nc.Dataset(f'{ROOT}/{exp}/wmt_results/wmt_southern_ocean_100years_{exp}.nc')
    pub.set_auto_mask(False)
    tv = pub['time']
    mon = np.array([d.month for d in nc.num2date(tv[:], tv.units, tv.calendar)])
    assert np.allclose(pub['sigma2_l_target'][:], cen['sig2c'])
    vn = ['total_heat_surface_exchange_flux_nonadvective_heat'] + \
        [f'surface_ocean_flux_advective_negative_rhs_{k}_salt'
         for k in ['sea_ice_melt', 'evaporation', 'snow', 'rain_and_ice', 'rivers']]
    raw = sum(np.nan_to_num(pub[v][:]) for v in vn)
    pub.close()
    Gc = {}
    for s, sel in [('jja', np.isin(mon, [6, 7, 8])), ('ann', np.ones(mon.size, bool))]:
        gp = -raw[sel].mean(axis=0) / 1e9
        Gc[s] = G(ALL, s, 'sig2c')
        k = int(np.argmax(gp))
        d = np.abs(Gc[s] - gp).max()
        P(f'reproduction {s.upper()} (published 0.2 bins): published peak {gp[k]:.3f} Sv at {cen["sig2c"][k]:.1f}, '
          f'this script {Gc[s][k]:.3f} Sv; max |difference| over all bins {d:.1e} Sv ({d / gp[k]:.1e} of peak)')
        key[f'repro_{s}'] = [float(gp[k]), float(Gc[s][k]), float(d)]

    # --------------------------------------------- class thresholds
    gf = G(ALL)
    gf[[0, -1]] = 0.0                    # under/overflow bins have no width
    kc = int(np.argmax(Gc['jja']))        # published peak bin
    pk_lo, pk_hi = E['sig2c'][kc], E['sig2c'][kc + 1]
    kf = int(np.argmax(gf))
    gmax = gf[kf]
    dense = np.arange(kf, gf.size)
    k10 = dense[gf[dense] >= 0.1 * gmax].max() + 1
    k50 = dense[gf[dense] >= 0.5 * gmax].max() + 1
    occ = (cntf[s60] * area[s60][:, None]).sum(axis=0)
    cdf = np.cumsum(occ) / occ.sum()
    kp99 = int(np.searchsorted(cdf, 0.99))
    ef = E['sig2f']
    thr = [('published peak bin, lower edge', kbin(ef, pk_lo + 1e-6)),
           ('published peak bin, upper edge', kbin(ef, pk_hi + 1e-6)),
           ('bottom water p10', kbin(ef, bq[0])), ('bottom water median', kbin(ef, bq[1])),
           ('bottom water p90', kbin(ef, bq[2])),
           ('sigma_50 (G<50% of fine peak beyond)', int(k50)),
           ('sigma_10 (G<10% of fine peak beyond)', int(k10)),
           ('surface sigma2 p99 (JJA)', kp99)]
    thr += [(f'fixed sigma2 {x:.1f}', kbin(ef, x + 1e-6)) for x in FIXED_S2]
    ks = {n: k for n, k in thr}
    cf = cen['sig2f']
    P(f'JJA published-bin peak {Gc["jja"][kc]:.2f} Sv in sigma2 {pk_lo:.1f}-{pk_hi:.1f}; '
      f'0.02-bin peak {gmax:.2f} Sv at {cf[kf]:.2f}; sigma_50 {cf[k50]:.2f}; sigma_10 {cf[k10]:.2f} '
      f'(G there {gf[k10]:.2f} Sv); surface JJA sigma2 p99 {cf[kp99]:.2f}')
    P(f'bottom water sigma2 (annual clim, deepest layer, H>={DEEP_FOR_BOTTOM:.0f} m, south of 60S, area weights): '
      f'p10 {bq[0]:.3f}, median {bq[1]:.3f}, p90 {bq[2]:.3f}; median bottom T {wq(tb[dp], area[dp], 0.5):.2f} C, '
      f'S {wq(sb[dp], area[dp], 0.5):.3f} (range S {sb[dp].min():.2f}-{sb[dp].max():.2f})')
    key.update(peak_bin=[float(pk_lo), float(pk_hi)], G_peak_bin=float(Gc['jja'][kc]),
               fine_peak=[float(cf[kf]), float(gmax)], sig10=float(cf[k10]), G_sig10=float(gf[k10]),
               sig50=float(cf[k50]), surf_p99=float(cf[kp99]))

    # --------------------------------------------- partitions
    parts = {}
    for H in SHELF_DEPTHS:
        shelf = zb < H
        conv = ~shelf & (mld_jja > MLD_CONV)
        ice = ~shelf & ~conv & (aice_jja >= ICE_EDGE)
        parts[f'depth{int(H)}'] = [(f'shelf/slope H<{int(H)} m', shelf),
                                   ('deep, convective MLD>400 m', conv),
                                   ('deep, under sea ice', ice),
                                   ('deep, ice-free', ~shelf & ~conv & ~ice)]
    # sensitivity: convective patch from MLD2 instead of MLD1 (MLD1 is what Fig. A2/R6 show)
    shelf = zb < 1000
    conv = ~shelf & (mld2_jja > MLD_CONV)
    ice = ~shelf & ~conv & (aice_jja >= ICE_EDGE)
    parts['depth1000_mld2'] = [('shelf/slope H<1000 m', shelf), ('deep, convective MLD2>400 m', conv),
                               ('deep, under sea ice', ice), ('deep, ice-free', ~shelf & ~conv & ~ice)]
    for L in COAST_KM:
        near = dist <= L
        conv = ~near & (mld_jja > MLD_CONV)
        ice = ~near & ~conv & (aice_jja >= ICE_EDGE)
        parts[f'coast{int(L)}'] = [(f'<= {int(L)} km of coast, JJA ice<0.7', near & (aice_jja < LOWICE)),
                                   (f'<= {int(L)} km of coast, JJA ice>=0.7', near & (aice_jja >= LOWICE)),
                                   ('offshore, convective MLD>400 m', conv),
                                   ('offshore, under sea ice', ice),
                                   ('offshore, ice-free', ~near & ~conv & ~ice)]
    for p, regs in parts.items():
        assert (sum(m.astype(int) for _, m in regs) == 1).all(), p

    Greg = {}
    for p, regs in parts.items():
        P('')
        P(f'--- partition {p} ---')
        for i, (name, msk) in enumerate(regs):
            m = msk & s60
            a = area[m].sum()
            P(f'R{i + 1} {name:38s} area {a / 1e12:6.3f}e12 m2 ({pct(a, area[s60].sum()):4.1f}%)'
              + (f', mean JJA ice {np.average(aice_jja[m], weights=area[m]):.2f}, '
                 f'mean JJA MLD {np.average(mld_jja[m], weights=area[m]):.0f} m' if a else ''))
        Greg[p] = {s: [G(m, s) for _, m in regs] for s in ['jja', 'ann']}
        Gcr = {s: [G(m, s, 'sig2c') for _, m in regs] for s in ['jja', 'ann']}
        for s in ['jja', 'ann']:
            gt = G(ALL, s)
            P(f'{s.upper()} transformation across isopycnal, Sv [share], 0.02 bins:')
            for n, k in thr:
                P(f'  {n:38s} {cf[k]:6.2f} total {gt[k]:6.2f} | ' +
                  ' '.join(f'R{i + 1} {c[k]:6.2f} [{pct(c[k], gt[k]):4.0f}%]' for i, c in enumerate(Greg[p][s])))
            P(f'  {"published peak bin (0.2 bin) " + f"{pk_lo:.1f}-{pk_hi:.1f}":45s} total {Gc[s][kc]:6.2f} | ' +
              ' '.join(f'R{i + 1} {c[kc]:6.2f} [{pct(c[kc], Gc[s][kc]):4.0f}%]' for i, c in enumerate(Gcr[s])))
            # formation by class range relative to the state's bottom water
            lo = ks['published peak bin, lower edge']
            rng = [('peak-bin lower edge -> bottom p10', lo, ks['bottom water p10']),
                   ('bottom p10 -> bottom p90', ks['bottom water p10'], ks['bottom water p90']),
                   ('denser than bottom p90', ks['bottom water p90'], None)]
            for n, a_, b_ in rng:
                if b_ is not None and b_ < a_:
                    P(f'  formation {n}: empty range (bottom-water density lighter than the peak bin)')
                    continue
                vals = [c[a_] - (c[b_] if b_ is not None else 0) for c in Greg[p][s]]
                tt = gt[a_] - (gt[b_] if b_ is not None else 0)
                P(f'  formation {n:34s} total {tt:6.2f} | ' +
                  ' '.join(f'R{i + 1} {v:6.2f}' for i, v in enumerate(vals)))
        if p in ('depth1000', 'coast200'):
            key[p] = {'names': [n for n, _ in regs],
                      'area': [float(area[m & s60].sum()) for _, m in regs],
                      'peak_bin_jja': [float(c[kc]) for c in Gcr['jja']],
                      'peak_bin_ann': [float(c[kc]) for c in Gcr['ann']],
                      'sig10_jja': [float(c[k10]) for c in Greg[p]['jja']],
                      'sig10_ann': [float(c[k10]) for c in Greg[p]['ann']],
                      'botp90_jja': [float(c[ks['bottom water p90']]) for c in Greg[p]['jja']],
                      'botmed_jja': [float(c[ks['bottom water median']]) for c in Greg[p]['jja']]}

    # --------------------------------------------- components, primary partition
    P('')
    P('--- heat / sea-ice FW / other FW parts, partition depth1000, JJA (Sv) ---')
    for n, b, k in [(f'published peak bin {pk_lo:.1f}-{pk_hi:.1f}', 'sig2c', kc),
                    (f'sigma_10 = {cf[k10]:.2f}', 'sig2f', k10)]:
        row = f'  {n}: '
        for name, msk in parts['depth1000']:
            h, i_, o_ = (G(msk, 'jja', b, (c,))[k] for c in ('heat', 'ice', 'other'))
            row += f'| {name}: heat {h:.2f} ice {i_:.2f} other {o_:.2f} '
        P(row)

    # --------------------------------------------- sectors
    P('')
    P('--- sectors (paper boxes; its Ross box split at 140W) x regions of partition depth1000, JJA (Sv) ---')
    sec = {}
    for sname, l0, l1 in SECTORS:
        inb = (lon >= l0) & (lon < l1)
        r = {'peak': [float(G(inb & m, 'jja', 'sig2c')[kc]) for _, m in parts['depth1000']],
             'sig10': [float(G(inb & m)[k10]) for _, m in parts['depth1000']],
             'area': [float(area[inb & m & s60].sum()) for _, m in parts['depth1000']]}
        sec[sname] = r
        P(f'  {sname:34s} peak bin: ' + ' '.join(f'R{i + 1} {v:6.2f}' for i, v in enumerate(r['peak'])) +
          ' | sigma_10: ' + ' '.join(f'R{i + 1} {v:5.2f}' for i, v in enumerate(r['sig10'])) +
          ' | area 1e12 m2: ' + ' '.join(f'{a / 1e12:.2f}' for a in r['area']))
    free = parts['depth1000'][3][1]
    f65 = free & (lat <= -65)
    P(f'  deep ice-free water south of 65S (open water near the continent in winter): area {area[f65 & s60].sum() / 1e12:.3f}e12 m2; '
      f'JJA G at peak bin {G(f65, "jja", "sig2c")[kc]:.2f} Sv, across sigma_10 {G(f65)[k10]:.2f} Sv, '
      f'across bottom median {G(f65)[ks["bottom water median"]]:.2f} Sv')
    key['free65'] = [float(area[f65 & s60].sum()), float(G(f65, 'jja', 'sig2c')[kc]), float(G(f65)[k10]),
                     float(G(f65)[ks['bottom water median']])]
    key['sectors'] = sec
    # densest surface spots
    top = np.argsort(-np.where(s60, sig_jja, -np.inf))[:5]
    P('  five nodes with the densest JJA-mean surface water: ' + '; '.join(
        f'{lat[i]:.1f}N {lon[i]:.1f}E sigma2 {sig_jja[i]:.2f} H {zb[i]:.0f} m' for i in top))
    hs = s60 & (sig_jja >= bq[2] + 0.5)
    P(f'  nodes whose JJA-mean surface sigma2 exceeds bottom p90 by >0.5: {hs.sum()}, area {area[hs].sum():.2e} m2; '
      f'their share of G(sigma_10) JJA {pct(G(hs)[k10], gf[k10]):.0f}%')
    key['hypersaline_share_sig10'] = float(pct(G(hs)[k10], gf[k10]))

    # --------------------------------------------- sigma0 stand-in for gamma_n
    P('')
    g0 = G(ALL, 'jja', 'sig0f')
    g0[[0, -1]] = 0
    k0 = int(np.argmax(g0))
    c0 = cen['sig0f']
    P(f'--- sigma0 coordinate (alpha, beta at p=0; a stand-in, NOT gamma_n), JJA, 0.02 bins ---')
    P(f'  sigma0 peak {g0[k0]:.2f} Sv at {c0[k0]:.2f}')
    key['sig0_peak'] = [float(c0[k0]), float(g0[k0])]
    for p in ['depth1000', 'coast200']:
        curves = [G(m, 'jja', 'sig0f') for _, m in parts[p]]
        for x in FIXED_S0:
            k = kbin(E['sig0f'], x + 1e-6)
            P(f'  {p:9s} sigma0 {c0[k]:.2f}: total {g0[k]:6.2f} | ' +
              ' '.join(f'R{i + 1} {c[k]:6.2f} [{pct(c[k], g0[k]):4.0f}%]' for i, c in enumerate(curves)))
            key[f'sig0_{x}_{p}'] = [float(g0[k])] + [float(c[k]) for c in curves]

    # --------------------------------------------- convection area check
    P('')
    P(f'JJA MLD>400 m area south of 60S: MLD1 {area[s60 & (mld_jja > 400)].sum() / 1e12:.3f}e12 m2, '
      f'MLD2 {area[s60 & (mld2_jja > 400)].sum() / 1e12:.3f}e12 m2; MLD1 JJA max {mld_jja[s60].max():.0f} m, '
      f'area-weighted p99 {wq(mld_jja[s60], area[s60], 0.99):.0f} m')

    # --------------------------------------------- save curves and map fields
    o = nc.Dataset(f'{HERE}/data/G_{exp}.nc', 'w')
    o.createDimension('binf', cf.size)
    o.createDimension('binc', cen['sig2c'].size)
    o.createVariable('sigma2_fine', 'f8', ('binf',))[:] = cf
    o.createVariable('sigma2_coarse', 'f8', ('binc',))[:] = cen['sig2c']
    o.createDimension('bin0', c0.size)
    o.createVariable('sigma0_fine', 'f8', ('bin0',))[:] = c0
    o.createVariable('G_total_jja_sig0', 'f8', ('bin0',))[:] = g0
    for s in ['jja', 'ann']:
        o.createVariable(f'G_total_{s}', 'f8', ('binf',))[:] = G(ALL, s)
        o.createVariable(f'G_total_coarse_{s}', 'f8', ('binc',))[:] = Gc[s]
    for p, cur in Greg.items():
        o.createDimension(f'reg_{p}', len(parts[p]))
        for s in ['jja', 'ann']:
            o.createVariable(f'G_{p}_{s}', 'f8', (f'reg_{p}', 'binf'))[:] = np.array(cur[s])
        setattr(o, f'regions_{p}', ' | '.join(n for n, _ in parts[p]))
    o.k_peak_coarse, o.k_peak_fine, o.k_sig10, o.k_sig50 = kc, kf, int(k10), int(k50)
    o.sig_bot_p10, o.sig_bot_med, o.sig_bot_p90 = bq
    o.close()

    r = nc.Dataset(f'{HERE}/data/regions_{exp}.nc', 'w')
    r.createDimension('node', sub.size)
    for name, v in [('node_index', sub), ('lon', lon), ('lat', lat), ('area', area), ('bottom_depth', zb),
                    ('coast_km', dist), ('aice_jja', aice_jja), ('mld1_jja', mld_jja),
                    ('sig2_bottom', sigb), ('sig2_surface_jja', sig_jja)]:
        r.createVariable(name, 'f8', ('node',))[:] = v
    for k, v in Fm.items():
        r.createVariable(k, 'f8', ('node',))[:] = v
    # JJA-mean density flux restricted to months/nodes in the class set (kg m-2 s-1)
    totc = sum(acc[f'acc_jja_sig2c_{c}'] for c in ('heat', 'ice', 'other')) / area[:, None]
    totf = sum(acc[f'acc_jja_sig2f_{c}'] for c in ('heat', 'ice', 'other')) / area[:, None]
    r.createVariable('F_jja_peakbin', 'f8', ('node',))[:] = totc[:, kc]
    r.createVariable('F_jja_dense_sig10', 'f8', ('node',))[:] = totf[:, k10:].sum(axis=1)
    r.createVariable('F_jja_dense_botp90', 'f8', ('node',))[:] = totf[:, ks['bottom water p90']:].sum(axis=1)
    r.peak_bin = f'{pk_lo:.1f}-{pk_hi:.1f}'
    r.sig10 = float(cf[k10])
    r.close()
    with open(f'{HERE}/data/key_{exp}.json', 'w') as fh:
        json.dump(key, fh, indent=1)
    txt = '\n'.join(out)
    print(txt)
    return txt, key


def summary(keys):
    L = []
    P = L.append
    P('Surface WMT south of 60S split by region and density class (fig_formation_location)')
    P('All numbers from analyze_formation_location.py; G = transformation across an isopycnal (Sv),')
    P('positive = towards higher density; G_r = part forced inside region r; shares in % of the total.')
    P('')
    P('1. Reproduction of the published curves (0.2 sigma2 bins, south of 60S)')
    for k in keys:
        a, b = k['repro_jja'], k['repro_ann']
        P(f'  {LABEL[k["exp"]]:4s} JJA peak published {a[0]:.3f}, here {a[1]:.3f}, max|diff| {a[2]:.1e} Sv; '
          f'annual peak {b[0]:.3f}, here {b[1]:.3f}, max|diff| {b[2]:.1e} Sv')
    P('')
    P('2. Per-state classes (sigma2)')
    for k in keys:
        P(f'  {LABEL[k["exp"]]:4s} peak bin {k["peak_bin"][0]:.1f}-{k["peak_bin"][1]:.1f} ({k["G_peak_bin"]:.1f} Sv); '
          f'0.02-bin peak {k["fine_peak"][1]:.1f} Sv at {k["fine_peak"][0]:.2f}; sigma_10 {k["sig10"]:.2f} '
          f'(G {k["G_sig10"]:.2f} Sv); bottom water p10/median/p90 {k["sig_bot"][0]:.2f}/{k["sig_bot"][1]:.2f}/'
          f'{k["sig_bot"][2]:.2f}; surface p99 {k["surf_p99"]:.2f}; sigma0 peak {k["sig0_peak"][1]:.1f} Sv at '
          f'{k["sig0_peak"][0]:.2f}')
    for p, title in [('depth1000', 'shelf/slope = bottom depth < 1000 m'),
                     ('coast200', 'near coast = within 200 km of the Antarctic coastline')]:
        P('')
        P(f'3. Regional shares, partition {p} ({title})')
        names = keys[0][p]['names']
        P('  regions: ' + '; '.join(f'R{i + 1} {n}' for i, n in enumerate(names)))
        for metric, lab in [('peak_bin_jja', 'JJA, published peak bin'), ('sig10_jja', 'JJA, across sigma_10 (densest tail)'),
                            ('botmed_jja', 'JJA, across bottom-water median'),
                            ('botp90_jja', 'JJA, across bottom-water p90'),
                            ('peak_bin_ann', 'annual, published peak bin'), ('sig10_ann', 'annual, across sigma_10')]:
            P(f'  {lab}:')
            for k in keys:
                v = k[p][metric]
                t = sum(v)
                P(f'    {LABEL[k["exp"]]:4s} total {t:6.2f} | ' + ' '.join(
                    f'R{i + 1} {x:6.2f} [{pct(x, t):4.0f}%]' for i, x in enumerate(v)))
        P('  region area (1e12 m2): ' + ' | '.join(
            f'{LABEL[k["exp"]]} ' + ' '.join(f'{a / 1e12:.2f}' for a in k[p]['area']) for k in keys))
    P('')
    P('4. sigma0 (stand-in for neutral density, NOT gamma_n), JJA, fixed thresholds, partition coast200')
    for k in keys:
        for x in FIXED_S0:
            v = k[f'sig0_{x}_coast200']
            P(f'  {LABEL[k["exp"]]:4s} sigma0 {x}: total {v[0]:6.2f} | ' +
              ' '.join(f'R{i + 1} {y:6.2f}' for i, y in enumerate(v[1:])))
    P('')
    P('5. Deep ice-free water south of 65S (winter open water near the continent), JJA')
    for k in keys:
        a, pk, d10, bm = k['free65']
        P(f'  {LABEL[k["exp"]]:4s} area {a / 1e12:.2f}e12 m2; peak bin {pk:.2f} Sv ({pct(pk, k["G_peak_bin"]):.0f}%); '
          f'across sigma_10 {d10:.2f} Sv ({pct(d10, k["G_sig10"]):.0f}%); across bottom median {bm:.2f} Sv')
    P('')
    P('6. Hypersaline coastal spots (JJA-mean surface sigma2 > bottom p90 + 0.5): share of G(sigma_10), JJA')
    P('  ' + ', '.join(f'{LABEL[k["exp"]]} {k["hypersaline_share_sig10"]:.0f}%' for k in keys))
    return '\n'.join(L)


if __name__ == '__main__':
    exps = sys.argv[1:] or EXPS
    res = [main(e) for e in exps]
    if exps == EXPS:
        with open(f'{HERE}/numbers.txt', 'w') as fh:
            fh.write(summary([k for _, k in res]) + '\n\n')
            fh.write('\n\n'.join(t for t, _ in res) + '\n')
        print('wrote numbers.txt')
