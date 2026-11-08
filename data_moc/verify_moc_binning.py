#!/usr/bin/env python
"""Independent check of finding D6 (no climdiag code used here).

Builds the mesh from the ASCII files (nod2d.out, elem2d.out, elvls.out), un-rotates the
rotated awicm2_final384 coordinates with FESOM's Euler angles (50, 15, -90), computes the
element areas FESOM-style, and integrates w over elements in plain numpy. Then:

1. reproduces diag_gmoc.nc of pi/mh/lig_age with ROTATED-latitude binning and compares
   with GEOGRAPHIC binning (file row label L = transport across L - 2, fpost2 slip);
2. identifies which model years the last records of the lgm/mis files are;
3. southern-boundary fingerprint: no ocean lies south of 78.6S geographically, so the
   transport across 79S must vanish; it does not in a rotated-binned file;
4. Weddell and Ross sinking: the latitude band where it sits in each frame;
5. compares the geographic numpy MOC with climdiag's (calc_moc_5exps.py output).

Needs the 20-year mean w written by calc_moc_5exps.py to SCRATCH. Writes
verify_moc_binning.txt next to this script.
"""
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
PROD = Path("/work/ba1066/a270064/production")
MESHES = Path("/home/a/a270064/bb1029/inputs")
SCRATCH = Path("/scratch/a/a270064/aabw_moc")
R_EARTH = 6367500.0                         # FESOM2 r_earth
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


def euler_matrix(al=50.0, be=15.0, ga=-90.0):
    al, be, ga = np.radians([al, be, ga])
    return np.array([
        [np.cos(ga) * np.cos(al) - np.sin(ga) * np.cos(be) * np.sin(al),
         np.cos(ga) * np.sin(al) + np.sin(ga) * np.cos(be) * np.cos(al), np.sin(ga) * np.sin(be)],
        [-np.sin(ga) * np.cos(al) - np.cos(ga) * np.cos(be) * np.sin(al),
         -np.sin(ga) * np.sin(al) + np.cos(ga) * np.cos(be) * np.cos(al), np.cos(ga) * np.sin(be)],
        [np.sin(be) * np.sin(al), -np.sin(be) * np.cos(al), np.cos(be)]])


def rotate(lon, lat, mat):
    lo, la = np.radians(lon), np.radians(lat)
    x = mat @ np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    return np.degrees(np.arctan2(x[1], x[0])), np.degrees(np.arcsin(np.clip(x[2], -1, 1)))


def ascii_mesh(name, stored):
    """stored = 'rotated' (awicm2_final384) or 'geographic' (glac1d meshes)."""
    d = MESHES / name
    nod = np.loadtxt(d / "nod2d.out", skiprows=1)
    elem = np.loadtxt(d / "elem2d.out", skiprows=1, dtype=np.int64) - 1
    nlev = np.loadtxt(d / "elvls.out", dtype=np.int64)
    assert elem.shape[0] == nlev.size
    R = euler_matrix()
    if stored == "rotated":
        rlon, rlat = nod[:, 1], nod[:, 2]
        glon, glat = rotate(rlon, rlat, R.T)           # rotated -> geographic
    else:
        glon, glat = nod[:, 1], nod[:, 2]
        rlon, rlat = rotate(glon, glat, R)
    return dict(glon=glon, glat=glat, rlon=rlon, rlat=rlat, elem=elem, nlev=nlev)


def planar_area(lon, lat, elem):
    """FESOM2 elem_area: planar triangle with the cos of the mean latitude."""
    lo, la = np.radians(lon[elem]), np.radians(lat[elem])
    dlo = (lo[:, 1:] - lo[:, :1] + np.pi) % (2 * np.pi) - np.pi
    dla = la[:, 1:] - la[:, :1]
    c = np.cos(la.mean(1))
    return 0.5 * np.abs(dlo[:, 0] * c * dla[:, 1] - dlo[:, 1] * c * dla[:, 0]) * R_EARTH ** 2


def psi_north(w, m, elat, area, cross, node_mask=None):
    """psi(phi) = -1e-6 sum over elements with elat >= phi of A wbar (Sv), shape (nz, ncross)."""
    wm = w if node_mask is None else w * node_mask[:, None]
    e = m["elem"]
    te = area[:, None] * (wm[e[:, 0]] + wm[e[:, 1]] + wm[e[:, 2]]) / 3.0
    te[np.arange(w.shape[1])[None, :] >= (m["nlev"][:, None] - 1)] = 0.0
    b = np.searchsorted(cross, elat, side="right")     # elements with elat >= cross[j]: b > j
    nb = cross.size + 1
    B = np.stack([np.bincount(b, weights=te[:, k], minlength=nb) for k in range(w.shape[1])], 1)
    tail = np.cumsum(B[::-1], 0)[::-1]
    return -1e-6 * tail[1:].T


def year_w(exp, y):
    with xr.open_dataset(PROD / f"{exp}_age/outdata/fesom/w.fesom.{y}01.01.nc",
                         decode_times=False) as ds:
        return ds["w"].values.astype(np.float64).mean(0)


def gmoc(exp):
    f = netCDF4.Dataset(PROD / f"{exp}_age/diag_gmoc.nc")
    return np.asarray(f["MOC"][:], dtype=float), np.asarray(f["lats"][:], dtype=float)


def main():
    core = ascii_mesh("awicm2_final384", "rotated")
    ref = np.loadtxt(MESHES / "mesh_core2/nod2d.out", skiprows=1)
    dlon = (core["glon"] - ref[:, 1] + 180) % 360 - 180
    say("== mesh: awicm2_final384 nod2d.out un-rotated (50,15,-90) vs mesh_core2 nod2d.out:")
    say(f"   max |dlat| {np.abs(core['glat'] - ref[:, 2]).max():.2e} deg, "
        f"max |dlon cos| {np.abs(dlon * np.cos(np.radians(ref[:, 2]))).max():.2e} deg")
    say(f"   rotated - geographic latitude: {np.min(core['rlat'] - core['glat']):.1f} .. "
        f"{np.max(core['rlat'] - core['glat']):.1f} deg; southernmost node geo "
        f"{core['glat'].min():.2f}, rotated {core['rlat'].min():.2f}")
    el_core2 = np.loadtxt(MESHES / "mesh_core2/elvls.out", dtype=np.int64)
    say(f"   elvls.out differs from mesh_core2 for {(core['nlev'] != el_core2).sum()} of "
        f"{core['nlev'].size} elements (max {np.abs(core['nlev'] - el_core2).max()} levels)")
    area = planar_area(core["rlon"], core["rlat"], core["elem"])     # model frame
    diag_area = np.asarray(netCDF4.Dataset(MESHES / "awicm2_final384/fesom.mesh.diag.nc")
                           ["elem_area"][:], dtype=float)
    say(f"   numpy elem area vs fesom.mesh.diag.nc: max rel diff "
        f"{np.max(np.abs(area - diag_area) / diag_area):.1e}")
    e = core["elem"]
    lat_rot = core["rlat"][e].mean(1)
    lat_geo = core["glat"][e].mean(1)

    say("\n== 1. diag_gmoc.nc reproduced by rotated vs geographic binning (row L = across L-2)")
    win = {"pi": range(2380, 2400), "mh": range(2380, 2400)}
    geo_fields = {}
    for exp in ["pi", "mh", "lig"]:
        moc, lab = gmoc(exp)
        cross = lab[1:] - 2.0
        if exp in win:
            w = np.load(SCRATCH / f"wmean_{exp}_2380_2399.npy")
            target = moc[-20:].mean(0)[:, 1:]
            what = "mean of last 20 records vs w 2380-2399"
        else:   # lig file: records 2330-2371 (stops at missing 2372); check its last record
            w = year_w(exp, 2371)
            target = moc[-1][:, 1:]
            what = "last record vs w 2371"
        pr = psi_north(w, core, lat_rot, area, cross)
        pg = psi_north(w, core, lat_geo, area, cross)
        say(f"   {exp} ({what}): max|file - rotated| {np.abs(target - pr).max():.4f} Sv; "
            f"max|file - geographic| {np.abs(target - pg).max():.2f} Sv")
        if exp in win:
            geo_fields[exp] = (w, psi_north(w, core, lat_geo, area, np.arange(-79.0, 90.0, 2.0)))

    say("\n== 2. which years the last records of the glacial files are")
    # candidates found by trial (2026-11-08): lgm 2499, 2399, 2449, 2450 match no record
    # (6-8 Sv off); 2350 matches. mis 2616 matches none; 2599 matches.
    for exp, mesh, ys in [("lgm", "mesh_glac1d", [2350, 2499]), ("mis", "mesh_glac1d_38k", [2599])]:
        g = ascii_mesh(mesh, "geographic")
        nod = np.asarray(netCDF4.Dataset(MESHES / mesh / "fesom.mesh.diag.nc")["nodes"][:])
        say(f"   {mesh}: nod2d.out lat vs diag nodes lat max diff "
            f"{np.abs(g['glat'] - np.degrees(nod[1])).max():.1e} deg (geographic storage)")
        ga = planar_area(g["glon"], g["glat"], g["elem"])
        moc, lab = gmoc(exp)
        cross = lab[1:] - 2.0
        for y in ys:
            p = psi_north(year_w(exp, y), g, g["glat"][g["elem"]].mean(1), ga, cross)
            d = [np.abs(moc[r][:, 1:] - p).max() for r in range(moc.shape[0])]
            r = int(np.argmin(d))
            say(f"   {exp} year {y}: best record {r} of {moc.shape[0]} (max diff {d[r]:.3f} Sv, "
                f"next best {sorted(d)[1]:.2f}) -> file years {y - r}-{y - r + moc.shape[0] - 1}")

    say("\n== 3. southern boundary: transport across 79S (file row labelled -77), max over depth")
    for exp in ["pi", "mh", "lig", "lgm", "mis"]:
        moc, lab = gmoc(exp)
        j = int(np.where(lab == -77)[0][0])
        say(f"   {exp} file: {np.abs(moc[-20:].mean(0)[:, j]).max():.2f} Sv")
    for exp, (w, pg) in geo_fields.items():
        say(f"   {exp} geographic numpy: {np.abs(pg[:, 0]).max():.2e} Sv (crossing -79)")

    say("\n== 4. where Weddell and Ross sinking sits: net downward flux across 1040 m per 1-deg"
        " band, PI 2380-2399")
    w = np.load(SCRATCH / "wmean_pi_2380_2399.npy")
    depth = np.abs(np.loadtxt(MESHES / "awicm2_final384/aux3d.out", skiprows=1, max_rows=48))
    k = int(np.where(depth == 1040)[0][0])
    we = (w[e[:, 0], k] + w[e[:, 1], k] + w[e[:, 2], k]) / 3 * area * (k < core["nlev"] - 1)
    gl = core["glon"][e]
    sectors = {"Weddell (60W-30E)": ((gl >= -60) & (gl <= 30)).all(1),
               "Ross (160E-150W)": ((gl >= 160) | (gl <= -150)).all(1)}
    bands = np.arange(-86.0, -49.0, 1.0)
    for name, sec in sectors.items():
        for frame, el in (("geographic", lat_geo), ("rotated", lat_rot)):
            sb = np.array([we[sec & (el >= b) & (el < b + 1)].sum() * 1e-6 for b in bands])
            j = int(np.argmin(sb))
            say(f"   {name:18s} {frame:10s}: strongest sinking in {bands[j]:.0f} to "
                f"{bands[j] + 1:.0f} deg ({sb[j]:.2f} Sv); southernmost element {el[sec].min():.1f}")
    shelf = (core["glat"] < -70) & (core["glon"] > -60) & (core["glon"] < -20)
    say(f"   Weddell shelf nodes (70-80S, 60-20W): geographic {core['glat'][shelf].mean():.1f}, "
        f"rotated {core['rlat'][shelf].mean():.1f} deg")

    say("\n== 5. this numpy MOC vs climdiag (moc_5exps_geo.nc), geographic, PI and MH")
    cd = xr.open_dataset(HERE / "moc_5exps_geo.nc")
    for exp, lab in (("pi", "PI"), ("mh", "MH")):
        pg = geo_fields[exp][1]
        c = cd["moc"].sel(exp=lab).sel(lat=np.arange(-79.0, 90.0, 2.0)).values
        say(f"   {exp}: max|numpy - climdiag| {np.abs(pg - c).max():.4f} Sv "
            f"(numpy uses its own planar areas and ASCII levels)")
    (HERE / "verify_moc_binning.txt").write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
