#!/usr/bin/env python
"""Global depth-space MOC of the five climate states from the raw FESOM2 vertical velocity,
binned by GEOGRAPHIC latitude (climdiag.moc_depth), with one window rule for all five.

Why not diag_gmoc.nc: for pi/mh/lig_age that file was binned by ROTATED latitude
(fpost2 rotated_grid=.false. on the rotated awicm2_final384 mesh; finding D6, see
CLAUDE.md here). Window rule: the last 20 model years of the 100-year period the WMT
figures use (PI/MH/LIG 2380-2399, LGM 2480-2499, MIS3 2580-2599).

Writes moc_5exps_geo.nc (mean psi, lat = latitude crossed) and moc_indices.csv, and the
20-year mean w per run to SCRATCH for verify_moc_binning.py. Run on a compute node
(slurm_moc.sh); about 100 yearly w files of 292 MB.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

CLIMDIAG = Path("/work/ba1066/a270064/cc_projects/climdiag")
sys.path.insert(0, str(CLIMDIAG / "src"))
import climdiag  # noqa: E402
from climdiag.moc import load_fesom_mesh, moc_depth  # noqa: E402

HERE = Path(__file__).resolve().parent
PROD = Path("/work/ba1066/a270064/production")
MESHES = "/home/a/a270064/bb1029/inputs"
SCRATCH = Path("/scratch/a/a270064/aabw_moc")
# mesh each run actually used (meshpath in production/{exp}_age/config/fesom/namelist.config_*)
MESH = {"pi": "awicm2_final384", "mh": "awicm2_final384", "lig": "awicm2_final384",
        "lgm": "mesh_glac1d", "mis": "mesh_glac1d_38k"}
LAST_YEAR = {"pi": 2399, "mh": 2399, "lig": 2399, "lgm": 2499, "mis": 2599}
NYEARS = 20
LABELS = {"pi": "PI", "mh": "MH", "lig": "LIG", "lgm": "LGM", "mis": "MIS3"}
# latitudes crossed; -81 only serves the fpost2-row variant below (row L = crossing L-2)
LATS = np.arange(-81.0, 90.0, 2.0)


def indices(psi, lat, depth):
    """The two indices of plot_moc_5exps.py: AABW = min over depth >= 2000 m and lat < -30;
    NADW = max over 500-4000 m and 20 < lat < 70 (0-70N until 2026-11-08, D8). Also the
    reference NADW_below1000 = max over 1000-4000 m, 20-70N. Returns values and where they sit."""
    ka, ja = depth >= 2000, lat < -30
    sub = psi[np.ix_(ka, ja)]
    i, j = np.unravel_index(np.nanargmin(sub), sub.shape)
    out = dict(AABW=float(sub[i, j]), AABW_lat=float(lat[ja][j]), AABW_depth=float(depth[ka][i]))
    jn = (lat > 20) & (lat < 70)
    for name, kn in (("NADW", (depth >= 500) & (depth <= 4000)),
                     ("NADW_below1000", (depth >= 1000) & (depth <= 4000))):
        subn = psi[np.ix_(kn, jn)]
        p, q = np.unravel_index(np.nanargmax(subn), subn.shape)
        out.update({name: float(subn[p, q]), f"{name}_lat": float(lat[jn][q]),
                    f"{name}_depth": float(depth[kn][p])})
    return out


def main():
    commit = subprocess.run(["git", "-C", str(CLIMDIAG), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    tag = f"climdiag {climdiag.__version__} ({commit})"
    SCRATCH.mkdir(parents=True, exist_ok=True)
    fields, rows, yearly = {}, [], []
    for exp in ["pi", "mh", "lig", "lgm", "mis"]:
        mesh = load_fesom_mesh(f"{MESHES}/{MESH[exp]}/fesom.mesh.diag.nc")
        years = list(range(LAST_YEAR[exp] - NYEARS + 1, LAST_YEAR[exp] + 1))
        acc, per_year = None, []
        for y in years:
            p = PROD / f"{exp}_age/outdata/fesom/w.fesom.{y}01.01.nc"
            with xr.open_dataset(p, decode_times=False) as ds:
                assert ds.sizes["time"] == 12, (p, ds.sizes)
                w = ds["w"].values.astype(np.float64).mean(0)            # (nod2, nz)
            acc = w if acc is None else acc + w
            psi_y = moc_depth(xr.DataArray(w, dims=("nod2", "nz")), mesh, basin="global",
                              basin_mask=None, lats=LATS, integrate_from="north")
            ix = indices(psi_y.values, LATS, psi_y["depth"].values)
            yearly.append({"exp": LABELS[exp], "year": y, **ix})
            print(exp, y, f"AABW {ix['AABW']:.2f} NADW {ix['NADW']:.2f}", flush=True)
        wbar = acc / len(years)
        np.save(SCRATCH / f"wmean_{exp}_{years[0]}_{years[-1]}.npy", wbar)
        psi = moc_depth(xr.DataArray(wbar, dims=("nod2", "nz")), mesh, basin="global",
                        basin_mask=None, lats=LATS, integrate_from="north")
        fields[exp] = psi
        depth = psi["depth"].values
        true = indices(psi.values[:, 1:], LATS[1:], depth)                # lat = crossing
        same = indices(psi.values[:, :-1], LATS[:-1] + 2.0, depth)        # fpost2 row labels
        yr = pd.DataFrame([r for r in yearly if r["exp"] == LABELS[exp]])
        rows.append({"exp": LABELS[exp], "years": f"{years[0]}-{years[-1]}", "mesh": MESH[exp],
                     **true, "AABW_sd_yearly": yr["AABW"].std(ddof=1),
                     "NADW_sd_yearly": yr["NADW"].std(ddof=1),
                     "AABW_fpost2_rows": same["AABW"], "NADW_fpost2_rows": same["NADW"]})
        print(rows[-1], flush=True)

    depth = fields["pi"]["depth"].values
    ds = xr.Dataset(
        {"moc": (("exp", "depth", "lat"), np.stack([fields[e].values[:, 1:] for e in fields]))},
        coords={"exp": [LABELS[e] for e in fields], "depth": depth, "lat": LATS[1:]})
    ds["moc"].attrs = {"units": "Sv", "long_name": "global overturning streamfunction",
                       "lat_meaning": "latitude the transport crosses (geographic)",
                       "depth_meaning": "interface depth, m, positive down"}
    ds.attrs = {"source": "raw w.fesom.YYYY01.01.nc, annual means, then mean over the window",
                "windows": "; ".join(f"{r['exp']} {r['years']}" for r in rows),
                "meshes": "; ".join(f"{LABELS[e]} {MESH[e]}" for e in fields),
                "climdiag": tag, "script": "data_moc/calc_moc_5exps.py"}
    ds.to_netcdf(HERE / "moc_5exps_geo.nc")
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "moc_indices.csv", index=False, float_format="%.3f")
    pd.DataFrame(yearly).to_csv(HERE / "moc_indices_yearly.csv", index=False, float_format="%.3f")
    print(out.to_string())
    print(tag)


if __name__ == "__main__":
    main()
