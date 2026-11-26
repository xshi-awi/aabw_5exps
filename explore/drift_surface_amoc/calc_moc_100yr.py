#!/usr/bin/env python
"""Yearly global and Atlantic depth-space MOC over the full 100-year analysis window of one run.

Usage: calc_moc_100yr.py {pi|mh|lig|lgm|mis}   (run via slurm_moc_100yr.sh, one array task per run)

Same engine and settings as data_moc/calc_moc_5exps.py (climdiag moc_depth, run mesh,
annual mean of the 12 monthly w fields, integrate from north, 2-degree latitudes crossed),
extended from the last 20 years to all years of the window. Adds an Atlantic(+Arctic)
streamfunction with the node mask from make_atlantic_masks.py.
Writes moc_yearly_{exp}.nc (psi fields) and moc_indices_yearly_{exp}.csv. Does not touch data_moc/.
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
MESH = {"pi": "awicm2_final384", "mh": "awicm2_final384", "lig": "awicm2_final384",
        "lgm": "mesh_glac1d", "mis": "mesh_glac1d_38k"}
FIRST_YEAR = {"pi": 2300, "mh": 2300, "lig": 2300, "lgm": 2400, "mis": 2500}
LABELS = {"pi": "PI", "mh": "MH", "lig": "LIG", "lgm": "LGM", "mis": "MIS3"}
LATS = np.arange(-81.0, 90.0, 2.0)          # latitudes crossed, as in data_moc


def window_extreme(psi, lat, depth, lat_sel, dep_sel, sign):
    sub = psi[np.ix_(dep_sel, lat_sel)]
    k = np.nanargmax(sub) if sign == "max" else np.nanargmin(sub)
    i, j = np.unravel_index(k, sub.shape)
    return float(sub[i, j]), float(lat[lat_sel][j]), float(depth[dep_sel][i])


def indices(psi_g, psi_a, lat, depth):
    out = {}
    up_lat, up_dep = (lat > 20) & (lat < 70), (depth >= 500) & (depth <= 4000)
    # Methods 4.4 definitions on the global streamfunction (data_moc names AABW / NADW)
    for name, p, ls, ds, s in (
            ("glob_upper", psi_g, up_lat, up_dep, "max"),
            ("glob_abyssal", psi_g, lat < -30, depth >= 2000, "min"),
            ("atl_upper", psi_a, up_lat, up_dep, "max"),
            # user's standard AMOC (climdiag AMOC_NCL): Atlantic, 20-90N, 800-2000 m
            ("atl_amoc_ncl", psi_a, lat > 20, (depth >= 800) & (depth <= 2000), "max")):
        v, la, de = window_extreme(p, lat, depth, ls, ds, s)
        out.update({name: v, f"{name}_lat": la, f"{name}_depth": de})
    # Atlantic overturning at 26.5N is not on the 2-degree grid; 27N row (crossing 27N)
    j27 = int(np.argmin(np.abs(lat - 27.0)))
    col = psi_a[:, j27]
    sel = (depth >= 500) & (depth <= 4000)
    out["atl_27N_max"] = float(np.nanmax(col[sel]))
    return out


def main(exp):
    commit = subprocess.run(["git", "-C", str(CLIMDIAG), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    tag = f"climdiag {climdiag.__version__} ({commit})"
    mesh = load_fesom_mesh(f"{MESHES}/{MESH[exp]}/fesom.mesh.diag.nc")
    amask = np.load(HERE / f"atlantic_mask_{MESH[exp]}.npy")
    assert amask.size == mesh.sizes["nod2"], (amask.size, mesh.sizes)
    am = xr.DataArray(amask, dims=("nod2",))
    rows, fg, fa, years = [], [], [], []
    for y in range(FIRST_YEAR[exp], FIRST_YEAR[exp] + 100):
        p = PROD / f"{exp}_age/outdata/fesom/w.fesom.{y}01.01.nc"
        if not p.exists():
            print(exp, y, "MISSING", flush=True)
            continue
        with xr.open_dataset(p, decode_times=False) as ds:
            assert ds.sizes["time"] == 12, (p, ds.sizes)
            w = ds["w"].values.astype(np.float64).mean(0)            # (nod2, nz)
        wd = xr.DataArray(w, dims=("nod2", "nz"))
        pg = moc_depth(wd, mesh, basin="global", basin_mask=None, lats=LATS,
                       integrate_from="north")
        pa = moc_depth(wd, mesh, basin="atlantic", basin_mask=am, lats=LATS,
                       integrate_from="north")
        depth = pg["depth"].values
        ix = indices(pg.values, pa.values, LATS, depth)
        rows.append({"exp": LABELS[exp], "year": y, **ix})
        fg.append(pg.values)
        fa.append(pa.values)
        years.append(y)
        print(exp, y, " ".join(f"{k} {ix[k]:.2f}" for k in
                                ("glob_upper", "glob_abyssal", "atl_upper", "atl_amoc_ncl")),
              flush=True)
    ds = xr.Dataset({"psi_global": (("year", "depth", "lat"), np.stack(fg)),
                     "psi_atlantic": (("year", "depth", "lat"), np.stack(fa))},
                    coords={"year": years, "depth": depth, "lat": LATS})
    for v in ("psi_global", "psi_atlantic"):
        ds[v].attrs = {"units": "Sv", "lat_meaning": "latitude crossed (geographic)",
                       "depth_meaning": "interface depth, m, positive down"}
    ds.attrs = {"exp": LABELS[exp], "mesh": MESH[exp], "climdiag": tag,
                "atlantic_mask": f"atlantic_mask_{MESH[exp]}.npy ({int(amask.sum())} nodes)",
                "source": "production/{exp}_age/outdata/fesom/w.fesom.YYYY01.01.nc, annual means",
                "script": "explore/drift_surface_amoc/calc_moc_100yr.py"}
    ds.to_netcdf(HERE / f"moc_yearly_{exp}.nc")
    pd.DataFrame(rows).to_csv(HERE / f"moc_indices_yearly_{exp}.csv", index=False,
                              float_format="%.4f")
    print(exp, "years", len(years), tag, "MOC_EXP_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
