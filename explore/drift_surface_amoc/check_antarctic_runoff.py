#!/usr/bin/env python
"""Where and when the FESOM runoff south of 60S enters (12-month climatology {exp}/runoff_clim.nc, m/s).

Coastal nodes = nodes on mesh boundary edges (an edge that belongs to one element only).
Reports the share of the runoff south of 60S that falls on coastal nodes and on their
first neighbours, the share within 100/200/300 km of the nearest coastal node, the number of receiving nodes, negative runoff, and the monthly cycle.
Writes antarctic_runoff_check.txt.
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

sys.path.insert(0, "/work/ba1066/a270064/cc_projects/climdiag/src")
from climdiag.moc import load_fesom_mesh  # noqa: E402

ROOT = Path("/work/ba1066/a270064/cc_projects/aabw_5exps")
HERE = Path(__file__).resolve().parent
MESH = {"pi": "awicm2_final384", "mh": "awicm2_final384", "lig": "awicm2_final384",
        "lgm": "mesh_glac1d", "mis": "mesh_glac1d_38k"}
LAB = {"pi": "PI", "mh": "MH", "lig": "LIG", "lgm": "LGM", "mis": "MIS3"}


def main():
    out = []
    for exp in LAB:
        mesh = load_fesom_mesh(f"/home/a/a270064/bb1029/inputs/{MESH[exp]}/fesom.mesh.diag.nc")
        el = mesh["elem_nodes"].values
        n = mesh.sizes["nod2"]
        e = np.sort(np.concatenate([el[:, [0, 1]], el[:, [1, 2]], el[:, [2, 0]]]), axis=1)
        uniq, cnt = np.unique(e, axis=0, return_counts=True)
        coast = np.zeros(n, bool)
        coast[uniq[cnt == 1].ravel()] = True
        ring1 = coast.copy()
        for a, b in uniq:
            if coast[a] or coast[b]:
                ring1[a] = ring1[b] = True
        with xr.open_dataset(ROOT / exp / f"surface_density_tendency_100years_{exp}.nc") as d:
            area = d.areacello.values.astype(np.float64)
            lat = d.lat.values.astype(np.float64)
        assert area.size == n
        with xr.open_dataset(ROOT / exp / "runoff_clim.nc", decode_times=False) as r:
            ro = r.runoff.values.astype(np.float64)                    # (12, nod2), m/s
        s = lat < -60
        flux = ro[:, s] * area[s] / 1e6                                # Sv per node, monthly
        ann = flux.mean(0)
        tot = ann.sum()
        cs, r1s = coast[s], ring1[s]
        lo, la = np.radians(mesh["node_lon"].values), np.radians(mesh["node_lat"].values)
        xyz = np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
        chord, _ = cKDTree(xyz[coast]).query(xyz[s])
        dist_km = 2 * 6371.0 * np.arcsin(np.clip(chord / 2, 0, 1))
        within = " ".join(f"<{k} km {ann[dist_km < k].sum() / tot:.1%}" for k in (100, 200, 300))
        lines = [f"{LAB[exp]} ({MESH[exp]}): runoff south of 60S {tot:.4f} Sv (plain 12-month mean); "
                 f"receiving nodes {np.sum(ann > 0)} of {s.sum()}; "
                 f"on coastal nodes {ann[cs].sum() / tot:.1%}, coastal or first neighbour "
                 f"{ann[r1s].sum() / tot:.1%}; negative node values {np.sum(ann < 0)} "
                 f"(sum {ann[ann < 0].sum():.2e} Sv)",
                 f"  distance to coast: {within}; runoff-weighted mean "
                 f"{np.sum(ann * dist_km) / tot:.0f} km",
                 "  monthly Sv: " + " ".join(f"{v:.3f}" for v in flux.sum(1))]
        out += lines
        print("\n".join(lines), flush=True)
    (HERE / "antarctic_runoff_check.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
