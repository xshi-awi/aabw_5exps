#!/usr/bin/env python
"""Atlantic(+Arctic) node masks for the three run meshes, for climdiag.moc_depth(basin="atlantic").

Geometry = the user's ~/ab0246/fpost2/create_mask.ncl boxes, applied to GEOGRAPHIC node
coordinates from fesom.mesh.diag.nc (awicm2_final384/nod2d.out is rotated):
  lat>=66 all lon; 30-66N lon -115..60; 16-30N -100..0; 8-16N -84..30; 32S-8N -70..30.
Then only the nodes connected (through mesh elements, staying inside the boxes) to a seed
at 30N 40W are kept, so Pacific pockets inside the boxes (Gulf of Panama, Gulf of
California, Pacific coast of Mexico) drop out. Compared with the user's fpost2 masks.
Writes atlantic_mask_{mesh}.npy (bool per node) and atlantic_masks_check.txt.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, "/work/ba1066/a270064/cc_projects/climdiag/src")
from climdiag.moc import load_fesom_mesh  # noqa: E402

HERE = Path(__file__).resolve().parent
MESHES = "/home/a/a270064/bb1029/inputs"
USER_MASK = {"awicm2_final384": "mask_NA_pi.dat", "mesh_glac1d": "mask_NA_lgm.dat",
             "mesh_glac1d_38k": "mask_NA_38k.dat"}


def boxes(lon, lat):
    return ((lat >= 66)
            | ((lat >= 30) & (lat < 66) & (lon > -115) & (lon < 60))
            | ((lat >= 16) & (lat < 30) & (lon > -100) & (lon < 0))
            | ((lat >= 8) & (lat < 16) & (lon > -84) & (lon < 30))
            | ((lat >= -32) & (lat < 8) & (lon > -70) & (lon < 30)))


def main():
    out = []
    for name in USER_MASK:
        mesh = load_fesom_mesh(f"{MESHES}/{name}/fesom.mesh.diag.nc")
        lon, lat = mesh["node_lon"].values, mesh["node_lat"].values
        lon = np.where(lon > 180, lon - 360, lon)
        el = mesh["elem_nodes"].values
        n = lon.size
        inbox = boxes(lon, lat)
        # graph of element edges with both ends inside the boxes
        i = np.concatenate([el[:, 0], el[:, 1], el[:, 2]])
        j = np.concatenate([el[:, 1], el[:, 2], el[:, 0]])
        keep = inbox[i] & inbox[j]
        g = coo_matrix((np.ones(keep.sum()), (i[keep], j[keep])), shape=(n, n))
        ncomp, lab = connected_components(g, directed=False)
        seed = np.argmin(np.where(inbox, (lat - 30) ** 2 + (lon + 40) ** 2, np.inf))
        mask = inbox & (lab == lab[seed])
        np.save(HERE / f"atlantic_mask_{name}.npy", mask)
        dropped = inbox & ~mask
        lines = [f"{name}: nodes {n}, in boxes {inbox.sum()}, connected Atlantic {mask.sum()}, "
                 f"dropped {dropped.sum()}"]
        if dropped.any():
            for lo, hi in ((-40, 8), (8, 16), (16, 30), (30, 66), (66, 91)):
                s = dropped & (lat >= lo) & (lat < hi)
                if s.any():
                    lines.append(f"  dropped {s.sum():5d} nodes at {lo}..{hi}N, lon "
                                 f"{lon[s].min():.1f}..{lon[s].max():.1f}")
        um = Path.home() / "ab0246/fpost2" / USER_MASK[name]
        idx = np.loadtxt(um, dtype=np.int64)[1:] - 1
        user = np.zeros(n, bool)
        user[idx] = True
        lines.append(f"  user mask {um.name}: {user.sum()} nodes; equal to boxes rule: "
                     f"{np.array_equal(user, inbox)}; user-not-ours {np.sum(user & ~mask)}, "
                     f"ours-not-user {np.sum(mask & ~user)}")
        out += lines
        print("\n".join(lines), flush=True)
    (HERE / "atlantic_masks_check.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
