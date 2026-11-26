#!/usr/bin/env python
"""Yearly surface series of the five runs over their 100-year analysis windows.

(1) Global-mean and Southern Ocean (south of 50S) SST from monthly `tos` (FESOM nodes,
    weighted by `areacello`) in {exp}/surface_density_tendency_100years_{exp}.nc.
    Annual mean = month-length-weighted mean of the 12 months of a calendar year.
(2) Freshwater fluxes integrated south of 60S, in Sv (1 Sv = 1e9 kg/s of water):
    friver (FESOM `runoff`, i.e. the HD-model discharge received from ECHAM), prlq, prsn,
    evs, fsitherm, wfo (net), and the iceberg term of the mass budget.
(3) For comparison, ECHAM precipitation minus evaporation over Antarctic glacier land
    (south of 60S) from {exp}/echam_clim.nc (12-month climatology).

Run via slurm_surface.sh (python3/2022.01-gcc-11.2.0, `python3 -s`).
Writes surface_yearly_{exp}.csv, surface_fw_mean.csv, echam_antarctic_pme.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path("/work/ba1066/a270064/cc_projects/aabw_5exps")
HERE = Path(__file__).resolve().parent
EXPS = {"pi": "PI", "mh": "MH", "lig": "LIG", "lgm": "LGM", "mis": "MIS3"}
FW = ["friver", "prlq", "prsn", "evs", "fsitherm", "wfo",
      "mass_rhs_sum_surface_exchange_flux_sum_rivers",
      "mass_rhs_sum_surface_exchange_flux_sum_icebergs"]
R_EARTH = 6.371e6


def main():
    fw_rows, all_years = [], {}
    for exp, lab in EXPS.items():
        f = ROOT / exp / f"surface_density_tendency_100years_{exp}.nc"
        ds = xr.open_dataset(f)
        yr = ds.time.dt.year.values
        dim = ds.time.dt.days_in_month.values.astype(np.float64)
        years = np.unique(yr)
        for y in years:
            assert (yr == y).sum() == 12, (exp, y)
        area = ds.areacello.values.astype(np.float64)
        lat = ds.lat.values.astype(np.float64)
        so50, so60 = lat < -50, lat < -60
        out = {"year": years}

        def annual(series):                         # (time,) -> (year,)
            return np.array([np.sum(series[yr == y] * dim[yr == y]) / np.sum(dim[yr == y])
                             for y in years])

        tos = ds.tos.values.astype(np.float64)                         # (time, nod2)
        assert np.isfinite(tos).all(), "NaN in tos"
        out["sst_global"] = annual(tos @ area / area.sum())
        out["sst_so50"] = annual(tos[:, so50] @ area[so50] / area[so50].sum())
        del tos
        for v in FW:
            x = ds[v].values.astype(np.float64)
            assert np.isfinite(x).all(), (exp, v)
            out[f"{v}_s60_Sv"] = annual(x[:, so60] @ area[so60] / 1e9)   # kg/s -> Sv
            del x
        df = pd.DataFrame(out)
        df.insert(0, "exp", lab)
        df.to_csv(HERE / f"surface_yearly_{exp}.csv", index=False, float_format="%.6g")
        all_years[lab] = df
        row = {"exp": lab, "years": f"{years[0]}-{years[-1]}", "n_years": len(years),
               "area_s60_km2": area[so60].sum() / 1e6}
        for v in FW:
            row[f"{v}_s60_Sv"] = df[f"{v}_s60_Sv"].mean()
        fw_rows.append(row)
        print(lab, len(years), years[0], years[-1],
              "SST %.3f SO50 %.3f" % (df.sst_global.mean(), df.sst_so50.mean()),
              " ".join(f"{v.split('_')[-1]} {row[f'{v}_s60_Sv']:.4f}" for v in FW), flush=True)
        ds.close()

        # cross-check: friver should be 1000 * FESOM runoff (m/s); runoff_clim.nc (12 months)
        rc = ROOT / exp / "runoff_clim.nc"
        if rc.exists():
            with xr.open_dataset(rc, decode_times=False) as r:
                ro = r.runoff.values.astype(np.float64).mean(0)          # m/s, plain mean
            print(f"  check {lab}: runoff_clim south of 60S {ro[so60] @ area[so60] / 1e6:.4f} Sv "
                  f"(plain 12-month mean)", flush=True)
    pd.DataFrame(fw_rows).to_csv(HERE / "surface_fw_mean.csv", index=False, float_format="%.5g")

    # ECHAM P-E over Antarctic glacier land (approximate: grid-box P and E times land and glacier fraction)
    rows = []
    for exp, lab in EXPS.items():
        with xr.open_dataset(ROOT / exp / "echam_clim.nc", decode_times=False) as e:
            lat = e.lat.values
            nlon = e.sizes["lon"]
            p = (e.var142 + e.var143).values.astype(np.float64)          # kg m-2 s-1
            ev = e.var182.values.astype(np.float64)                      # kg m-2 s-1, negative up
            slf = e.var194.values.astype(np.float64)                     # land fraction
            glac = e.var232.values.astype(np.float64)                    # glacier fraction
        x, w = np.polynomial.legendre.leggauss(lat.size)
        gl = np.degrees(np.arcsin(x))[::-1] if lat[0] > lat[-1] else np.degrees(np.arcsin(x))
        assert np.allclose(gl, lat, atol=1e-3), "not a Gaussian grid"
        wts = w[::-1] if lat[0] > lat[-1] else w
        cell = (R_EARTH ** 2) * (2 * np.pi / nlon) * wts                 # m2 per cell in a row
        a2 = np.broadcast_to(cell[:, None], (lat.size, nlon))
        s = lat[:, None] < -60
        frac = (slf.mean(0) if slf.ndim == 3 else slf) * (glac.mean(0) if glac.ndim == 3 else glac)
        pm = (p + ev).mean(0)                                            # plain 12-month mean
        flux = np.sum(np.where(s, pm * frac * a2, 0.0)) / 1e9
        rows.append({"exp": lab, "glacier_area_s60_km2": np.sum(np.where(s, frac * a2, 0)) / 1e6,
                     "P_s60_glacier_Sv": np.sum(np.where(s, p.mean(0) * frac * a2, 0)) / 1e9,
                     "PminusE_s60_glacier_Sv": flux})
        print("ECHAM", rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(HERE / "echam_antarctic_pme.csv", index=False, float_format="%.5g")
    print("SURFACE_DONE", flush=True)


if __name__ == "__main__":
    main()
