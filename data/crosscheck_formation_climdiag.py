#!/usr/bin/env python
"""Cross-check of the regenerated formation-rate CSVs (finding D2) with climdiag.

Recomputes the surface WMT of all 1200 (LIG 1176) months from the inputs of the 100-year
pipeline ({exp}/100years/*_mergetime*.nc, transcribed from {exp}/wmt_100years_timeseries_{exp}.py)
with climdiag.surface_wmt in 0.2 sigma2 bins, then the formation rates with
climdiag.formation_rate, and compares them with
  - the 100-year WMT files the main figure reads ({exp}/wmt_results/), every month and bin;
  - data/aabw_formation_rates_final.csv and data/aabw_formation_anomalies_final.csv.
Only nodes south of 60S are used: every region of the pipeline lies there.
Run on a compute node (slurm_crosscheck.sh). Writes crosscheck_formation_climdiag.txt.
"""
import subprocess
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

CLIMDIAG = Path("/work/ba1066/a270064/cc_projects/climdiag")
sys.path.insert(0, str(CLIMDIAG / "src"))
import climdiag  # noqa: E402
from climdiag.wmt import formation_rate, surface_wmt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
MESH = {"pi": "mesh_core2", "mh": "mesh_core2", "lig": "mesh_core2", "lgm": "mesh_glac1d",
        "mis": "mesh_glac1d_38k"}          # what the 100-year pipeline reads (surface only)
WINDOW = {"pi": "pi_age 2300-2399", "mh": "mh_age 2300-2399",
          "lig": "lig_age 2300-2399 without 2318 and 2372", "lgm": "lgm_age 2400-2499",
          "mis": "mis_age 2500-2599"}
REGIONS = {"southern_ocean": ((-90, -60), (-180, 180)), "ross_sea": ((-90, -60), (-180, -60)),
           "weddell_sea": ((-90, -60), (-60, 79)), "adelie": ((-90, -60), (79, 180))}
REGION_NAMES = {"southern_ocean": "Southern Ocean", "weddell_sea": "Weddell Sea",
                "ross_sea": "Ross Sea", "adelie": "Adélie Land"}
NAMES = {"pi": "PI", "mh": "MH", "lig": "LIG", "lgm": "LGM", "mis": "MIS3"}
THRESH = {"pi": 36.625, "mh": 36.625, "lig": 36.625, "lgm": 37.625, "mis": 37.125}
FW = {"evaporation": "evs", "snow": "prsn", "sea_ice_melt": "fsitherm",
      "rain_and_ice": "prlq", "rivers": "friver"}
FILE_VAR = {"heat_total_heat": "total_heat_surface_exchange_flux_nonadvective_heat",
            **{f"freshwater_{k}": f"surface_ocean_flux_advective_negative_rhs_{k}_salt" for k in FW},
            "salt_basal": "surface_exchange_flux_nonadvective_salt"}
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


def inputs(exp):
    d = ROOT / exp / "100years"
    with netCDF4.Dataset(f"/home/a/a270064/bb1029/inputs/{MESH[exp]}/fesom.mesh.diag.nc") as m:
        area = np.asarray(m["nod_area"][0], dtype=float)
        nodes = np.asarray(m["nodes"][:], dtype=float)
    lon, lat = np.degrees(nodes[0]), np.degrees(nodes[1])
    sel = lat <= -60
    def fes(name, var):
        with netCDF4.Dataset(d / f"{name}_mergetime.nc") as f:
            return np.asarray(f[var][:])[:, sel]
    rho = 1000.0
    x = dict(lon=lon[sel], lat=lat[sel], area=area[sel],
             sst=fes("sst", "sst"), sss=fes("sss", "sss"))
    x["prlq"] = fes("prec", "prec") * rho
    x["prsn"] = fes("snow", "snow") * rho
    x["evs"] = fes("evap", "evap") * rho
    x["friver"] = fes("runoff", "runoff") * rho
    wfo = -fes("fw", "fw") * rho
    x["fsitherm"] = wfo - x["evs"] - x["prlq"] - x["prsn"] - x["friver"]
    x["sfdsi"] = x["fsitherm"] * 5.0 * 0.001
    heat = None
    for v in ("92", "95", "111", "120"):
        with netCDF4.Dataset(d / f"{v}_mergetime_echam.nc") as e:
            elon = np.asarray(e["lon"][:]); elat = np.asarray(e["lat"][:])
            a = np.asarray(e[f"var{v}"][:])
        elon = np.where(elon > 180, elon - 360, elon)
        LO, LA = np.meshgrid(elon, elat)
        pts = np.column_stack([LO.ravel(), LA.ravel()])
        xi = np.column_stack([x["lon"], x["lat"]])
        out = np.zeros((a.shape[0], xi.shape[0]), dtype=np.float32)
        for t in range(a.shape[0]):
            out[t] = griddata(pts, a[t].ravel(), xi, method="nearest")
        heat = out.astype(np.float64) if heat is None else heat + out
    x["heat"] = heat
    return x


def main():
    commit = subprocess.run(["git", "-C", str(CLIMDIAG), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    say(f"climdiag {climdiag.__version__} ({commit})")
    rates = pd.read_csv(HERE / "aabw_formation_rates_final.csv")
    anoms = pd.read_csv(HERE / "aabw_formation_anomalies_final.csv")
    got = {}
    worst_rate = 0.0
    for exp in ["pi", "mh", "lig", "lgm", "mis"]:
        x = inputs(exp)
        da = lambda v: xr.DataArray(np.asarray(v), dims=("time", "nod2"))  # noqa: E731
        one = lambda v: xr.DataArray(np.asarray(v), dims=("nod2",))  # noqa: E731
        for reg, ((la0, la1), (lo0, lo1)) in REGIONS.items():
            mask = (x["lat"] >= la0) & (x["lat"] <= la1) & (x["lon"] >= lo0) & (x["lon"] <= lo1)
            w = surface_wmt(
                sst=da(x["sst"]), sss=da(x["sss"]), area=one(x["area"]), lat=one(x["lat"]),
                region=one(mask), region_name=reg, bins=np.arange(0, 40, 0.2),
                p_ref="p_from_z(-2000m)", temperature="conservative", salinity="absolute",
                cp=3992.0, rho_ref=1035.0, rho_volume=1000.0,
                heat={"total_heat": da(x["heat"])}, heat_positive="into_ocean",
                freshwater={k: da(x[v]) for k, v in FW.items()}, freshwater_positive="into_ocean",
                salt={"basal": da(x["sfdsi"])}, salt_positive="into_ocean",
                averaging_window=WINDOW[exp] + ", monthly")
            with xr.open_dataset(ROOT / exp / "wmt_results" / f"wmt_{reg}_100years_{exp}.nc") as f:
                f = f.load()
            assert np.allclose(w["sigma2"].values, f["sigma2_l_target"].values)
            assert w.sizes["time"] == f.sizes["time"]
            diffs = []
            for mine, theirs in FILE_VAR.items():
                ref = -f[theirs].values / 1e9
                diffs.append(np.abs(w[mine].values - ref).max() / max(np.abs(ref).max(), 1e-30))
            fr = lambda v: float(formation_rate(w[v], THRESH[exp], time_mean="equal_weights",  # noqa
                                                method="linear"))
            heat = fr("heat_total_heat")
            sea = fr("freshwater_sea_ice_melt")
            oth = sum(fr(f"freshwater_{k}") for k in ("evaporation", "rain_and_ice", "snow", "rivers"))
            got[(exp, reg)] = dict(Total_WMT=heat + sea + oth, Heat=heat, SeaIce=sea, Other_FW=oth)
            row = rates[(rates.Experiment == NAMES[exp]) & (rates.Region == REGION_NAMES[reg])].iloc[0]
            d = max(abs(got[(exp, reg)][k] - row[k]) for k in got[(exp, reg)])
            worst_rate = max(worst_rate, d)
            say(f"{exp:4s} {reg:15s} months {w.sizes['time']}: files vs climdiag max rel diff "
                f"{max(diffs):.1e}; CSV vs climdiag max |diff| {d:.4f} Sv "
                f"(Total {got[(exp, reg)]['Total_WMT']:.3f})")
    worst_anom = 0.0
    for (exp, reg), v in got.items():
        if exp == "pi":
            continue
        a = anoms[(anoms.Experiment == NAMES[exp]) & (anoms.Region == REGION_NAMES[reg])].iloc[0]
        pi = got[("pi", reg)]
        for k, col in (("Total_WMT", "ΔWMT"), ("Heat", "ΔHeat"), ("SeaIce", "ΔSeaIce"),
                       ("Other_FW", "ΔOther_FW")):
            worst_anom = max(worst_anom, abs((v[k] - pi[k]) - a[col]))
    say(f"worst |CSV - climdiag|: rates {worst_rate:.4f} Sv, anomalies {worst_anom:.4f} Sv "
        "(CSVs carry 3 decimals)")
    (HERE / "crosscheck_formation_climdiag.txt").write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
