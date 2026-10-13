# Can we compute an observational WMT from a reanalysis?

Asked because Jiping Liu (call of 2026-10-13) wants our pre-industrial WMT
compared against an observational estimate, and suggested the Copernicus ocean
reanalysis. Reviewer 3 had made the same suggestion, naming GLORYS, and the
submitted response letter declined it.

**Answer: no, not without building the sea-ice term ourselves. The fallback is
to reproduce Pellichero et al. (2018) Fig. 3a and compare against it.**

## What a WMT needs

Our pipeline (`calc_wmt_siz_pellichero.py`) needs, per surface cell:

| term | our source |
|---|---|
| net surface heat flux | model |
| net surface freshwater flux | `fw_clim.nc` → `fw2 = -fw*rho` |
| sea-ice thermodynamic flux | **residual**: `fsitherm = fw2 - evs - prlq - prsn - friv` |
| surface T, S | model, for alpha and beta |

The residual only works because the model's total freshwater flux already
contains the ice term. A product whose total excludes it cannot give the
residual, and the ice term is what dominates our glacial result.

## What the candidates actually carry

Checked 2026-10-13, partly by the `aas` session against the CMEMS catalogue,
partly here against Levante.

**CMEMS / Copernicus — no flux fields at all.**
- `GLOBAL_MULTIYEAR_PHY_001_030` (GLORYS12): state, sea ice, no fluxes.
- `GLOBAL_MULTIYEAR_PHY_ENS_001_031` (GREP): same, as ensemble members.
- `GLOBAL_ANALYSISFORECAST_PHY_001_024`: same.

**ORAS5 — has heat and freshwater, but NOT the brine term.**
`/pool/data/ICDC/ocean_syntheses/oras5/{ORCA025,r1x1}/<var>/opa0..opa4`,
monthly from 1979-01. `sohefldo` (W/m2, net downward heat), `sowaflup`
(kg/m2/s, net upward water), `sosstsst`, `sosaline`, `ileadfra`, `iicethic`.

Tested whether `sowaflup` includes sea-ice freezing and melting, since the
metadata does not say. Took the September 1990 ice zone (`ileadfra` > 0.15,
south of 50S, 154712 points) and compared peak freezing against peak melt:

    SEP (max freezing)  mean -1.216e-05 kg/m2/s
    FEB (max melt)      mean -1.563e-05 kg/m2/s

Positive would mean the ocean loses freshwater, i.e. brine. September is
negative and barely differs from February, so there is no freezing signal:
`sowaflup` is E - P - R only. Basin-wide south of 60S gives the same answer,
July -9.0e-06 against January -2.1e-05.

**ERA5 — atmospheric fluxes only.**
`/pool/data/ERA5/E5/sf/fc/1M/<param>/`, 1940 onwards. Codes from
`READMEs/README_ERA5_PARAMIDs.txt`: 146 sensible, 147 latent, 176 net SW,
177 net LW, 182 evaporation, 144 snowfall, 205 runoff, plus 031 sea-ice cover
and 034 SST under `sf/an/1M`. Reduced Gaussian grid, so `cdo -setgridtype,regular`
first. Over ice these are atmosphere-to-ice fluxes, not fluxes into the ocean.

**GECCO3** `/pool/data/ICDC/ocean_syntheses/GECCO3/regular_1x1_grid/`, 852
months from 1948: `emp`, `qnet`, `tflux`, `sflux`, `heff`, `area`, `uice/vice`.
Whether `emp` includes the ice term was NOT tested; run the same
September-against-February check before relying on it.

**Not on Levante:** JRA55-do, ECCO v4r4, B-SOSE. The old ICDC `ECCO` is a
coarse version ending around 2002.

## If someone does want to build it

Derive the sea-ice freshwater flux from an ice mass budget (Haumann et al.
2016 style): volume tendency plus the divergence of the ice volume flux, from
`iicethic`, `ileadfra`, `iicevelu/v`. That is a separate piece of work, not a
data lookup, and ice-shelf melt and icebergs would still need a climatology.

## Trap

ORAS5 filenames differ by variable family:

    sowaflup/opa0/sowaflup_ORAS5_1m_199009_grid_T_02.nc
    ileadfra/opa0/ileadfra_ORAS5_1m_199009_icemod_02.nc

`grid_T` for T-grid fields, `icemod` for ice fields.
