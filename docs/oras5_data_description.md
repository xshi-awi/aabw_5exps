# ORAS5: data description

Text for the response letter or the manuscript, with the facts checked against
Zuo et al. (2019) and against the files on Levante. The important caveat is in
the last paragraph: ORAS5 is not an observational product, and its surface
fluxes in particular are not observed.

## Short version, for a figure caption

> ORAS5 is the ECMWF Ocean ReAnalysis System 5 (Zuo et al., 2019), a global
> ocean and sea-ice reanalysis rather than an observational product. It couples
> the NEMO 3.4.1 ocean model to the LIM2 sea-ice model on the eddy-permitting
> ORCA025.L75 tripolar grid, about 25 km horizontally with 75 vertical levels,
> forced by ERA-Interim and constrained by assimilation of temperature and
> salinity profiles, altimetric sea level, sea surface temperature and sea ice
> concentration using NEMOVAR 3D-Var FGAT. We use the 1999-2018 monthly means
> of ensemble member opa0.

## Long version, for the letter

> The reanalysis used here is ORAS5, the ECMWF Ocean ReAnalysis System 5 (Zuo
> et al., 2019). It couples the NEMO ocean model version 3.4.1 to the LIM2
> sea-ice model with viscous-plastic rheology, on the global tripolar
> ORCA025.L75 grid: eddy-permitting, approximately 25 km in the tropics and
> finer at high latitudes, with 75 vertical levels from 1 m at the surface to
> 200 m in the deep ocean. Surface forcing is taken from the ERA-Interim
> atmospheric reanalysis, with heat and freshwater fluxes computed through the
> revised CORE bulk formulae of Large and Yeager (2009).
>
> Observations enter through NEMOVAR in its 3D-Var FGAT configuration with a
> five-day window: temperature and salinity profiles from EN4, altimetric sea
> level anomalies from AVISO, sea surface temperature from HadISST2.1 and later
> OSTIA, and sea ice concentration. Sea surface temperature is not assimilated
> in the strict sense but imposed by modifying the surface non-solar heat flux,
> with a restoring coefficient of -200 W m-2 K-1, and surface salinity is
> restored towards climatology. ORAS5 covers 1979 onwards; the copy held at
> DKRZ runs from January 1979 to December 2018 as complete monthly means, in
> five ensemble members. We use member opa0 over 1999-2018, the part of the
> record in which Argo gives the Southern Ocean its strongest observational
> constraint, and which overlaps the periods used by Pellichero et al. (2018).

## The caveat that has to be stated

ORAS5 is a model constrained by observations, not a measurement. That
distinction matters more than usual for this comparison, because **the two
fields we need are the two that are least constrained**.

No observing system measures the air-sea heat or freshwater flux directly.
Argo measures temperature and salinity, altimeters measure sea level,
radiometers measure skin temperature and ice cover. ORAS5's `sohefldo` and
`sowaflup` are therefore computed by the model from ERA-Interim forcing, and
are then adjusted by the SST and salinity restoring terms rather than by any
flux observation. In the seasonal ice zone, where profile coverage is sparse
even in the Argo era, that leaves them weakly constrained.

This is exactly the objection Pellichero et al. (2018) raise when they decline
to use a reanalysis for this calculation:

> "Ocean buoyancy fluxes are commonly obtained from atmospheric reanalysis
> datasets. Unfortunately, in the Southern Ocean such reanalysis products are
> poorly constrained, and essentially unusable in the seasonally sea-ice
> covered region."

So a comparison against ORAS5 is a comparison between two models, with one of
them pulled towards observations. It is not independent validation. It is still
informative, because ORAS5 uses a different ocean model, a different grid and
different atmospheric forcing, so agreement is not guaranteed by construction,
but the claim it supports is weaker than agreement with Pellichero et al.

A second limitation is specific to this project. ORAS5's `sowaflup` carries no
sea ice thermodynamic term: tested inside the September ice zone it shows no
freezing signal, -1.2e-5 kg m-2 s-1 at peak freezing against -1.6e-5 at peak
melt. It is evaporation minus precipitation minus runoff. Any comparison must
therefore remove the sea ice contribution from our own flux as well, which is
why the comparison panels show the model with that term taken out.

## References

Zuo, H., Balmaseda, M. A., Tietsche, S., Mogensen, K., and Mayer, M. (2019).
The ECMWF operational ensemble reanalysis-analysis system for ocean and sea
ice: a description of the system and assessment. *Ocean Science*, 15, 779-808.
doi:10.5194/os-15-779-2019

Large, W. G. and Yeager, S. G. (2009). The global climatology of an
interannually varying air-sea flux data set. *Climate Dynamics*, 33, 341-364.

## Data location

    /pool/data/ICDC/ocean_syntheses/oras5/ORCA025/<var>/opa0/
        <var>_ORAS5_1m_<YYYYMM>_grid_T_02.nc      T-grid fields
        <var>_ORAS5_1m_<YYYYMM>_icemod_02.nc      sea ice fields

Variables used: `sohefldo` net downward heat flux (W m-2), `sowaflup` net
upward water flux (kg m-2 s-1), `sosstsst`, `sosaline`, `ileadfra` ice
concentration. Cell areas from `mesh/mesh_mask.nc` as `e1t * e2t`.

Note the filename trap: T-grid fields use the `grid_T` infix and sea ice fields
use `icemod`.

`sohefldo` is the flux into the ocean, already accounting for ice cover, and
must not be multiplied by ice concentration again. Checked directly: over cells
with more than 98% ice concentration south of 60 S in September it averages
-13.6 W m-2, the magnitude of an ice-ocean flux rather than of an
atmosphere-ice flux, which would be several times larger.
