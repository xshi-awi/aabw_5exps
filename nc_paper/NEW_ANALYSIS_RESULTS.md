# New quantitative results computed for the revision

All numbers produced in this session from model output. Scripts named per item.

## 1. Domain sensitivity of the thermal/haline partition  (Issue A: Pellichero mismatch)
Script `calc_wmt_siz_pellichero.py` -> `wmt_siz_{exp}.nc`; figure `plot_wmt_domain_sensitivity.py`
-> `figures/figR1_wmt_domain_sensitivity.{pdf,png}`.

Two integration domains, annual-mean climatological WMT (Sv), peak of total transformation:

| exp | domain | area 1e13 m2 | peak Sv | sigma2 | heat | sea ice | thermal % |
|-----|--------|------|------|------|------|------|------|
| PI   | <60S | 2.07 | 10.60 | 36.70 | 16.70 |  0.86 | 68% |
| PI   | SIZ  | 1.25 |  6.82 | 36.90 |  5.60 |  1.84 | 70% |
| MH   | <60S | 2.07 | 10.10 | 36.90 |  9.50 |  1.56 | 79% |
| MH   | SIZ  | 1.11 |  3.77 | 36.70 |  6.89 |  1.47 | 53% |
| LIG  | <60S | 2.07 |  9.97 | 36.90 |  9.99 |  0.65 | 88% |
| LIG  | SIZ  | 0.84 |  4.39 | 36.70 |  8.38 | -1.50 | 68% |
| LGM  | <60S | 1.76 | 10.22 | 37.70 |  3.43 | 13.13 | 15% |
| LGM  | SIZ  | 3.23 |  9.31 | 37.70 |  3.66 | 13.58 | 15% |
| MIS3 | <60S | 1.82 |  8.24 | 37.10 |  4.66 | 18.40 | 12% |
| MIS3 | SIZ  | 3.00 |  4.13 | 37.30 |  2.95 |  7.90 | 17% |

SIZ = seasonal sea ice zone, climatological max sea ice concentration > 15%.

Key numbers for the response letter:
- 41% of the published <60S PI domain is open water that never sees sea ice. That
  sub-domain alone is 89-91% thermal in the dense classes, so it does dilute the haline share.
- Restricting to the ice-covered sector moves the partition in the direction Pellichero
  report (PI dense-class thermal share 69% -> 61%; sea-ice term roughly doubles,
  0.86 -> 1.84 Sv at the peak) but does NOT reverse it.
- The regime shift itself is domain-independent: glacials are 15%/17% thermal in BOTH
  domains, interglacials 53-88%. So the headline result is not a domain artefact.
- Honest conclusion to state: part of the discrepancy is the domain, the remainder is
  a genuine model bias (open-ocean convection instead of shelf processes). Both should
  be said plainly.

Validation: this climatology-based pipeline reproduces the published 100-year result
(annual peak 10.60 vs 11.39 Sv; JJA peak recovers the quoted 60-80 Sv, at 69.86 Sv).

## 2. Overturning streamfunction  (Reviewer 3 request)
Source: model's own online diagnostic `/work/ba1066/a270064/production/{exp}_age/diag_gmoc.nc`,
mean of the final 20 diagnostic years. Script `plot_moc_5exps.py` -> `figures/figR2_moc_5exps.{pdf,png}`.

| exp | AABW cell (Sv) | NADW max (Sv) |
|-----|------|------|
| PI   | -10.0 | 19.1 |
| MH   | -10.1 | 19.1 |
| LIG  |  -9.2 | 21.3 |
| LGM  |  -2.4 | 20.0 |
| MIS3 |  -1.6 | 21.7 |

The abyssal cell collapses by roughly 75-85% in the glacials while the upper NADW cell
is nearly unchanged. This is independent confirmation of the ventilation-age result and
answers the reviewer's point that overturning should be shown directly.

## 3. Global ideal age  (Reviewer 3: show ventilation globally, not one 4000 m level)
Script `plot_age_global_basins.py` -> `figures/figR3_age_global.{pdf,png}`.
Basin-mean ideal age below 2000 m (years):

| basin | PI | MH | LIG | LGM | MIS3 |
|-------|----|----|-----|-----|------|
| Atlantic | 377 | 367 | 361 |  439 |  508 |
| Indian   | 493 | 492 | 498 |  737 |  760 |
| Pacific  | 847 | 828 | 829 | 1291 | 1314 |
| Southern | 381 | 382 | 378 |  789 |  844 |

Note for the text: below 2000 m the Atlantic ages by only +62 yr at LGM whereas the
Southern Ocean more than doubles (381 -> 789 yr). The basin contrast is much stronger
than the single 4000 m map conveyed.

## 4. Where does the model actually form deep water?  (Reviewer 3)
JJA mixed layer depth, MLD1_clim.nc, area with MLD above threshold south of 55S.

PI, MLD > 400 m: total 7.51e11 m2, of which
  Weddell sector (60W-79E) 5.26e11 (70%), mean latitude 68.0 S
  Ross sector (180W-60W)   1.97e11 (26%), mean latitude 56.8 S
  Adelie sector (79E-180)  0.28e11 (4%)
PI, MLD > 600 m: 2.42e11 m2, 97% of it in the Weddell sector at 68.7 S. Max MLD 991 m.

LGM: MLD>400 m area 1.66e11 m2 (22% of PI), max MLD 819 m, MLD>600 m essentially gone (3 nodes).
MIS3: MLD>400 m area 1.61e11 m2, max MLD 768 m, MLD>600 m 3 nodes.

Interpretation to give the reviewer: in PI the model forms its dense water by open-ocean
convection in the Weddell gyre interior near 68 S, not on the shelf. That is the bias the
reviewer suspects, and we should name it explicitly. It also quantifies the insulation
argument: glacial deep-convection area falls to about one fifth of PI.

## 5. Boundary conditions, read from the actual run configurations
(For the summary table requested by Reviewer 1. These are the real namelist values.)

| exp | CO2 (ppm) | CH4 (ppb) | N2O (ppb) | eccentricity | obliquity | perihelion |
|-----|------|------|------|------|------|------|
| PI   | 284.3 | 808.2 | 273.0 | yr_perp = 1850 (VSOP87) | | |
| MH   | 264.4 | 597.0 | 262.0 | 0.018682 | 24.105  | 180.87 |
| LIG  | 275.0 | 685.0 | 255.0 | 0.039378 | 24.040  |  95.41 |
| LGM  | 190.0 | 375.0 | 200.0 | 0.018994 | 22.949  | 294.42 |
| MIS3 | 210.5 | 556.2 | 247.4 | 0.013676 | 23.2591 |  25.99 |

Meshes / ice sheets:
  PI, MH, LIG -> mesh_core2 (awicm2_final384), modern land-sea mask and bathymetry
  LGM  -> mesh_glac1d,      GLAC-1D 21 ka
  MIS3 -> mesh_glac1d_38k,  GLAC-1D 38 ka
GLAC-1D sets the ocean mesh (bathymetry and land-sea mask), the ECHAM/JSBACH T63 surface
files, the river routing (hdpara.nc) and the vegetation boundary file
(jsbach ... natural-veg.GLAC1D_21k.nc / _38k.nc). Scenario flag "PALEO".
No prescribed land-ice meltwater flux was applied (use_landice_water: False) — worth
stating explicitly, since a reviewer asks about ice-sheet freshwater.
Glacial runs initialised from a previous glacial state; ocean initial hydrography from
PHC3.0 (interglacials) or the LGM hydrography directory (glacials).

## Caveats to carry into the text
- No ice-shelf cavities in this configuration, so ice-shelf basal melt is absent. Say so.
- Ideal age equilibration: runs are 1000 yr, Pacific deep ages reach ~1300 yr, so the
  glacial ages are not fully equilibrated and should be described as a lower bound.
  Reviewer 3 raises exactly this (Millet et al. 2025).

---

## 6. Refined Pellichero comparison (after checking their exact domain definition)

Pellichero et al. define their sector as the region inside the WINTER (September) sea-ice
edge with concentration > 15%. Our seasonal-ice-zone mask (annual maximum > 15%) gives
1.2490e13 m2; their September definition applied to our PI run gives 1.2177e13 m2. The two
differ by 2.5% in area with identical latitude span (78.5 S to 58.2 S), so the domain
comparison already IS like-for-like and the conclusion in section 1 stands.

### Is the thermal expansion coefficient argument applicable to our model?
Pellichero attribute part of the haline dominance to alpha being near zero at the freezing
point. Checked in our PI run, JJA, inside the September ice zone:
  area-weighted SST = -1.08 C, SSS = 33.90
  SST percentiles: 5% -1.84, 50% -1.71, 95% +0.06
  alpha = 3.51e-05 /K, beta = 7.85e-04 /psu
  55% of the ice-zone area is within 0.5 C of the local freezing point
So our ice zone IS in the near-freezing, low-alpha regime they describe. The model is not
warm-biased there, and the residual thermal dominance cannot be explained away by alpha.
This is worth stating: it removes one possible excuse and makes the answer more honest.

### Where the two studies actually agree and disagree — by density class
PI, seasonal ice zone, annual mean (Sv):
  sigma2 36.5-37.0 : total 10.97 | heat 21.55 | sea ice  5.16 | other -15.74
  sigma2 37.0-37.5 : total  0.74 | heat  0.06 | sea ice  0.79 | other  -0.11
  sigma2 36.8-38.5 : total  7.84 | heat  5.70 | sea ice  2.96 | other  -0.81

Key point for the response: in the DENSEST classes (sigma2 >= 37.0), which are the classes
relevant to bottom water, our PI transformation IS haline dominated (sea ice 0.79 Sv vs heat
0.06 Sv). The thermal dominance we report sits at INTERMEDIATE densities (36.5-37.0), which
in the real ocean correspond more closely to mode/intermediate water than to AABW.
Pellichero's "to denser water 5 +/- 5 Sv" is the same order as our 7.84 Sv over 36.8-38.5.

So the honest framing is: the disagreement is about WHICH DENSITY CLASS dominates the total,
not a wholesale contradiction. Our model transforms too much water thermally at intermediate
densities because it convects in the open Weddell gyre, and that intermediate-density thermal
branch swamps the total when integrated over all classes. Restricting either the domain (to
the ice zone) or the density range (to the bottom-water classes) recovers haline dominance.
This connects directly to Reviewer 3's suggestion (L175) that the result may be more
applicable to AAIW/SAMW formation than to AABW - that suggestion is essentially correct and
should be conceded and incorporated rather than resisted.

---

## 7. The decisive check: our density classes vs Pellichero's density classes

Two checks prompted by the exact wording of Pellichero's method.

### Check 1 - does our published domain extend equatorward of their boundary? YES.
Their northern boundary is the September 15% sea ice contour; there are no latitude bounds.
Applied to PI:
  our published <60S domain          2.0708e13 m2
  September ice zone                 1.2177e13 m2
  <60S but OUTSIDE the Sept ice zone 8.8382e12 m2 = 42.7% of our domain
  Sept ice zone north of 60S         3.0739e11 m2 = 2.5% of the ice zone
So 42.7% of the area we integrated over is water Pellichero et al. never analysed. That alone
makes the published comparison not like-for-like, and it is the open, ice-free part where
thermal forcing is strongest.

### Check 2 - do our thermal-dominant classes even overlap their dense cell? NO.
Converting our PI JJA surface properties inside the September ice zone from sigma2 to a
gamma-comparable sigma0:
  sigma2 36.5-37.0 (our THERMAL-dominant band) -> sigma0 27.09-27.58, median 27.28
  sigma2 37.0-37.5 (our haline-dominant band)  -> sigma0 27.57-27.94, median 27.67
  Pellichero's denser/lower cell                = gamma_n 27.9-28.8

Our thermally dominated transformation sits at 27.1-27.6, entirely LIGHTER than their lower
cell. Even our densest surface outcrop classes only just reach the bottom edge of their range
at about 27.94. Their haline-dominated lower cell is therefore a density range that our PI
surface transformation barely populates at all.

CONCLUSION FOR THE RESPONSE LETTER. The apparent contradiction largely dissolves once domain
and density class are matched. The two studies are not measuring the same water:
  - Pellichero: inside the winter ice edge, gamma_n 27.9-28.8, the LOWER (bottom water) cell.
  - Our headline PI number: all water south of 60S, integrated over all classes, whose peak
    sits at sigma2 36.7-36.9 = gamma_n ~27.3, i.e. UPPER/INTERMEDIATE water.
When we match them on either axis we recover haline dominance (section 6: at sigma2 >= 37.0
sea ice 0.79 Sv vs heat 0.06 Sv).
This validates Reviewer 3's own suggestion at L175 that our result may be more applicable to
AAIW/SAMW than to AABW. Concede it and use it - it is the cleanest resolution available and it
is defensible from our own output.
Residual honest caveat: the reason our surface transformation does not reach the denser classes
in PI is that the model convects in the open Weddell gyre instead of producing dense shelf
water, which is the genuine bias. Both statements should appear together.
