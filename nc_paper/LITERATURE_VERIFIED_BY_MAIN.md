# Literature facts verified directly (main session, via Crossref/search/abstracts)

## Chen et al. 2025, GRL 52, e2025GL114809  — the paper Reviewer 2 says we must discuss
"Mechanisms Driving the Extensive Antarctic Bottom Water in the Glacial Atlantic"
Chen, Yugeng; Song, Pengyang; Chen, Xianyao; Lohmann, Gerrit. doi:10.1029/2025GL114809

What they did and found (from the published abstract):
- Proxy data indicate LGM AABW volume in the Atlantic was nearly FOUR TIMES the modern volume.
- They used an OCEAN-ONLY model (not the coupled AWI-ESM used here) forced with glacial
  ocean and sea-ice conditions.
- Two mechanisms: (1) sea ice moves to much lower latitudes than today, giving a steady,
  abundant supply of very salty dense water near the Antarctic coast that sinks to form AABW;
  (2) WEAKER MIXING between NADW and AABW at the LGM lets AABW retain the cold, dense
  properties of its deep shelf-water origin.

Why this matters for our response (Reviewer 2 asks what WMT adds that Chen et al. did not show):
- Chen et al. diagnose AABW VOLUME and interior MIXING. They do not decompose the surface
  buoyancy forcing. Their result is about what happens to dense water AFTER it forms and how
  it is preserved; ours is about WHY and HOW MUCH dense water is formed at the surface.
- They are ocean-only with prescribed glacial sea ice; we are fully coupled, so sea ice and the
  atmospheric fluxes are free to evolve and we can attribute transformation to individual
  flux components (SW, LW, latent, sensible, sea ice, precipitation, runoff).
- They analyse the glacial Atlantic; we cover five climate states including two interglacials,
  which is what makes the thermal-to-haline REGIME SHIFT visible at all. A single glacial
  snapshot cannot show a regime shift.
- Honest overlap to acknowledge: the physical mechanism on the glacial side (equatorward sea
  ice, coastal brine rejection, dense shelf water) is consistent between the two papers. Our
  contribution is quantifying the partition and showing it reverses between climate states.
- MUST cite and discuss. Lohmann is a co-author on both, so silence looks bad.

## Gray et al. 2023, Paleoceanography and Paleoclimatology 38, e2023PA004666
"Poleward Shift in the Southern Hemisphere Westerly Winds Synchronous With the Deglacial Rise in CO2"
Gray, de Lavergne, Jnglin Wills, Menviel, Spence, Holzer, Kageyama, Michel. doi:10.1029/2023PA004666

Verified findings:
- A 4.8 degree EQUATORWARD shift (95% CI 2.9-7.1 deg) and about 25% WEAKENING of the
  westerlies at the LGM (20 ka) relative to the mid-Holocene (6.5 ka).
- The poleward shift over the deglaciation mirrors the CO2 rise (R2 = 0.98).
- Equatorward westerlies reduce overturning below 2 km and suppress CO2 outgassing from the
  polar Southern Ocean.

Reviewer 3's framing is CORRECT: proxies indicate equatorward + weaker glacial westerlies, and
models generally fail to reproduce this. ACCEPT THE POINT, DO NOT PUSH BACK.

IMPORTANT CLARIFICATION (do not get this wrong in the letter): the "poleward shift" in the
paper's TITLE refers to the DEGLACIAL TRANSITION (20 -> 10 ka, winds migrating poleward as CO2
rises). The reviewer is describing the LGM STATE relative to the Holocene. Same paper, same
result, two different statements. There is no contradiction to exploit.

Verbatim from the conclusions: "We infer a 4.8 degrees (2.9-7.1 degrees, 95% CI) equatorward
shift and a ~25% weakening of the westerlies during the LGM relative to the mid-Holocene."
Wind strength: peak westerly wind stress weaker by 0.034 N/m2 (~25%), giving an LGM strength of
0.106 (0.085-0.12) N/m2, taking mid-Holocene = modern climatology 0.14 N/m2.
Reference state matters: 4.8 deg is LGM vs mid-Holocene (6.5 ka); a separate 6.3 deg (4.3-8.7)
is LGM vs 10 ka. Always name the reference state when quoting.
On models: the inferred LGM-to-mid-Holocene shift is "substantially greater than that predicted
by ANY of the models within the PMIP3/4 ensemble" - so this is a PMIP-wide bias, not an
AWI-ESM2 defect. Say that.

BONUS THAT PROTECTS OUR RESULT: Gray et al. explicitly tested sea ice as a confound and found
none. "we find no correlation between Antarctic sea ice extent and the SST front latitude in
the model ensemble (Figure 2e; R2 = 0.02), nor do we find a correlation between sea ice extent
and the wind latitude (R2 = 0.03)." So a westerly-position bias does not automatically
propagate onto sea-ice extent, which is what our mean-state WMT result actually rests on.
Recommended posture: concede the wind bias, note it is PMIP-wide, report our own LGM-PI wind
shift honestly against their 4.8 deg, then use the R2 = 0.02-0.03 decoupling to bound the
scope of the damage. Their transient runs also show equatorward winds REDUCE overturning below
2 km, which is the direction a reviewer would ask about.

OUR MODEL DISAGREES WITH THE PROXY, and we must say so plainly. Measured from our own output
(JJA zonal-mean 10 m zonal wind, maximum between 70S and 30S):
  PI   jet at 45.70 S, 5.83 m/s
  MH   jet at 45.70 S, 5.84 m/s   (dlat  0.00, dU +0.02)
  LIG  jet at 47.56 S, 6.92 m/s   (dlat -1.87, dU +1.09)
  LGM  jet at 49.43 S, 6.30 m/s   (dlat -3.73, dU +0.47)
  MIS3 jet at 49.43 S, 6.85 m/s   (dlat -3.73, dU +1.03)
So our glacial jet is 3.7 deg POLEWARD and STRONGER, opposite in sign to Gray et al.
This is the same bias Reviewer 3 attributes to models generally. Implication to state: if the
real glacial westerlies were equatorward and weaker, our simulated glacial wind forcing is
biased, which would affect Ekman divergence and the SAM-related mechanism. The mean-state
haline dominance is driven by sea ice extent and brine rejection rather than by jet position,
so the regime shift itself is robust to this bias, but the SAM/wind results should be
presented with this caveat attached rather than as a confident paleo-reconstruction.
It also answers Reviewer 3's separate question "why do the westerlies intensify in both the
glacial and last interglacial?" - LIG intensification is orbitally forced (high eccentricity,
different perihelion), the glacial one follows the expanded ice/steepened meridional
temperature gradient. Both are model responses, not drift, and MH is essentially unchanged
from PI which argues against a drift explanation.

---

## Citation audit of the revised manuscript (done at the end of the revision)

Prompted by a warning about possible key collisions in `Ref.bib`. All clear:

- **No duplicate BibTeX keys** in `build/ref.bib` (checked by extracting and sorting all keys).
- **Gray2024 vs Gray2023 are distinct and both correct.** `Gray2024` is A. R. Gray's
  carbon-cycle review, and it is cited only for Southern Ocean heat/carbon storage and the ~40%
  anthropogenic CO2 uptake figure, which is what that review actually covers. `Gray2023` is
  W. R. Gray et al. on the westerlies, newly added, cited only for the glacial wind shift. The
  compiled bibliography renders both, as "Gray, A.~R." and "Gray, W.~R." respectively.
- **Lhardy2022 vs Lhardy2021.** `Lhardy2022` (the companion sea-ice evaluation paper) exists in
  the bib but is NOT cited anywhere in the manuscript, so there is no risk of it being read as the
  2021 model analysis. `Lhardy2021` is newly added and resolves correctly in the .bbl.
- **All five new references verified in the compiled bibliography**: Chen2025, Lhardy2021,
  Gray2023, Millet2025, ToggweilerSamuels1995 all render with the correct titles.

### The AWI-evaluation constraint is satisfied
No Southern Ocean bias number is attributed to any AWI paper anywhere in the revision.
`sidorenko2019evaluation` is cited exactly once, in the Methods, purely as model provenance
("a coupled Earth system model developed at the Alfred Wegener Institute"). Rackow 2019,
Semmler 2020 and Sidorenko 2021 are not cited at all in the revised text.
Every statement about this model convecting in the open ocean is carried as OUR diagnosis from
OUR output (the Weddell-sector mixed-layer statistics), supported by Heuze 2021 as the
community-wide picture. This is the correct and defensible arrangement; do not "improve" it later
by attaching an AWI citation to a bias number that no AWI paper actually states.

Open item if a concrete AWI-specific sea-ice or MLD bias figure is ever wanted: the Sidorenko 2019
PDF would have to be read directly first. Nothing in the current revision depends on it.
