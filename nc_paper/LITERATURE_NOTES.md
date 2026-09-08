# Literature Notes for Reviewer Response

Substantive content notes for the NC revision. Companion to `reviewer_refs_verified.bib`
(all bibliographic fields Crossref/OpenAlex-verified).

**Sourcing convention used throughout:** text in quotation marks is verbatim from the paper.
Everything else is close paraphrase. Where I could not verify something, it says so explicitly.
No number in this document is invented.

---

## 1. Chen et al. 2025 GRL — the "what does WMT add?" question

**Chen, Y., Song, P., Chen, X., & Lohmann, G. (2025).** Mechanisms Driving the Extensive
Antarctic Bottom Water in the Glacial Atlantic. *Geophysical Research Letters* **52**(8),
e2025GL114809. DOI 10.1029/2025GL114809. Open access, CC-BY.

### Model and setup

Ocean–sea-ice only. **FESOM2** (Danilov et al. 2017), i.e. the same ocean core as our AWI-ESM2,
but run **without an interactive atmosphere**. Present-day run forced by JRA55-do reanalysis
(1958–2020). LGM atmospheric forcing taken from Zhang et al. (2013). Results reported as
**62-year averages**. Two states only: **PD and LGM**.

### What they actually diagnose

Their archived output (Zenodo 10.5281/zenodo.15080037 — files `PD.nc`, `LGM.nc`,
`Streamfunction.nc`) contains:

- Atlantic meridional overturning streamfunction
- Mixed layer depth (PD vs LGM)
- Sea-ice velocity, thickness, concentration

That is the complete diagnostic set. **No water mass transformation calculation, no surface
buoyancy or density flux decomposition, no separation of thermal from haline forcing, no
per-component (heat / sea ice / precipitation / runoff) attribution.**

> Caveat on sourcing: Wiley/Cloudflare blocks automated full-text retrieval, so this diagnostic
> list is established from the verified abstract, the Zenodo supplement file contents, and
> search-indexed body text — not from a full read of the PDF. The paper is CC-BY and downloads
> fine in a browser if we want to confirm the figure list before submitting the response.

### What they conclude

Abstract, verbatim:

> "Paleo-proxy data indicate that during the Last Glacial Maximum (LGM), the volume of Antarctic
> Bottom Water (AABW) in the Atlantic was nearly four times greater than it is today. We employed
> an ocean-only model to simulate the galcial ocean and sea-ice conditions. Our simulations reveal
> two key mechanisms driving its greater volume. First, while present-day sea ice formation is
> driven largely by seasonal changes, the glacial mechanism is the substantial export of sea ice
> toward lower latitudes. The glacial sea ice formation was more than quadruple current levels,
> providing a steady source of Dense Shelf Water (DSW) crucial for AABW expansion. Second, weaker
> mixing between North Atlantic Deep Water (NADW) and AABW during the LGM allows the latter to
> maintain the colder, denser properties of its DSW origin. Together, these factors clarify how
> glacial conditions supported significantly greater AABW volumes, aligning well with paleo-proxy
> evidence."

Supporting points from the body: glacial sea-ice formation is sustained by **export** to lower
latitudes rather than by the seasonal freeze/melt cycle, so the glacial mixed layer stays deep
**year-round**, giving a continuous DSW supply. The weaker NADW–AABW mixing comes from elevated
glacial deep-Atlantic salinity combined with a rapid temperature decrease with depth, which raises
stratification; AABW and NADW then behave as **relatively independent closed overturning cells**.

### Honest answer to the reviewer

Three things are true simultaneously, and the response should say all three.

**(a) There is real overlap, and we should concede it.** Chen et al. already argue that glacial
sea ice is central to dense water formation, and that enhanced sea-ice production sustains DSW.
Our finding that the sea-ice/haline pathway gains importance under glacial conditions is
*consistent with*, not independent of, their result. Claiming novelty for the qualitative
mechanism would be overreach.

**(b) What the WMT decomposition adds is quantitative attribution, which their diagnostics cannot
produce.** Streamfunction, MLD and sea-ice concentration tell you *that* dense water forms and
roughly where. They do not tell you *how much* density flux comes from surface heat loss versus
brine rejection versus precipitation versus runoff, in which density classes, or in which season.
The WMT framework converts surface buoyancy fluxes into a transformation rate per density class,
so the thermal and haline contributions can be added, differenced and compared across climate
states in the same units (Sv). Chen et al. infer the sea-ice role from correlated fields; a WMT
budget *measures* it. That distinction is defensible and worth stating plainly.

**(c) Scope differs in two ways that matter.** Chen et al. compare **two** states (PD, LGM) using
an **ocean-only** model with prescribed atmospheric forcing. We compare **five** states (PI, MH,
LIG, LGM, MIS3) in a **coupled** model, so surface fluxes are free to respond to the changing
climate rather than being imposed. A regime *shift* across a glacial–interglacial sequence — the
core claim of our title — is not addressable in a two-state ocean-only design, because the
forcing that would drive the shift is prescribed rather than simulated. Also relevant: an
ocean-only configuration cannot exhibit the coupled sea-ice/heat-flux feedbacks that our
decomposition partitions.

Suggested framing: rather than claiming Chen et al. missed something, say the WMT analysis
**tests and quantifies** the mechanism they proposed, extends it from two states to a
glacial–interglacial sequence, and does so in a coupled framework where surface forcing is
prognostic.

---

## 2. Pellichero et al. 2018 — domain, numbers, and the thermal/haline split

**Pellichero, V., Sallée, J.-B., Chapman, C. C., & Downes, S. M. (2018).** The Southern Ocean
meridional overturning in the sea-ice sector is driven by freshwater fluxes. *Nature
Communications* **9**(1), 1789. DOI 10.1038/s41467-018-04101-2.
(Our existing `Ref.bib` entry lacks the DOI — worth adding.)

### Domain — this is the crux of the like-for-like comparison

The **sea-ice sector only**, defined verbatim as "the region seasonally capped by sea-ice, i.e.,
the region south of the winter (September) sea-ice extension with an ice concentration greater
than 15%." Circumpolar, extending from the Antarctic coast to the September ice edge.

**They do not analyse the open ocean north of the winter ice edge.** No fixed latitude bounds are
given; the northern boundary is the September 15% ice-concentration contour, which varies with
longitude. So the domain is defined by sea-ice climatology, not by a latitude circle.

Data: Argo floats, ship-based observations, and instrumented marine mammals.

### Density coordinate and range

**Neutral density** (γ), converted from potential density where needed. Transformation analysed
over **26.3–28.8 kg m⁻³**, with the reported structure being upper subduction near 26.3–27.3 γ,
upwelling/transformation near 27.3–27.9 γ, and dense water subduction at 27.9–28.8 γ.

### Transformation rates

| Quantity | Value |
|---|---|
| Deep water upwelled to the surface | **27 ± 7 Sv** |
| Transformed to lighter water (upper cell) | **22 ± 4 Sv**, peak at 27.3 γ |
| Transformed to denser water (lower cell) | **5 ± 5 Sv**, peak at 27.9 γ |
| Peak subduction | 9 ± 4 Sv at 27.1 ± 0.05 γ |

Note the lower-cell figure, 5 ± 5 Sv, has an uncertainty as large as the estimate itself.

### Thermal vs freshwater — and the answer to "is thermal ever comparable?"

The key sentence, verbatim:

> "The heat flux contributes only marginally (a factor ~2–5 lower than the freshwater
> contribution), and mostly in regions near the winter sea-ice edge that spend much of the year
> ice-free."

And the physical reason they give:

> "At near freezing temperatures, the thermal expansion coefficient of sea-water is close to zero,
> meaning the density of water is quite insensitive to heat fluxes."

Also: "The meridional gradient in the buoyancy flux and its seasonal evolution is largely
dominated by the freshwater flux contribution."

**Direct answer to the question asked.** Yes — they do identify where the thermal term becomes
non-negligible, and it is precisely **near the winter sea-ice edge, in waters that are ice-free
for much of the year**. The factor of 2–5 is the ratio across their domain, not a uniform value;
their own text localises the thermal contribution to the warmer, seasonally ice-free margin of
the domain. They do **not** report a region where thermal exceeds haline, but the trend they
describe points that way as you move equatorward — and equatorward of their northern boundary
they simply have no analysis.

### How to use this in the response

This is a **defence, not a concession**. Their haline dominance is diagnosed strictly inside the
winter ice edge, where α ≈ 0 mechanically suppresses the thermal term. Two checks to run before
answering the reviewer:

1. **Compare domains.** If our PI domain extends equatorward of the September ice edge — into open
   water where α is not small — then a thermally dominated PI result is not in conflict with
   Pellichero et al. It is a different region of the ocean.
2. **Compare density classes.** Their lower-cell (dense) transformation is 5 ± 5 Sv over
   27.9–28.8 γ. If our thermal dominance sits in different density classes, the comparison again
   is not like-for-like.

If both checks show a genuine overlap in domain and density class, then the discrepancy is real
and should be reported honestly as a model bias. But the comparison must be made explicitly
before conceding. Recommend stating the domain definition used in our analysis in the response,
side by side with theirs.

---

## 3. Gray et al. 2023 — the reviewer is RIGHT, despite the title

**Gray, W. R., de Lavergne, C., Jnglin Wills, R. C., Menviel, L., Spence, P., Holzer, M.,
Kageyama, M., & Michel, E. (2023).** Poleward Shift in the Southern Hemisphere Westerly Winds
Synchronous With the Deglacial Rise in CO₂. *Paleoceanography and Paleoclimatology* **38**(7),
e2023PA004666. DOI 10.1029/2023PA004666.

**Verified from the full text PDF, not just the abstract.**

### Resolving the apparent contradiction

The title says *poleward*; the reviewer says the proxies show *equatorward and weakened*. **Both
are correct, and they are not in conflict.** The title describes the **deglacial transition**
(20 ka → 10 ka, winds migrating poleward as CO₂ rises). The reviewer describes the **LGM state**
(glacial winds sitting equatorward of, and weaker than, the Holocene). Same paper, same result,
two different things being described.

**Do not push back on the reviewer here.** Their characterisation of the LGM state is exactly what
the paper concludes.

### What the paper concludes for the LGM state

Conclusion section, verbatim:

> "We infer a 4.8° (2.9–7.1°, 95% CI) equatorward shift and a ∼25% weakening of the westerlies
> during the LGM relative to the mid-Holocene."

Wind strength, quantified (Section 3.5):

> "Our reconstructed equatorward shift in the wind latitude implies a weakening of the peak
> westerly wind stress by 0.034 N m⁻² (about 25%) during the LGM relative to the mid Holocene,
> resulting in a LGM wind strength of 0.106 (0.085–0.12, 95% CI) N m⁻², assuming mid-Holocene wind
> strength is equal to the modern climatology (0.14 N m⁻²)."

A separate, larger figure for the raw wind-latitude reconstruction also appears: "We infer a 6.3°
(4.3–8.7°, 95% CI) equatorward shift in the wind latitude during the LGM (20 ka) relative to
[10 ka]". The headline 4.8° is LGM relative to **mid-Holocene (6.5 ka)**. Use 4.8° and cite the
reference state explicitly, since the paper reports several.

Method: planktic foraminiferal δ¹⁸O compilation across the Southern Ocean → SST front latitude →
wind latitude, via emergent relationships in a PMIP3/4 model ensemble.

### The models-fail-to-capture-it claim — verified and quantified

Abstract: "Climate models from the Palaeoclimate Modeling Intercomparison Project substantially
underestimate this inferred equatorward wind shift."

Body, more precisely:

> "the magnitude of the inferred LGM to mid-Holocene wind shift is substantially greater than that
> predicted by any of the models within the PMIP3/4 ensemble between LGM and preindustrial states"

Introduction, on the state of the field: models "show a relatively clear and consistent signal of
an equatorward shift in the Northern Hemisphere surface westerlies under glacial forcings ... they
show little consistency in the magnitude or sign of change in the Southern Hemisphere."

**"any of the models"** is a strong, quotable claim — no PMIP3/4 member reproduces the
reconstructed magnitude.

### One finding directly useful to us

Gray et al. explicitly tested whether Antarctic sea ice drives the SST-front/wind relationship:

> "we find no correlation between Antarctic sea ice extent and the SST front latitude in the model
> ensemble (Figure 2e; R² = 0.02), nor do we find a correlation between sea ice extent and the
> wind latitude (R² = 0.03)."

Useful for us in two ways. It supports treating the wind bias as **independent of** any sea-ice
bias in our simulations, so a wind-position error does not automatically invalidate our sea-ice
result. And their transient experiments show equatorward-shifted winds **reduce** overturning
below 2 km — relevant if a reviewer asks how a wind bias would propagate into our WMT numbers.

### Suggested response posture

Accept the reviewer's point. State that our simulated westerlies likely share the PMIP-wide bias
(too little equatorward shift, too little weakening at LGM), note that this is a documented
ensemble-wide limitation rather than a defect specific to AWI-ESM2, and — if we can check it —
report our own LGM-minus-PI wind-latitude change against the 4.8° reconstruction. Then argue
scope: our claims rest on the **surface buoyancy flux decomposition**, and the Gray et al. R² =
0.02–0.03 result indicates the wind-latitude bias does not propagate directly onto sea-ice extent.

---

## 4. Lhardy et al. 2021 — PMIP failures and excessive convection

**Lhardy, F., Bouttes, N., Roche, D. M., Crosta, X., Waelbroeck, C., & Paillard, D. (2021).**
Impact of Southern Ocean surface conditions on deep ocean circulation during the LGM: a model
analysis. *Climate of the Past* **17**(3), 1139–1159. DOI 10.5194/cp-17-1139-2021.

### On PMIP models failing on glacial Southern Ocean sea ice

Verbatim: "PMIP models struggle to reproduce the glacial sea-ice extent suggested by sea-ice proxy
data and especially its seasonality."

On the AMOC side: "most models from previous Paleoclimate Modelling Intercomparison Project (PMIP)
phases showing a tendency to simulate a strong and deep North Atlantic Deep Water (NADW) instead
of the shoaling inferred from proxy records". Only a minority of PMIP2 models produce NADW
shoaling; most PMIP3 models produce an intensified and deepened NADW.

**Not verified:** an exact count or named list of PMIP models failing specifically on Southern
Ocean sea ice. The paper states the tendency qualitatively and by PMIP phase. **Do not cite a
number of models for this claim.** If the response needs a hard count, Lhardy et al. is not the
source for it.

### On excessive open-ocean deep convection at LGM — the important part

This is the study's central result and it is directly usable:

> "the only simulation which does not display a much deeper NADW is obtained by parameterizing the
> sinking of brines along Antarctica, a modeling choice reducing the open-ocean convection in the
> Southern Ocean."

And the warning that cuts against the intuitive fix:

> "colder conditions rather tend to intensify the Southern Ocean open-ocean convection, a process
> which leads to inaccurate AABW properties."

Their framing of the underlying problem: "Inadequate representation of surface conditions, driving
deep convection around Antarctica, may explain inaccurately simulated bottom water properties in
the Southern Ocean." Overall conclusion — "the importance of the representation of convection
processes, which have a large impact on the water mass properties, while the choice of boundary
conditions appears secondary."

### Numbers verified

- Simulated seasonal sea-ice range: **65–94%** of the proxy-inferred range
- Southern Ocean warm bias: **2–6 °C** vs MARGO
- Pre-industrial AMOC maximum: **10.1 and 11.2 Sv** at **1225 m** (observed range 13.5–20.9 Sv)
- LGM sea-ice extent estimates used: **~10.2 × 10⁶ km²** (summer), **~32.9 × 10⁶ km²** (winter)

Design: nine iLOVECLIM LGM simulations varying ice-sheet boundary conditions and modelling choices
for sea-ice export, brine formation, and freshwater input.

### Relevance to us

This is the strongest support for the "known, community-wide problem" framing. If our LGM run
convects too readily in the open ocean, Lhardy et al. establish that (i) this is a general LGM
modelling problem, not ours alone, (ii) colder glacial forcing actively *worsens* it, so it is not
fixed by better boundary conditions, and (iii) the effective remedy is a brine-sinking
parameterisation, which is a model-development answer rather than a flaw in the WMT diagnostic.

---

## 5. Millet et al. 2025 — ideal age equilibration

**Millet, B., de Lavergne, C., Gray, W. R., Éthé, C., Madec, G., Holzer, M., DeVries, T.,
Gebbie, G., & Roche, D. M. (2025).** Deep Ocean Ventilation: A Comparison Between a General
Circulation Model and Data-Constrained Inverse Models. *Journal of Advances in Modeling Earth
Systems* **17**(7), e2024MS004914. DOI 10.1029/2024MS004914. Verified against full OA text.

### ⚠️ The reviewer's claim is a misreading — handle carefully

The "factor of two" in this paper is about **filling time**, not ideal-age spin-up:

> "we find that the time needed to fill (or renew) 99% of the ocean volume is roughly twice the
> maximum ideal age."

The paper explicitly distinguishes the two timescales:

> "ideal age equilibration requires the uniform interior source to be balanced by transport from
> the surface sink, whereas dye tracer equilibration requires unbroken transport from surface
> source to surface sink via the ocean interior. Hence, dye tracer equilibration time is more akin
> to residence time, which is the sum of time since last surface contact (ideal age) and time to
> next surface contact."

So the paper does **not** state that ideal age requires a spin-up of at least twice the maximum
ideal age. I searched the full text and the wider literature and **could not find any paper making
that claim in the reviewer's form.** Do not cite Millet et al. for it.

### What the paper does say — all usable, all verbatim

> "As with the dye concentrations, the ideal age tracer is not fully equilibrated after 3,000 years
> of simulation."

> "It takes almost 6,000 years to fill over 99% of the global ocean volume in REF, compared to only
> about 3,000 years in TMI and 4,000 years in OCIM."

> "over 3,000 years are needed to entirely renew the mid-depth Pacific"

> "we estimate that near complete (99%) renewal of the mid-depth (1–3 km) Pacific requires between
> 3,000 and 4,000 years."

Best single sentence for a spin-up caveat:

> "The large filling times found in the three models, exceeding 3,000 years, underscore the need
> for multi-millennial climate simulations to achieve climate states without artificial deep ocean
> drift."

Additional detail: in REF the maximum zonal-mean Pacific ideal age is ~2,500 years while the
shadow zone resists complete ventilation for ~6,000 years — the concrete illustration of the
factor of two. Because ideal age was still unequilibrated at year 3,000, the authors extrapolated
to steady state and validated against a 6,000-year run to within **2.5%**.

The factor-of-two relation is attributed to **Holzer, M., & Primeau, F. W. (2008)**, *JGR* **113**,
C01018, DOI 10.1029/2006JC003976. I did not read that paper, so I cannot confirm whether it states
the relation in the form the reviewer wants.

### Suggested response posture

Accept the underlying concern — our age tracer is very likely not fully equilibrated — and cite the
"multi-millennial simulations ... without artificial deep ocean drift" sentence, which supports the
reviewer's point accurately. Then state our actual spin-up length and quantify the residual drift
if we can. Correcting the "twice the maximum ideal age" phrasing is optional; if we do correct it,
do so lightly, since the reviewer's substantive concern is legitimate even though the specific
formulation is not what the paper says.

---

## 6. Heuzé 2013 and 2021 — how CMIP models make AABW

### Heuzé 2021 (CMIP6)

**Heuzé, C. (2021).** Antarctic Bottom Water and North Atlantic Deep Water in CMIP6 models.
*Ocean Science* **17**(1), 59–90. DOI 10.5194/os-17-59-2021.

**35 CMIP6 models.** Headline result, verbatim:

> "Several CMIP6 models are correctly forming AABW via shelf processes, but **28 models in the
> Southern Ocean and all 35 models in the North Atlantic** form deep and bottom water via
> open-ocean deep convection too deeply, too often, and/or over too large an area."

So **28 of 35 (80%)** in the Southern Ocean. Also verbatim:

> "Most models form their AABW via open-ocean deep convection. Even the models that seem to
> represent shelf processes accurately exhibit open-ocean deep convection."

On the few that get it right:

> "In no model is there any (obvious) shelf export in the Weddell Sea. INM-CM5 (terrain following,
> high horizontal resolution model) and the two NorESM2 (hybrid isopycnic models) are the only ones
> forming AABW accurately via shelf processes, in the Ross sector only."

Bias numbers: 10 models have negligible bottom density bias (RMSE < 0.05 kg m⁻³); 12 more are
acceptable (< 0.1 kg m⁻³). Southern Ocean multi-model mean is **0.06 kg m⁻³ too light, 0.85 °C too
warm, 0.02 psu too fresh**. Multi-model mean AABW: T = −0.45 ± 0.73 °C (ref −0.88 °C),
S = 34.606 ± 0.154 (ref 34.641). "The four CESM2 models with their overflow parameterisation are
among the most accurate models."

**⚠️ Citation-accuracy point: no AWI model is among Heuzé's 35.** I checked the full Table 1 list
(ACCESS-CM2 … UKESM1-0-LL); AWI-CM-1-1-MR and AWI-ESM do not appear. Cite this as the general state
of CMIP-class models — **not** as an independent evaluation of our model.

### Heuzé 2013 (CMIP5)

**Heuzé, C., Heywood, K. J., Stevens, D. P., & Ridley, J. K. (2013).** Southern Ocean bottom water
characteristics in CMIP5 models. *Geophysical Research Letters* **40**(7), 1409–1414.
DOI 10.1002/grl.50287.

**15 CMIP5 models.** Abstract, verbatim:

> "Bottom properties are reasonably accurate for half the models. Ten models create dense water on
> the Antarctic shelf, but it mixes with lighter water and is not exported as bottom water as in
> reality. Instead, most models create deep water by open ocean deep convection, a process
> occurring rarely in reality. Models with extensive deep convection are those with strong
> seasonality in sea ice. Optimum bottom properties occur in models with deep convection in the
> Weddell and Ross Gyres."

Two things worth carrying into the response. **10 of 15** CMIP5 models form shelf dense water that
fails to export as bottom water. And "Models with extensive deep convection are those with strong
seasonality in sea ice" — a direct link between sea-ice seasonality and the convection bias, which
connects to Chen et al.'s glacial argument that the seasonal cycle gives way to export-driven
year-round formation.

### Combined use

Together these give the CMIP5→CMIP6 trajectory: the open-ocean-convection bias is near-universal,
persistent across two model generations, and improving only slowly ("There are clear improvements
since CMIP5: several CMIP6 models correctly represent or parameterise Antarctic shelf processes,
fewer models exhibit Southern Ocean deep convection ... However, more improvements are required").
This is the strongest available support for "our model shares a documented, community-wide
limitation."

---

## 7. AWI-CM / FESOM Southern Ocean biases — what is citable, and what is not

### ⚠️ The honest gap, stated first

**I could not find any published AWI-CM / AWI-ESM / FESOM paper that explicitly states the model
forms AABW by open-ocean deep convection rather than by shelf processes.** I searched the model
description and evaluation papers below plus targeted searches on the FESOM/AWI-CM Weddell polynya
issue. Rackow et al. 2019 mentions denser water production around Antarctica without naming the
mechanism.

**Do not attribute that statement to these references.** If we say our model forms dense water by
open-ocean convection, it must be presented as **our own diagnosis from our own output**, supported
by the *general* CMIP finding (Heuzé) rather than by an AWI-specific citation. This is the single
most important accuracy constraint in this document.

### Rackow et al. 2019 — the most useful Southern Ocean statements

**Rackow, T., Sein, D. V., Semmler, T., Danilov, S., Koldunov, N. V., Sidorenko, D., Wang, Q., &
Jung, T. (2019).** Sensitivity of deep ocean biases to horizontal resolution in prototype CMIP6
simulations with AWI-CM1.0. *Geoscientific Model Development* **12**(7), 2635–2656.
DOI 10.5194/gmd-12-2635-2019. Verified from the open-access full text.

Verbatim:

> "Furthermore, the Southern Ocean density structure is equally improved with locally explicitly
> resolved eddies compared to parameterized eddies."

> "It is related to the fact that the eddy parameterization (GM) has difficulties in representing
> the slope of the isopycnals, which is determined by the counteracting effects of Ekman pumping
> and eddy transport."

> "Already at MR, the simulated isopycnal slope is about halved compared to LR and much closer to
> the observed slope ... with strongly reduced temperature biases, suggesting that the explicitly
> resolved eddies outperform the eddy parameterization."

On deep water production near Antarctica — note the mechanism is *not* specified:

> "the slight drift in HR towards colder temperatures in the 3000–5000 m range is due to a
> production of denser waters around Antarctica, coinciding with a stronger deep overturning cell
> in this model configuration."

Context on why deep biases matter: "the mean absolute error in deeper ocean layers is larger than
the interannual variability ... It is also larger than the climate change signal as determined from
RCP8.5 and RCP4.5 emission scenarios."

**Useful framing for us:** at the ~1° (CMIP5-type) resolution class that our paleo runs use, the
Southern Ocean density structure and isopycnal slopes are known to be biased, and that bias is
resolution-dependent rather than a flaw in the analysis method. This is an honest, citable
acknowledgement of a resolution limitation.

**Not verified in this paper:** specific Antarctic sea-ice extent bias numbers, mixed layer depth
bias numbers, and any Weddell polynya discussion.

### Sidorenko et al. 2019 — our own group's evaluation paper

**Sidorenko, D., Goessling, H. F., Koldunov, N. V., Scholz, P., Danilov, S., Barbi, D., Cabos, W.,
Gurses, O., Harig, S., Hinrichs, C., Juricke, S., Lohmann, G., Losch, M., Mu, L., Rackow, T.,
Rakowsky, N., Sein, D., Semmler, T., Shi, X., Stepanek, C., Streffing, J., Wang, Q., Wekerle, C.,
Yang, H., & Jung, T. (2019).** Evaluation of FESOM2.0 Coupled to ECHAM6.3: Preindustrial and
HighResMIP Simulations. *Journal of Advances in Modeling Earth Systems* **11**(11), 3794–3815.
DOI 10.1029/2019MS001696.

The natural evaluation citation for our model configuration, and it includes **X. Shi, G. Lohmann
and C. Stepanek** as co-authors — so citing it for known biases is self-consistent rather than
defensive.

**Not verified:** I did not retrieve specific Southern Ocean bias numbers (sea-ice extent, MLD,
stratification) from this paper. If the response needs a concrete number from it, someone should
open the PDF and pull the relevant figure/table. Cite it for general model provenance and
evaluation unless specific numbers are confirmed.

### Sidorenko et al. 2021 — methodological precedent worth citing

**Sidorenko, D., Danilov, S., Streffing, J., Fofonova, V., Goessling, H. F., Scholz, P., Wang, Q.,
Androsov, A., Cabos, W., Juricke, S., Koldunov, N., Rackow, T., Sein, D. V., & Jung, T. (2021).**
AMOC Variability and Watermass Transformations in the AWI Climate Model. *Journal of Advances in
Modeling Earth Systems* **13**(10), e2021MS002582. DOI 10.1029/2021MS002582.

**Same model family, same WMT method.** Valuable as precedent that water mass transformation
analysis is an established, validated diagnostic in AWI-CM — useful if a reviewer questions
whether the method is appropriate for this model. Content beyond title and metadata **not
verified**; recommend a look before citing for any specific claim.

### Semmler et al. 2020 and Danilov et al. 2017

**Semmler, T., et al. (2020).** Simulations for CMIP6 With the AWI Climate Model AWI-CM-1-1.
*JAMES* **12**(9), e2019MS002009. DOI 10.1029/2019MS002009. — standard CMIP6 configuration citation.

**Danilov, S., Sidorenko, D., Wang, Q., & Jung, T. (2017).** The Finite-volumE Sea ice-Ocean Model
(FESOM2). *Geoscientific Model Development* **10**(2), 765–789. DOI 10.5194/gmd-10-765-2017. —
the model description paper. Also the FESOM2 reference cited by Chen et al. 2025, which is worth
noting: same ocean core, different configuration.

Southern Ocean bias specifics from both: **not verified.**

---

## 8. Orsi et al. 1999 and Toggweiler & Samuels 1995 — shelf-based AABW formation

### Orsi, Johnson & Bullister 1999

**Orsi, A. H., Johnson, G. C., & Bullister, J. L. (1999).** Circulation, mixing, and production of
Antarctic Bottom Water. *Progress in Oceanography* **43**(1), 55–109.
DOI 10.1016/S0079-6611(99)00004-X.

Method: a **CFC budget** for the AABW layer offshore of the 2500 m isobath, with density at the
Drake Passage sill depth separating circumpolar deep water from denser bottom water of Antarctic
margin origin.

Production rates, all verbatim from the abstract:

> "The resulting total AABW production rate is about **8 Sv**, which is a conservative figure that
> neglects the loss of CFC-bearing waters across the top isopycnal in recent years, whereas about
> **9.5 Sv** is calculated assuming a well-mixed bottom layer."

> "A typical basin-wide rate of deep upwelling of 3×10⁻⁷ m s⁻¹ requires **10 Sv** ... of
> newly-formed AABW to sink down the slope around Antarctica."

A non-uniform upwelling field yields **~12 Sv** exported across the top isopycnal. So: 8 Sv is the
headline conservative CFC estimate; 8–12 Sv spans the reported range.

On shelf origin — the reason this is the classic shelf-process citation:

> "Over the shelf regime multiple localized sources of specific AABW types contribute to the
> abyssal layer of the adjacent Antarctic basins. Characteristics of these dense bottom waters
> reflect closely those observed on the parent Shelf Water mass."

**⚠️ Not verified:** any per-sector (Weddell / Ross / Adélie) breakdown in Sv. The abstract reports
a **combined** circumpolar rate. Do not cite Orsi et al. 1999 for a per-sector split without
checking the paper body — commonly used per-source numbers usually come from Orsi et al. 2002 or
Jacobs 2004. Also not verified: any explicit quantitative shelf-versus-open-ocean-convection
fraction.

**Use:** the observational benchmark that real AABW forms from dense shelf water descending the
slope, at ~8–10 Sv — the reference point against which an open-ocean-convection model is judged.

### Toggweiler & Samuels 1995 — ⚠️ two papers, and the reviewer's framing is off

**The one about Antarctic shelf processes:**

**Toggweiler, J. R., & Samuels, B. (1995).** Effect of Sea Ice on the Salinity of Antarctic Bottom
Waters. *Journal of Physical Oceanography* **25**(9), 1980–1997.
DOI 10.1175/1520-0485(1995)025<1980:EOSIOT>2.0.CO;2

Verbatim: "Brine rejection during the formation of Antarctic sea ice is known to enhance the
salinity of dense shelf waters in the Weddell and Ross Seas. As these shelf waters flow off the
shelves and descend to the bottom, they entrain ambient deep water to create new bottom water."

They find on-shelf/off-shelf salinity changes in the Weddell and Ross Seas are "fairly small,
**0.15–0.20 salinity units**", equivalent to the salt drained from **≤0.50 m** of new sea ice per
year. Their conclusion is the part that matters for us:

> "salt from sea ice is probably not a major influence on the salinity of Antarctic bottom waters.
> Predicted salinities in ocean GCMs are too fresh because of circulation deficiencies, not because
> of inadequate boundary conditions. Models that employ large salinity modifications near
> Antarctica run the risk of grossly distorting the processes of deep-water formation."

**The other 1995 paper — the trap:**

**Toggweiler, J. R., & Samuels, B. (1995).** Effect of Drake Passage on the global thermohaline
circulation. *Deep-Sea Research Part I* **42**(4), 477–500. DOI 10.1016/0967-0637(95)00012-U

This is the wind-driven upwelling / Drake Passage paper, and it is far more cited — it is what
"Toggweiler & Samuels 1995" usually means without qualification. It concerns global overturning
geometry, not shelf processes. (Its argument as commonly summarised is **not verified** from a
primary source here; no abstract was retrievable.)

**On the reviewer's "ice shelf melt" framing.** Neither paper treats **ice shelves** (floating
glacier tongues) or ice shelf melt. The JPO paper is about **sea ice** brine rejection — a routine
conflation, but a real one. If the reviewer's substantive concern is ice-shelf meltwater
suppressing AABW formation, Toggweiler & Samuels is the wrong citation and the relevant literature
is elsewhere (e.g. Silvano et al. 2018; Williams et al. 2016 on Prydz Bay).

**Suggested posture:** respond on the sea-ice/brine reading, which is defensible and genuinely
relevant to our haline forcing discussion, and note the ice-shelf distinction politely rather than
citing the paper for a claim it does not make. Their warning about GCM salinity biases arising from
"circulation deficiencies, not ... inadequate boundary conditions" is directly on point for a model
that convects in the open ocean instead of exporting shelf water.

---

## Quick cross-reference: which reference answers which reviewer point

| Reviewer point | Use |
|---|---|
| "What does WMT add beyond Chen et al. 2025?" | §1 — quantitative attribution; 5 coupled states vs 2 ocean-only |
| "Your PI is thermally dominated, Pellichero finds haline" | §2 — check domain vs September ice edge before conceding |
| "Glacial westerlies were equatorward and weaker" | §3 — reviewer is right; PMIP-wide bias, R²=0.02 sea-ice decoupling |
| "Models fail on glacial SO sea ice / over-convect" | §4 — Lhardy; colder forcing worsens convection |
| "Ideal age not equilibrated" | §5 — accept concern, cite the multi-millennial-drift sentence |
| "Your model convects in the open ocean" | §6 — Heuzé 28/35 CMIP6, 10/15 CMIP5; **not** an AWI-specific citation |
| "Known AWI-CM Southern Ocean biases" | §7 — Rackow resolution/isopycnal slope; **no** AWI open-convection citation exists |
| "Real AABW forms on shelves" | §8 — Orsi ~8–10 Sv; Toggweiler & Samuels on GCM salinity biases |

---

## Summary of what could NOT be verified

Listed so nothing here gets cited beyond what the sources support.

1. **Chen et al. 2025 body text** — Wiley/Cloudflare blocks automated access. Abstract, author list,
   and archived diagnostic variables are confirmed; the figure-by-figure method description is
   inferred from the Zenodo supplement and indexed text.
2. **Orsi et al. 1999 per-sector Sv breakdown** (Weddell / Ross / Adélie) — not in the abstract.
3. **Any AWI-CM/AWI-ESM/FESOM statement that the model forms AABW by open-ocean convection** — no
   such published statement found.
4. **Southern Ocean bias numbers in Sidorenko 2019, Semmler 2020, Sidorenko 2021** — papers verified
   bibliographically; specific sea-ice/MLD/stratification numbers not extracted.
5. **A count of PMIP models failing on Southern Ocean sea ice in Lhardy 2021** — stated
   qualitatively and by phase, not as a number.
6. **Any paper stating ideal age needs a spin-up of twice the maximum ideal age** — searched, not
   found; Millet et al. say this of *filling time*.
7. **Toggweiler & Samuels 1995 (Drake Passage) argument** — no abstract retrievable from a primary
   source.
