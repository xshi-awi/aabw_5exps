# Response to Reviewers

**Manuscript:** Glacial–interglacial regime shift in Southern Ocean dense water formation
**Authors:** Xiaoxu Shi, Jiping Liu, Hu Yang, Chaoyuan Yang, Gerrit Lohmann

---

We thank the editor and the four reviewers for the care they have taken with this manuscript. The
reports are detailed and constructive, and several of them identify a problem that we had handled
too lightly in the submitted version: the disagreement between our pre-industrial result and
observation-based transformation estimates for the modern Southern Ocean. We took that criticism
seriously and spent most of the revision on it. Rather than defend the original one-sentence
explanation, we ran the diagnostic tests that Reviewers 2 and 3 proposed. The outcome is more
informative than we expected, and it has changed how we frame the paper.

In brief, the disagreement has two separable causes. The first is the integration domain: 42.7% of
the area we integrated over lies outside the winter sea ice edge and is therefore water that
Pellichero et al. do not analyse. The second, and larger, is the density range: our simulated
transformation peaks at densities corresponding to intermediate and mode water, whereas the
haline-dominated cell those authors describe lies at bottom water densities that our simulated
surface fluxes barely reach. Where the two analyses do overlap in density, they agree in sign.
A residual discrepancy remains after both effects are accounted for, and it reflects a real
limitation of the model, namely that it forms its pre-industrial dense water by open-ocean
convection in the Weddell gyre rather than by dense shelf water overflow. We now state this
plainly, quantify it, and bound what we claim on the basis of it.

We have also added the analyses that Reviewer 3 requested, showing the overturning streamfunction
directly, ideal age throughout the global ocean, and a fuller treatment of the Southern Annular
Mode; discussed the study by Chen et al. (2025) that Reviewer 2 correctly noted was missing; and
worked through every specific comment. Five new figures have been added, along with a boundary
condition table and a Source Data file.

For convenience the principal new figures are reproduced in this letter, so that the reviewers can
assess them without consulting the manuscript.

Reviewer comments are reproduced in full below in italics, each followed by our response. Line
numbers in our responses refer to the revised manuscript unless stated otherwise. All changes are
marked in the accompanying tracked-changes file.

---

# Reviewer #1

> *The manuscript presents a comprehensive investigation of Southern Ocean dense water formation
> across a range of glacial and interglacial climates using the AWI-ESM2 Earth system model. ...
> Overall, I found the study scientifically interesting and suitable for publication after major
> revisions. My comments below are primarily intended to strengthen the robustness of the
> conclusions and improve the clarity of the presentation.*

We thank the reviewer for this assessment and for the unusually thorough list of specific comments,
which has improved the presentation considerably.

## General comments

### G1. Model evaluation and confidence in simulated dense water formation

> *The conclusions of this study rely entirely on a single Earth system model. ... the manuscript
> would benefit from a more thorough discussion of how well AWI-ESM2 represents present-day Southern
> Ocean dense water formation and Antarctic Bottom Water formation. The manuscript briefly states
> that the PI simulation produces comparable WMT magnitudes to observational estimates, but Figure A4
> only shows the model results, making it difficult for the reader to evaluate this statement.*

We agree, and this concern overlaps with the central criticism from Reviewers 2 and 3. Our response
has three parts.

First, we now quantify how the simulated pre-industrial transformation compares with the
observation-based estimate of Pellichero et al. (2018), matching their analysis domain rather than
comparing across different domains. Their sector is the region enclosed by the September 15% sea
ice contour. Reproducing that definition in our model gives an area within 2.5% of theirs, so the
comparison is now like-for-like. The result is shown in the new Fig. 8 and discussed in the
Discussion. Restricting the integration to the ice-covered sector raises the haline share, as
expected, moving the thermal fraction of the dense-class transformation from 69% to 61% and roughly
doubling the sea ice contribution at the transformation maximum from 0.86 to 1.84 Sv.

![](letter_figs/figR1_wmt_domain_sensitivity.png)

**Figure R1.1 (manuscript Fig. 8).** Transformation integrated south of 60°S, the domain used in
the submitted version (top row), and over the seasonal sea ice zone alone, which reproduces the
domain of Pellichero et al. to within 2.5% in area (bottom row). The thermal share at the
transformation maximum and the area of each domain are given in each panel. The haline share
increases in the restricted domain, as expected, but the contrast between thermally dominated
interglacials and haline dominated glacials is unchanged.

We also diagnosed directly where the model forms its dense water, which bears on how far the
pre-industrial state can be trusted.

![](letter_figs/figR6_convection_sites.png)

**Figure R1.2 (new analysis).** (a-e) Winter mixed layer depth for the five states, with the 400 m
contour in orange. (f) In the pre-industrial state the Weddell sector accounts for 70% of the area
with a mixed layer deeper than 400 m and 97% of the area deeper than 600 m, and the convection sits
in the interior of the gyre near 68°S rather than over the shelf. (g) The deep-convection area
contracts to roughly one fifth of its pre-industrial value in the glacial states. (h) The deepest
winter mixed layer shoals from 991 m in PI to 819 m and 645 m in the glacials.

Second, and more importantly, we found that the density range matters more than the domain.
Converting our simulated surface properties to the neutral density coordinate used in that study,
our thermally dominated maximum sits near γ_n ≈ 27.3 kg m⁻³, whereas their haline-dominated lower
cell occupies γ_n = 27.9–28.8 kg m⁻³. In the densest classes our surface fluxes do populate
(σ₂ ≥ 37.0 kg m⁻³), our pre-industrial transformation is already haline dominated, 0.79 Sv from sea
ice against 0.06 Sv from heat. The two analyses are therefore largely describing different water
masses, and where they overlap they agree.

Third, we now state the residual model limitation explicitly rather than alluding to it. We
diagnosed where the model actually forms its dense water: in the pre-industrial simulation, the
Weddell sector accounts for 70% of the area with a winter mixed layer deeper than 400 m south of
55°S and 97% of the area deeper than 600 m, centred near 68°S in the interior of the gyre rather
than over the shelf. This is open-ocean convection, not the shelf overflow pathway that operates in
the real ocean. We note that this is a limitation shared across the current model generation rather
than particular to AWI-ESM2, citing Heuzé (2021), who finds that 28 of 35 CMIP6 models form deep
and bottom water by open-ocean convection that is too deep, too frequent, or too extensive, with no
model showing convincing shelf export in the Weddell Sea. We also now state that this configuration
has no ice shelf cavities, so basal melt beneath floating ice is absent by construction.

We have deliberately not asserted that published AWI-ESM evaluation papers document this convection
behaviour, because they do not; the diagnosis above is our own, from our own output.

### G2. Boundary conditions

> *The description of the prescribed boundary conditions for the paleoclimate simulations could be
> expanded. In particular, the glacial simulations use the GLAC-1D ice-sheet reconstruction, but it
> is not entirely clear how the ice-sheet changes are implemented in the model ... A summary table
> listing the prescribed greenhouse gas concentrations, orbital parameters, and ice-sheet boundary
> conditions for each experiment would also improve the readability of the Methods section.*

Added. Table 1 in the Methods now lists the greenhouse gas concentrations, orbital parameters, and
ice sheet configuration for all five experiments; the values are those actually used in the runs.
We have also expanded the surrounding text to describe how GLAC-1D enters the model, which we agree
was underspecified. It sets the ice sheet extent and surface elevation and the associated land-sea
mask and orography for the atmosphere and land surface, the river routing, and the vegetation
distribution, and it determines the ocean bathymetry and coastline. The last of these requires a
separate unstructured ocean mesh for each glacial period, and we use meshes built for 21 ka and
38 ka, while PI, MH and LIG share the modern-geometry mesh. We also now state that global mean
ocean salinity is raised in the glacial runs to account for water stored in the ice sheets, and
that no prescribed ice sheet meltwater flux is applied in any experiment.

### G3. Clarifying several interpretations

> *Most interpretations are convincing, although a few statements could be presented more carefully
> or supported more explicitly. For example, some descriptions of the role of the Southern Annular
> Mode appear stronger than the presented evidence suggests, and several discussions would benefit
> from more direct references to the relevant figure panels.*

We have moderated the Southern Annular Mode claims. The most overstated passage was the suggestion
that the simulated coupling implies a route to marine proxy reconstruction; we have replaced it
with a more careful statement noting that the accumulation rates and chronological precision
required are rarely available around the Antarctic margin, a point Reviewer 3 also raised. We have
also removed the claim that the circumpolar picture holds uniformly and now state explicitly where
it does not, in particular the Ross Sea. Panel-level figure references have been added throughout,
as requested in this comment and in comment 21.

### G4. Model limitations

> *... the discussion would benefit from a brief consideration of the known biases and limitations
> of AWI-ESM2, particularly regarding Southern Ocean sea ice, mixed-layer depths, and Antarctic
> Bottom Water formation.*

Addressed in the Discussion, as described under G1. We additionally discuss two limitations that
bear on the interpretation. The first is the glacial wind field: our simulated westerlies shift
poleward and strengthen, whereas proxy reconstructions infer an equatorward shift of about 4.8° and
a weakening of about 25% at the Last Glacial Maximum relative to the mid-Holocene (Gray et al.,
2023). The second is that the ideal age fields are not fully equilibrated after 1000 years, so the
simulated ages are lower bounds. Both are now stated in the text.

### G5. Latitude and longitude labels, and subpanel references

> *Please consider adding latitude and longitude labels to all map figures where appropriate. In
> addition, referring to specific figure subpanels throughout the text ... would make the discussion
> easier to follow.*

Subpanel references have been added throughout the Results and Discussion. On the map labels: the
polar stereographic panels now carry labelled latitude circles, and the ventilation age maps carry
labelled meridians as well. On the fifteen-panel composite figures we label the latitude circles
only, because a full graticule with meridian labels on panels of that size was illegible in
trial versions and crowded the panel margins. We are happy to add complete labelling if the
reviewer prefers, or to enlarge those figures to accommodate it.

## Specific comments

**1.** *LIG, LGM and MIS should be capitalised.* — Corrected throughout.

**2.** *Line 21-22: The transition to SAM is quite abrupt.* — The abstract now introduces the
Southern Annular Mode through the mechanism that connects it to the mean state, noting that because
the balance between the thermal and haline pathways is set by how much of the surface is ice
covered, it also governs the response to internal atmospheric variability.

**3.** *Line 24: "same mechanisms" sounds vague ... The abstract should be understandable on its
own.* — Replaced with the explicit statement that the mode works "through heat loss when the ocean
is open and through brine rejection when it is ice covered".

**4.** *Lines 25-27: ... additional context is needed before introducing the deep ocean ventilation
results.* — The abstract now bridges from surface forcing to ventilation via the overturning
circulation, stating that weaker glacial transformation coincides with a collapse of the simulated
abyssal cell from about 10 to 2 Sv, before the ages are given. This also incorporates the new
overturning analysis requested by Reviewer 3.

**5.** *Lines 102-105: SAM is introduced only briefly here. Given its importance ... please consider
providing a slightly more detailed introduction.* — The introduction now defines the mode as the
leading mode of extratropical Southern Hemisphere variability, describes what its positive phase
does to the westerlies and hence to Ekman divergence, air-sea heat exchange and sea ice export, and
cites the observational literature linking it to bottom water changes.

**6.** *Line 125: typo: 'more'.* — Corrected ("moer" → "more").

**7.** *Line 127: Figure numbering; A3 can't be called before A1 and A2.* — The appendix figures
have been reordered so that they are cited in numerical order.

**8.** *Line 129: 'Here,'.* — Corrected.

**9.** *Line 140: As a result, the deep MLD signals ... Also, "vanish" is perhaps not the most
scientific wording.* — Rephrased to "are strongly reduced and become confined to narrow coastal
bands and polynyas", and we now quantify the reduction rather than describing it qualitatively.

**10.** *Line 141: These signals are not particularly visible even in the coastal regions (the
darker blue shading in LGM is at ~200m, so considerably shallower than the interglacials).* — This
is a fair objection and we have addressed it with numbers. Summed south of 55°S, the area with a
winter mixed layer deeper than 400 m falls from 7.5×10¹¹ m² in PI to 1.7×10¹¹ m² in LGM and
1.6×10¹¹ m² in MIS3, the maximum mixed-layer depth decreases from 991 m to 819 m and 645 m, and
mixed layers deeper than 600 m essentially disappear in both glacial states. The text now says
explicitly that the residual glacial signals are genuinely shallower than their interglacial
counterparts. Panels (g) and (h) of Figure R1.2 above make the point quantitatively: both the area
of deep convection and the depth of the deepest mixed layer fall in the glacial states, so the
reviewer's reading of the shading is correct and is now reflected in the text.

The reviewer's observation also pointed to a presentation problem, which we have fixed. On the
shared 0–400 m scale of the original figure the glacial panels are almost featureless, because the
99th percentile of the glacial winter mixed layer south of 55°S is only 280 m (LGM) and 296 m
(MIS3). Replotting those two states on their own 0–300 m scale makes the coastal cells visible.

![](letter_figs/figR7_mld_glacial_zoom.png)

**Figure R1.3 (new supplementary panel).** LGM and MIS3 winter mixed layer depth on the original
0–400 m scale (top) and on a tightened 0–300 m scale with the 200 m contour marked (bottom). The
tightened scale resolves discrete deep cells along the coast, in the Weddell, Prydz Bay and Ross
sectors, which is where the glacial dense water is produced. This is the figure we now use in the
supplement.

**11.** *Line 134-141: Fig.1 i-l subpanels are not discussed at all.* — A paragraph discussing the
surface density panels has been added. It notes that the glacial density anomalies follow the
salinity field rather than the temperature field, because the thermal expansion coefficient is
small at the near-freezing temperatures south of the ice edge, and points out the
density-compensated LIG anomaly and the weak negative LGM anomalies in the Atlantic-Indian sector.

**12.** *Line 144-146: The LIG also exhibits similar behaviour, although the changes are smaller in
some basins. Likewise, the LGM shows some negative anomalies in the Atlantic–Indian sector.* — Both
points are now stated in the text; see the response to comment 11.

**13.** *Line 146-148: An explanation of why the wind stress changes do not lead to increased sea
ice in the Ross Sea would be useful. Is this due to the presence of relatively warm water despite
enhanced salinity, density, and Ekman transport in this region?* — We checked this
directly, and the answer turned out to be more interesting than the question assumed, so we set it
out in full.

The mechanism the reviewer proposes is real, but it operates in the last interglacial rather than in
the glacial states the comment refers to. Averaged between 60 and 75°S in winter, the glacial sea
ice response is in fact fairly uniform around the continent: concentration rises by 0.62 in the
Ross sector, 0.70 in the Weddell sector and 0.53 in the Adélie sector at the Last Glacial Maximum,
and all three sectors cool to within about 0.1 K of the surface freezing point, so ice growth is
not limited by surface heat content anywhere.

![](letter_figs/figR9_ross_sector.png)

**Figure R1.4 (new analysis).** (a) Winter surface temperature above the local freezing point and
(b) winter sea ice concentration, by sector and climate state. (c) The same quantities as anomalies
relative to PI. The Ross sector separates from the others only in the last interglacial, where it
warms to 2.7 K above freezing and loses 0.19 of ice concentration, an order of magnitude more than
the Weddell or Adélie response. In the glacial states all three sectors collapse onto the freezing
point and gain ice comparably.

So in the last interglacial the Ross sector is 0.97 K warmer than in PI and sits 2.72 K above the
freezing point, against changes below 0.3 K elsewhere, and its winter ice concentration falls by
0.19. There the wind can export ice efficiently but the water is too far from freezing for the
exported ice to be replaced, which is precisely the reviewer's proposed mechanism. We have
rewritten the passage to say this, rather than attributing the behaviour to the glacial wind
response as the submitted version implied. This asymmetry is also why the Ross sector departs from
the circumpolar pattern in the variability analysis.

**14.** *Line 167: It would be helpful to quantify the thermal contribution, similar to the
quantification provided for the haline contribution in line 157.* — Added. The thermal term
accounts for 70–85% of the total at the transformation maximum in the interglacials, against
15–30% for the haline term, and falls to 12–17% in the glacial states.

**15.** *Fig. 2 Would it be possible to share a figure with the same y-axis limits for all panels?*
— Yes. Supplementary Fig. S5 now shows the same transformation curves on common vertical and
horizontal axes, so magnitudes and density positions can be compared directly across climate states
and regions. We have kept the independently scaled version in the main text because the shape of
the individual curves is otherwise hard to read, and the main text now points to the common-axis
version explicitly.

![](letter_figs/figS_common_axis-1.png)

**Figure R1.5 (new Supplementary Fig. S5).** Winter transformation for the four sectors and five
climate states on a common vertical and horizontal axis. Plotted this way the glacial weakening of
the transformation maximum and its shift to denser classes are directly comparable between panels,
which the independently scaled main-text version does not allow.

**16.** *Line 174: "Section 4?" If this is intended to refer to the Methods section.* — Corrected to
refer to the Methods section by name.

**17.** *Fig.3 Please consider adding the climate-state labels at the top of each column, as in the
other figures.* — Added, here and on the other multi-panel figures; Reviewer 3 made the same request.

**18.** *Line 180/181: The LIG is first described as "showing modest changes", but later as
"exhibiting a pronounced increase". These descriptions appear inconsistent.* — The two statements
referred to different quantities, which was not clear. The changes are modest in the circumpolar
integral and pronounced locally in the Ross Sea sector. The text now says so explicitly.

**19.** *Could the authors briefly explain what is meant by thermal density tendency and
sea-ice-driven density tendency in the methods.* — Definitions have been added to the Methods,
including the sign convention and the physical meaning of brine rejection versus melt.

**20.** *Line 209: Please add the appropriate reference(s) here to support this statement.* — We now
cite Cerovečki et al. (2013) and Abernathey et al. (2016) in support of the statement that turbulent
rather than radiative fluxes control the surface buoyancy budget.

**21.** *Line 216-246: Please refer to the relevant subpanels of Figure A5.* — Done.

**22.** *Line 249, 255: The phrase "across at high latitudes" is somewhat unclear.* — Both instances
have been rewritten. They now read "over the ice-free parts of the Southern Ocean south of about
50°S" and "south of the ice edge" respectively.

**23.** *Could the authors clarify where the primary deep water formation regions occur in this
model? ... relatively little signal is apparent over the Weddell and Ross Seas.* — We have added
this diagnosis to the Results, and it is shown in Figure R1.2 above. In the pre-industrial state the
Weddell sector accounts for 70% of the area with winter mixed layers deeper than 400 m south of
55°S and 97% of the area deeper than 600 m, centred near 68°S in the gyre interior. The formation
region is therefore the open Weddell gyre rather than the shelf, which is directly relevant to the
model limitation discussed under G1.
This also explains the pattern the reviewer noticed: the Southern Annular Mode heat flux signal is
strongest over open water, and in the glacial states the coastal regions where dense water is
produced are precisely those insulated by ice, so the surface flux anomalies there are small even
though the transformation response is not.

**24.** *Lines 261-263: Please consider expanding on the physical mechanism linking positive SAM to
freshwater loss and intensified brine rejection.* — Expanded, with the mechanism stated as a
sequence: strengthened and poleward-shifted westerlies increase offshore Ekman transport of sea ice,
which opens coastal polynyas, which exposes water at the freezing point to the atmosphere and
sustains rapid new ice growth, which concentrates brine locally while the exported ice carries the
compensating freshwater away to melt further north. References added.

**25.** *Line 291, 294: Please check citation format.* — Corrected.

**26.** *Line 301: Figure A4 appears to show only model results. Please consider including the
observational estimates in the figure (or referring to them directly).* — We have added the
quantitative comparison to the text and to the new Fig. 8, reproduced as Figure R1.1 above, which
shows our transformation recomputed over the Pellichero domain alongside the standard domain. We report their published
values directly in the Discussion, including the 5 ± 5 Sv transformation into denser classes and
the stated factor of 2–5 by which the freshwater term exceeds the heat term in their sector, and we
compare our numbers against them. We chose to make the comparison quantitatively in the text and
in a dedicated figure rather than overplotting their curve, because their analysis is in neutral
density over a different density range and an overlay would misleadingly imply a
point-by-point correspondence that does not exist; the density-range mismatch is itself one of our
findings. The appendix figures are now also cited in numerical order.

**27.** *Lines 308-321: The discussion quantifies the interglacial changes ... but similar
quantification is not provided for the glacial changes.* — Added. The glacial transformation
maximum weakens to below 60 Sv and moves to σ₂ = 37.0–37.5 kg m⁻³, with the thermal share falling
to 12–17%.

**28.** *Lines 326 appear to repeat the statement made in line 323. In addition, the general
statement in lines 323–325 does not appear to hold equally across all climate states.* — The
repetition has been removed by merging the two sentences. We have also added the qualification the
reviewer asks for, noting that the Ross Sea departs from the circumpolar picture and that in the
glacial states the signal is carried largely by the Weddell sector.

**29.** *Line 339: Please provide a reference for the sediment-record evidence discussed here.* —
On reflection we think the passage was speculative rather than supported, and Reviewer 3 raised the
same objection from the opposite direction, noting that accumulation rates around the Antarctic
margin are very low. We have replaced the proposal with a statement of the difficulty, and now
present the persistence of the coupling as a property of the simulated system rather than as a
practical route to a marine reconstruction.

**30.** *Section 4.1: Please define the full names of the abbreviated components.* — Done. The
Methods now spell out that the radiative component is the sum of the shortwave and longwave fluxes
and the turbulent component the sum of the latent and sensible heat fluxes.

**31.** *Line 388: Please check the citation formatting.* — Corrected.

**32.** *Lines 394-403: Please consider including a summary table listing the prescribed boundary
conditions for each experiment.* — Added as Table 1; see G2.

---

# Reviewer #2

> *So in principle this is an interesting and topical paper and I would like to be able to support
> its publication. However, I have two big issues with the present MS that in my view mean it needs
> at least some substantial revision.*

We are grateful for both issues, which were well aimed. We address them in turn.

## Major issue 1: mismatch with observational studies of the modern ocean

> *There is a mismatch with observational studies of the modern ocean. In their model the
> pre-industrial situation is dominated by thermal buoyancy forcing whereas Pellichero et al, (2018)
> find the opposite ... Shi et al dismiss this difference in a single sentence, saying it is due to
> "model biases or regional differences in analysis domains". Comparing their Fig 3a, Fig 4a with
> Pellichero Fig 2a and c, the regions look similar, suggesting it's mostly model bias. But the
> switch from thermal to haline forcing is the present paper's main result, so if indeed it is due
> to model bias, this seems like a major problem.*

The reviewer is right that the original single sentence was inadequate, and right to insist that
the answer matters for the paper's main claim. We therefore tested it directly rather than
speculating. Three findings emerged, and we report all three, including the one that is
unfavourable to us.

**The domains are less similar than they appear.** This is worth showing directly,
since the similarity of the two regions is the premise of the reviewer's argument.

![](letter_figs/figR8_domain_map.png)

**Figure R2.1 (new analysis).** (a) The domain used in the submitted manuscript, everything south
of 60°S, covering 2.07×10¹³ m². (b) The seasonal sea ice zone of the model, inside the September
15% contour, which is the definition Pellichero et al. use, covering 1.22×10¹³ m². (c) The two
overlaid. The orange ring is inside our domain but outside the ice zone: it is 42.7% of the area we
integrated over, it is open water all year, and it is water their analysis does not include.

Pellichero et al. define their sector as the
region enclosed by the September 15% sea ice contour, so its northern boundary is an ice contour
that varies with longitude, not a latitude circle. Our published integral covers everything south
of 60°S. In our pre-industrial simulation, 42.7% of that area lies outside the September ice
contour and is therefore water their analysis does not include. It is also the part of the domain
that stays exposed to the atmosphere year round, and taken alone it is 89–91% thermally driven in
the dense classes. Recomputing our transformation over a sea ice sector defined exactly as theirs,
which reproduces their domain to within 2.5% in area, moves the partition in the direction they
report: the thermal share of the dense-class transformation falls from 69% to 61%, and the sea ice
contribution at the transformation maximum roughly doubles from 0.86 to 1.84 Sv.

![](letter_figs/figR1_wmt_domain_sensitivity.png)

**Figure R2.2 (manuscript Fig. 8).** Transformation integrated south of 60°S, the domain used in
the submitted version (top row), and over the seasonal sea ice zone alone, which reproduces the
domain of Pellichero et al. to within 2.5% in area (bottom row). Restricting the integration to the
ice-covered sector raises the haline share, as the reviewer would expect, but the contrast between
thermally dominated interglacials and haline dominated glacials survives intact.

**The density range matters more than the domain.** This was the more consequential finding.
Converting our simulated surface properties inside the September ice zone to the neutral density
coordinate that Pellichero et al. use, our thermally dominated transformation maximum sits near
γ_n ≈ 27.3 kg m⁻³, whereas the haline-dominated lower cell they describe occupies
γ_n = 27.9–28.8 kg m⁻³. Our simulated surface transformation barely reaches the lighter edge of
their range. In the densest classes our fluxes do populate (σ₂ ≥ 37.0 kg m⁻³), our pre-industrial
transformation is haline dominated, with 0.79 Sv from sea ice against 0.06 Sv from heat. The two
analyses are therefore largely describing different water masses, ours weighted toward mode and
intermediate densities and theirs toward bottom water densities, and where they overlap they agree
in sign. We now state in the Results that the interglacial thermal pathway we identify is more
relevant to intermediate and mode water formation than to bottom water proper.

**A residual discrepancy remains, and it is genuine model bias.** We can rule out one candidate
explanation. Pellichero et al. attribute part of the haline dominance to the thermal expansion
coefficient being near zero at the freezing point, so we checked whether our ice-covered sector
reproduces that regime. It does: the area-weighted winter surface temperature inside the September
ice zone is −1.08 °C, the median is −1.71 °C, 55% of the area lies within 0.5 °C of the local
freezing point, and α = 3.5×10⁻⁵ K⁻¹. The model is not warm-biased there, so we cannot appeal to
that. What differs is where dense water is made. In the model, pre-industrial dense water forms by
open-ocean convection in the interior of the Weddell gyre near 68°S rather than by the sequence of
shelf water formation and downslope overflow that operates in the real ocean. Open-ocean convection
exposes a large area to the atmosphere and so recruits an excessive thermal contribution while
producing water that is not dense enough, which is precisely the offset in density class described
above.

![](letter_figs/figR6_convection_sites.png)

**Figure R2.3 (new analysis).** Where the model actually convects. (a-e) Winter mixed layer depth,
400 m contour in orange. (f) In the pre-industrial state 70% of the area with a mixed layer deeper
than 400 m, and 97% of the area deeper than 600 m, lies in the Weddell sector, centred near 68°S in
the open gyre rather than over the shelf. (g, h) The deep-convection area and the maximum mixed
layer depth both contract sharply in the glacial states. We now say this plainly in the Discussion, note that it is a limitation shared across the
current model generation rather than specific to AWI-ESM2 (Heuzé, 2021: 28 of 35 CMIP6 models), and
state that this configuration has no ice shelf cavities.

**Does the main result survive?** We think it does, for two reasons that we now give in the text.
First, the regime shift survives the domain test. Recomputed over the sea ice sector alone, the
glacial states remain 15% and 17% thermally driven while the interglacials remain 53–88% thermally
driven, so the contrast is not manufactured by including open water in the interglacial integrals.
Second, the mechanism is the ice cover itself rather than the convection style: insulation of the
surface and concentration of brine rejection follow from the areal expansion of sea ice, and both
would operate in a model that formed its dense water on the shelf.

The consequences for the large-scale circulation are visible in the overturning itself.

![](letter_figs/figR2_moc_5exps.png)

**Figure R2.4 (manuscript Fig. 6).** Global overturning streamfunction (top) and anomalies relative
to PI (bottom). The abyssal cell that carries southern-sourced bottom water weakens from about
10 Sv in the interglacials to 2.4 and 1.6 Sv in LGM and MIS3, while the upper cell is largely
unchanged. The glacial reorganisation is concentrated in the cell our surface analysis addresses. What such a model would change
is the density and the geographic origin of the resulting water, not the direction of the shift in
the surface buoyancy budget. We have adjusted the framing of the paper accordingly, so that the
claim is about the partitioning of surface buoyancy forcing and its state dependence rather than
about bottom water formation rates.

## Major issue 2: relation to Chen et al. (2025)

> *I'm puzzled that they don't acknowledge or discuss the relation of their work with recent studies
> showing very similar results but analysed through different tools and language. Using the same
> ocean model Chen et al, 2025 (GRL 52, e2025GL114809) describe formation of very dense AABW in
> glacial time as due to the enhanced ice formation ... One of the authors (Lohmann) is on both
> papers, yet this paper is not referenced, let alone discussed, by Shi et al. I ask myself what is
> learned from the WMT analysis that is not already discussed in that paper?*

The omission was an oversight on our part and we apologise for it. The paper is now cited in both
the Introduction and the Discussion, and we address the reviewer's question directly rather than
merely adding the reference.

We first acknowledge the genuine agreement. Chen et al. attribute the large glacial volume of
Atlantic bottom water to a substantial increase in sea ice export toward lower latitudes, which
supplies dense shelf water year round, and to weaker mixing between northern- and southern-sourced
water, which lets that water retain the properties of its origin. Our glacial conclusions are
consistent with theirs, and we say so explicitly rather than presenting our result as new in that
respect.

What the transformation analysis adds is of two kinds. The first is measurement rather than
inference. Chen et al. diagnose water mass volumes, the overturning streamfunction, and mixed-layer
and sea ice fields, from which the role of sea ice is inferred because those fields covary with it.
A decomposed surface buoyancy budget measures the transformation directly and attributes it to
individual flux components in sverdrups, so the sea ice contribution is quantified rather than
inferred. Our analysis therefore tests the mechanism they proposed, and finds it supported.

The second is scope. Their simulations are ocean-only, with a prescribed atmosphere, and cover the
Last Glacial Maximum and the present day. Ours are fully coupled, so sea ice and the atmospheric
fluxes evolve together, and they span five climate states. A shift between two forcing regimes
cannot be identified from a single glacial state; it requires the interglacial end members for the
contrast to exist at all. The regime shift that is our central result is therefore not addressable
within their experimental design, which is why the two studies reach compatible conclusions about
the glacial ocean while answering different questions.

We have phrased this in the manuscript as the present analysis testing and quantifying the
mechanism Chen et al. proposed, rather than as a claim that they missed something.

## More minor points

> *The fundamental issue must be getting right the formation of sea ice and rejection of brine close
> to the continent ... In this regard the model looks good, by comparison to most paleo studies. The
> comparatively high resolution close to the continent must be helpful I'm sure, (though at 25 km it
> may still not be enough to realistically model the dynamics of polynya formation and brine
> rejection). Some more information on just how this is being managed would add to the value of the
> paper.*

We agree with the reviewer's caveat and have expanded the Methods accordingly. The unstructured
mesh refines to about 25 km around Antarctica, which resolves the larger coastal polynyas as
features but does not resolve the boundary layer processes that set polynya dynamics. The relevant
mechanics are that ice growth and brine release are computed thermodynamically at each surface node
and enter the ocean as a salt flux, so the brine signal is represented where the model puts the
ice, but its intensity depends on the simulated wind-driven divergence rather than on resolved
polynya circulation. We now state this, together with the more fundamental point that the
configuration has no ice shelf cavities and therefore omits basal melt entirely.

> *The authors highlight the lack of deep convection in the glacial open ocean due to the very
> extensive winter-time ice cover. A little more discussion of how/why the model achieves this ice
> cover would be welcome, as previous paleo-model studies often don't get that ... (Another paper
> they don't cite is Lhardy, Climate of the Past, 17, 1139–1159, 2021 ...)*

Thank you for this reference, which we have added and discussed. We now note that maintaining an
extensive glacial ice cover is not a general feature of paleoclimate simulations, that many PMIP
models underestimate glacial Southern Ocean sea ice and its seasonality, and that colder glacial
forcing on its own tends to intensify open-ocean convection rather than suppress it, so that better
boundary conditions do not by themselves fix the problem. We highlight the finding of Lhardy et al.
that the only configuration avoiding an unrealistically deep northern-sourced cell was the one in
which brine sinking along the Antarctic margin was parameterised explicitly. In our simulations the
glacial ice expands sufficiently to insulate the interior gyres, and the area of deep winter mixed
layers contracts to roughly one fifth of its pre-industrial value. We now state that our glacial
result depends on that simulated expansion being realistic, and that while proxy reconstructions
support the direction of the change, the quantitative agreement remains uncertain.

> *In Results section 2.1 the acronyms (MH, LIG etc) are used without introduction. They are defined
> later, section 4, but as the paper is laid out this is not obvious to the reader coming to
> section 2.*

Corrected. The five experiments and their abbreviations are now introduced at the start of the
Results, at first use, in addition to the Methods definition. Reviewer 3 made the same point.

> *The discussion of the role of the effects of the SAM is interesting, but could I think be
> shortened, since it is somewhat secondary to the main messages of the paper. Figs 5 and 6 could be
> moved to the supplementary information.*

We have done this. All of the Southern Annular Mode figures, including Figs 5 and 6, have been
moved to the Supplementary Information, so the main text no longer carries any figure devoted to
the mode. We have also shortened the section, removing the repetition that Reviewer 1 flagged and
cutting the speculative passage about marine proxy reconstruction. What remains in the main text is
a compact account of the result, with the supporting composites available to readers who want them.

We note that Reviewer 3 took the opposite view of this material, describing it as the more
generalisable part of the study. Moving the figures to the supplement while keeping a concise
statement of the result in the main text seemed to us the arrangement most likely to satisfy both
readings, and we are content with the outcome.

> *For figs 5 and 6, I think it would be helpful if the forcings were on the same equivalent scales
> of buoyancy anomaly -- currently they are in different units and it is impossible to see how they
> compare in absolute values.*

This is a fair criticism. We have added a statement of the comparison in equivalent surface density
tendency terms to the Results: expressed that way, the freshwater contribution is the larger of the
two wherever sea ice is present, and the heat contribution is larger only in the ice-free sector,
which is the same partition that governs the mean state. We have kept the native units in the
figures themselves, which are now supplementary, because the heat and freshwater fields are what
the model diagnoses and readers may wish to compare them against other studies in conventional
units. The text now makes the relative magnitudes explicit. If the reviewer would prefer the
figures recast in buoyancy units, we will do so.

---

# Reviewer #3

> *Shi et al apply a water mass transformation (WMT) framework to various glacial/interglacial
> paleoclimate sims using AWI-ESM2. This is an interesting framework for thinking about mechanisms
> of deepwater formation and how production regimes may shift between climate states, and a novel
> application. ... The observation that the reduction in buoyancy loss through atmospheric heat loss
> from enhanced sea ice cover is compensated by increased buoyancy loss via sea ice export, is an
> interesting result and points to an interesting stabilising mechanism for ventilation.*

We thank the reviewer for this reading, and particularly for identifying the compensation mechanism
as the interesting result, which has helped us frame the revision.

## Main issue: the PI state does not form AABW the way the real ocean does

> *... to the best of our knowledge today densewater formation in the Southern Ocean is not
> primarily a thermally driven regime. Like many models, AWI-ESM2 appears to form AABW via open
> ocean convection under PI conditions whereas in the real ocean AABW is largely a salt/freshwater
> controlled process occurring on the shelves (e.g. Orsi et al 1999). ... This must be the starting
> point for thinking about glacial changes, not thermally driven open ocean convection, unless the
> authors believe this is somehow an overlooked process in the real modern ocean.*

We accept this criticism. We do not believe shelf processes are overlooked in the real ocean, and
we have restructured the Discussion so that the model's convection behaviour is stated at the
outset rather than left implicit.

We confirmed the reviewer's suspicion diagnostically. In the pre-industrial simulation the Weddell
sector accounts for 70% of the area with a winter mixed layer deeper than 400 m south of 55°S and
97% of the area deeper than 600 m, centred near 68°S in the interior of the gyre rather than over
the continental shelf.

![](letter_figs/figR6_convection_sites.png)

**Figure R3.1 (new analysis).** (a-e) Winter mixed layer depth with the 400 m contour in orange.
(f) The pre-industrial sector breakdown, showing that convection is concentrated in the Weddell
gyre interior rather than on the shelf, which is the bias the reviewer identified. (g, h) The
glacial contraction of both the deep-convection area and the maximum mixed-layer depth. The model therefore does form its dense water by open-ocean convection, as
the reviewer supposed. We now state this explicitly, cite Orsi et al. (1999) and Toggweiler and
Samuels (1995) for the shelf pathway that operates in the real ocean, note that this configuration
has no ice shelf cavities and therefore cannot represent basal melt at all, and place the model in
context by citing Heuzé (2021), who finds 28 of 35 CMIP6 models forming deep and bottom water by
open-ocean convection with no model showing convincing shelf export in the Weddell Sea.

We note in passing, and with no criticism intended, that Toggweiler and Samuels (1995) concerns
sea ice brine rejection rather than ice shelf melt; there are two papers by those authors from that
year and the more frequently cited one concerns Drake Passage. We have cited the sea ice paper,
whose conclusion that GCM salinity biases near Antarctica arise from circulation deficiencies rather
than from inadequate boundary conditions is directly relevant to our case.

> **On the footnote:** *This could be tested by comparing your method only in the regions with
> observations. Alternatively you could apply the WMT framework to a reanalysis product e.g. Glorys.*

We took the first of these suggestions. In summary: 42.7% of our published domain lies outside the
September sea ice edge; restricting to their domain moves the thermal share of dense-class
transformation from 69% to 61%; and, more importantly, our thermally dominated maximum sits at
γ_n ≈ 27.3 kg m⁻³ while their haline-dominated cell occupies γ_n = 27.9–28.8 kg m⁻³, so the two
analyses largely concern different water masses. In the densest classes we do populate, our result
is haline dominated and therefore agrees with theirs.

![](letter_figs/figR1_wmt_domain_sensitivity.png)

**Figure R3.2 (manuscript Fig. 8).** The like-for-like comparison the reviewer asked for.
Transformation integrated over the published domain south of 60°S (top) and over the seasonal sea
ice zone that reproduces the observational domain (bottom). The haline share rises in the
restricted domain, but the glacial-interglacial contrast is unaffected, which is why we conclude
the regime shift is not an artefact of the biased pre-industrial end member.

We did not carry out the reanalysis calculation. Doing it properly would require surface flux
fields consistent with the reanalysis ocean state, and a transformation budget assembled from
mismatched flux and hydrography products would introduce its own imbalance that we could not
separate cleanly from the model-observation difference we are trying to diagnose. Since the
restricted-domain and density-class comparisons turned out to localise the discrepancy quite
precisely, we judged the reanalysis calculation to be a substantial undertaking with limited
additional diagnostic value here. We would be glad to attempt it if the reviewer considers it
essential.

> *how applicable is this result to thinking about the real ocean and glacial interglacial change
> when the interglacial regime is not representative of the real preindustrial/interglacial ocean?
> ... Given coastal polynyas are a key process in how AABW forms today are we just looking at a
> shift from a more biased PI state to a more realistic glacial state?*

This is the sharpest form of the objection and we have tried to answer it honestly rather than
deflect it. Two arguments, both now in the Discussion.

The regime shift survives the domain test. Recomputed over the sea ice sector alone, the glacial
states remain 15% and 17% thermally driven while the interglacials remain 53–88% thermally driven.
The contrast is therefore not an artefact of including open water in the interglacial integrals,
which is the most obvious way the biased end member could have manufactured it.

More fundamentally, the mechanism is the ice cover rather than the convection style. Insulation of
the surface from the atmosphere and the concentration of brine rejection both follow from the areal
expansion of sea ice. Both would operate in a model that formed its dense water on the shelf. What
such a model would change is the density and the geographic origin of the water produced, not the
direction of the shift in the surface buoyancy budget. So we do think the reviewer is partly right
that the glacial state is the better-represented of the two, and we now say so; but the shift itself
does not depend on the interglacial bias.

We have correspondingly narrowed what we claim. The paper now presents a result about how the
partitioning of surface buoyancy forcing between heat and freshwater responds to changing sea ice
cover, which is robust in these simulations, and explicitly not a quantitative reconstruction of
past bottom water formation rates.

> *I find the SAM wind results interesting and I think more generalisable. The results suggest
> changes in wind shift/strength impacts not just the ekman divergence i.e. upwelling of deepwaters,
> but also the salt pump via sea ice export, and thus the formation of deepwaters. This suggests
> changes in the wind patterns effect both the push and pull driving the overturning.*

We are grateful for this observation, which identified a mechanism our original analysis had not
isolated, and we have carried out new work to test it directly. The reviewer's framing of the wind
acting on both the push and the pull turns out to be quantitatively supported, and it has become
one of the more informative results in the revision.

We separated the two routes explicitly. For the same high and low SAM composites, we computed the
zonally integrated northward Ekman transport at 60°S from the simulated zonal wind stress, which
measures the push, and the coastal brine input integrated south of 65°S, which measures the pull.
Both were computed for all five climate states.

![](letter_figs/figR4_sam_push_pull.png)

**Figure R3.3 (new Supplementary Figure).** (a) The positive phase strengthens the Ekman transport at
60°S by 8.0--12.8 Sv in every climate state. (b) In the same winters it increases the coastal brine
input, by 51 and 42 mSv in PI and LIG and by 24 and 31 mSv in LGM and MIS3. (c) The two responses
scale together across the five states (r = 0.77). (d) The resulting anomaly in the peak
transformation rate, with the density class of the maximum indicated.

Three results follow, and all three are new relative to the submitted version.

The push and the pull are not independent expressions of the same forcing but two halves of one
mechanism. A positive anomaly strengthens the Ekman divergence that brings deep water to the
surface, and in the same winters it strengthens the coastal salt pump that converts that water into
denser classes. Because they respond together, wind variability modulates dense water formation
more effectively than either route alone would imply. This is the reviewer's hypothesis, and we can
now state it quantitatively.

The balance between the routes is state dependent, and in a way we did not anticipate. The Ekman
response is largest in the interglacials (11.2 Sv in PI, 12.8 Sv in LIG) and weakest at the Last
Glacial Maximum (8.0 Sv), even though the glacial mean wind stress is stronger. Extensive glacial
ice cover transmits stress to the ocean less efficiently and damps the divergence that the same wind
anomaly would otherwise produce. The haline route strengthens correspondingly, so the net modulation
remains comparable while operating through a different pathway. This mirrors the mean-state regime
shift, and it means the regime shift governs not only the mean state but also the response to
internal variability.

The spatial patterns show the mechanism directly.

![](letter_figs/figR5_sam_wind_ice.png)

**Figure R3.4 (new Supplementary Figure).** Zonal wind stress anomaly (top) and sea ice concentration
anomaly (bottom) for high-SAM minus low-SAM composites. The westerlies strengthen over the
circumpolar belt in every state. The sea ice response is a dipole, with ice lost near the coast
where polynyas open and brine is rejected, and gained further north where the exported ice melts.
The dipole sits further north and is substantially stronger in the glacial states.

These results are now described in the main text, and the supporting figures are in the
Supplementary Information.

> *It is worth noting that available estimates suggest a equatorward and weakening of the glacial
> westerlies, that the models do not seem to capture under glacial forcings (e.g. Gray et al 2023).
> Such a shift may be worth discussing in light of the 'SAM' wind mechanism, and given such a
> shift/weakening is absent in your glacial simulations.*

The reviewer is correct and we now discuss this. We measured our own jet: the winter zonal-mean
maximum sits at 45.7°S with 5.83 m s⁻¹ in PI and moves to 49.4°S with 6.30 and 6.85 m s⁻¹ in LGM
and MIS3, a poleward shift of 3.7° with strengthening. Gray et al. (2023) infer an equatorward
shift of 4.8° (2.9–7.1°, 95% CI) and a weakening of about 25% at the Last Glacial Maximum relative
to the mid-Holocene. Our simulated response is therefore of the opposite sign, and we state this in
both the Results and the Discussion.

Two considerations bound the consequences, and we give both. The discrepancy is shared across the
model ensemble rather than specific to AWI-ESM2, since Gray et al. report that the inferred shift
exceeds that produced by any PMIP3/4 member. And the mean-state regime shift depends on the areal
extent of sea ice and on brine rejection rather than on jet position; Gray et al. themselves find
no correlation across the model ensemble between Antarctic sea ice extent and either the latitude
of the westerlies or the position of the SST front, so a jet-position bias does not automatically
propagate into the sea ice field on which our main result rests. We are careful, however, to note
that the Southern Annular Mode results describe a wind perturbation imposed on a background state
whose winds are biased, and should be read as a statement about mechanism rather than as a
reconstruction of glacial wind-driven variability.

> *Interestingly the westerlies seem to intensify in both the glacial and interglacial simulations;
> why is this? Could this relate to the initializing simulation not being at equilibrium? Is the PI
> sim extended for the same additional time as the other simulations to adjust for model drift?*

The two intensifications have different origins, and we now explain both. The LIG strengthening
follows that period's orbital configuration, with its high eccentricity and different perihelion
producing a distinct seasonal insolation distribution; the jet moves to 47.6°S and 6.92 m s⁻¹. The
glacial strengthening follows the steepened meridional temperature gradient imposed by the expanded
ice sheets and sea ice.

On drift: MH is indistinguishable from PI in both jet latitude (45.7°S in both) and strength (5.83
against 5.84 m s⁻¹). Since MH branches from the same equilibrated PI state and was integrated for
the same length as the other paleo experiments, a drift explanation would have to produce a signal
in LIG, LGM and MIS3 while leaving MH unchanged, which we think is implausible. We have added this
argument to the text. For completeness, the PI control was integrated for 1500 years and all paleo
simulations for 1000 years, with the final 100 years analysed in each case.

> *The authors mention that they use ideal age as a proxy for overturning which seems odd given the
> overturning can be directly computed within the model. ... Plots of both overturning streamfunction
> and ideal age throughout the global ocean would be required ... Furthermore the methods mention
> that the simulations have been run for 1000yr which is enough to equilibrate physical properties
> but not ideal ages globally.*

We agree on all three points and have added the analyses.

The overturning streamfunction, diagnosed online by the model, is now shown in the new Fig. 6. The
abyssal cell weakens from −10.0 Sv in PI, −10.1 Sv in MH and −9.2 Sv in LIG to −2.4 Sv in LGM and
−1.6 Sv in MIS3, a reduction of roughly 75–85%, while the upper cell changes comparatively little
(19.1 Sv in PI against 20.0 and 21.7 Sv in the glacials). The glacial reorganisation in these
simulations is therefore concentrated in the abyssal cell.

![](letter_figs/figR2_moc_5exps.png)

**Figure R3.5 (manuscript Fig. 6).** Global overturning streamfunction for the five climate states
(top) and anomalies relative to PI (bottom). The blue abyssal cell, representing northward
spreading of southern-sourced bottom water, nearly disappears in the glacial states while the upper
cell is largely maintained.

Global ideal age is now shown in the new Fig. 7, as basin-mean vertical profiles and zonal-mean
sections. Below 2000 m the ageing is strongly basin dependent: the Atlantic increases only modestly
from 377 to 439 years (LGM) and 508 years (MIS3), while the Southern Ocean more than doubles from
381 to 789 and 844 years and the Pacific goes from 847 to 1291 and 1314 years. This basin contrast
is much clearer than the single 4000 m map conveyed, and it is consistent with the abyssal cell
weakening while the upper cell is maintained.

![](letter_figs/figR3_age_global.png)

**Figure R3.6 (manuscript Fig. 7).** Basin-mean ideal age profiles (a-d), mean age below 2000 m
(e), and zonal-mean sections for the five states (f-j). The glacial ageing is concentrated below
about 2000 m and is largest in the deep Pacific and Southern Ocean, while the Atlantic, which
remains ventilated from the north, changes comparatively little.

On equilibration, the reviewer is right and we now state the caveat explicitly: each experiment was
integrated for 1000 years, which is shorter than the oldest simulated ages, so the glacial age
fields are not fully equilibrated and should be read as lower bounds on the true equilibrium
contrast. We cite Millet et al. (2025) for the point that equilibrated deep-ocean age tracers
require multi-millennial integrations. We also now note that ideal age responds to mixing as well as
to advection, so it measures the combined effect of a weaker abyssal cell and altered interior
mixing, and that the streamfunction is the more direct diagnostic; the two agree in this case. The
introduction has been corrected accordingly.

## Minor comments

> *MH, LIG are not abbreviation that commonly used so I would remove them from the text and spell out
> each time. Also I don't think these were not defined at any point of the manuscript.*

The abbreviations are now defined at first use in the Results as well as in the Methods. We have
retained them thereafter because the figures are organised in five columns labelled by these codes
and spelling the names out at every occurrence made several passages considerably harder to read.
If the reviewer feels strongly we will expand them throughout.

> *L19: 'in the model' needs to be clearly stated as this is not how AABW forms in the real modern
> ocean.* — Added to the abstract, which now reads "In the model, interglacial dense water formation
is driven mainly by turbulent heat loss...". The same qualification has been added in the Results.

> *L25: change deep from "clear imprint on deep-ocean ventilation" to abyssal Southern Ocean. ... The
> deep ocean is usually considered as all the water below 1000 m.* — The abstract now refers to the
abyssal overturning cell and abyssal water specifically, and the new global age figure covers the
whole water column rather than a single level.

> *L102-105: a slightly expanded intro on the SAM may be helpful.* — Expanded; see Reviewer 1,
comment 5.

> *L 125: typo, more.* — Corrected.

> *L129: it would be much more useful to show the salinity minus the whole ocean change applied, to
> see what arises from local dynamics.* — This is a good suggestion. The glacial salinity anomalies
in Fig. 1 include the uniform global increase applied at initialisation to represent water stored in
the ice sheets, which obscures the locally generated signal. We have noted this explicitly in the
text so that readers can interpret the panels correctly.

> *L134: 400m is not very deep given you are comparing to what is happening to ideal age at 4000 m.*
— Agreed; the 400 m contour marks active convection rather than the depth reached by the resulting
water. We now also report the 600 m statistics and the maximum mixed-layer depth, and the new
overturning figure connects the surface signal to the abyssal circulation directly rather than by
implication.

> *L144: the westerlies appear to increase under both glacial and last interglacial forcings, why?*
— Answered above.

> *L152: it needs to be clearer that this is in the model.* — Rephrased to "the model produces deep
mixed layers...".

> *L175: I wonder if this result isn't more applicable to AAIW/SAMW formation?* — The reviewer's
intuition is borne out by the density-class analysis described above. Our thermally dominated
transformation maximum sits at γ_n ≈ 27.3 kg m⁻³, which is intermediate and mode water density
rather than bottom water density. We now state in the Results that the interglacial thermal pathway
is more relevant to intermediate and mode water formation than to bottom water proper. We are
grateful for the suggestion, which turned out to be one of the more useful reframings in the
revision.

> *L195: in the real ocean freshwater input from icesheet melting is important today.* — Added, with
the explicit acknowledgement that this configuration has no ice shelf cavities and applies no
prescribed meltwater flux, so that pathway is absent from our simulations.

> *L206-209: should be moved to the introduction maybe?* — Moved.

> *L291-297: this point seems fundamental to the rest of the manuscript.* — Agreed; addressed at
length above.

> *L302-305: can't you just compare you model results directly in the regions with observations?* —
Done; see above and Fig. 8.

> *L339: this would be great, but sed rates around the Antarctic margin are very low.* — Agreed, and
the passage has been replaced with a statement of that difficulty. Reviewer 1 asked for a supporting
reference for the same passage; we concluded the proposal was not well founded and have withdrawn
it.

> *L347: not the water, the carbon in the water – given the substantial and variable preformed 14C
> aging.* — Corrected. The text now refers to the radiocarbon age of dissolved inorganic carbon.

> *General comment on the figures: would it be possible to show whats happening under the ice
> shelves? At the moment they are just whited out. Anyway to show this?* — Unfortunately not. The
model configuration does not include ice shelf cavities, so there is no ocean beneath the floating
ice to show; the white areas are genuinely outside the model domain rather than masked output. We
have stated this in the Methods and figure captions so that readers are not left wondering, and we
list it among the limitations, since the absence of cavities also means basal melt is not
represented.

> *Please be consistent across figures and label MH, LIG, LGM, MIS3 on top of all panels.* — Done
throughout; Reviewer 1 made the same request.

---

# Reviewer #4

> *I co-reviewed this manuscript with one of the reviewers who provided the listed reports.*

We thank Reviewer 4 for their contribution to the review, and the journal for supporting
early-career researchers in peer review.

---

# Editorial requirements

**Colour vision deficiency.** The new figures use the Okabe-Ito colour-blind-safe palette, and the
transformation curves are additionally distinguished by line style. We have replaced the red/green
pairing in the transformation figures with vermillion and bluish-green, and the overturning
anomaly panels use a purple-orange diverging scale rather than red-green.

**Data availability.** A statement has been added. The model output needed to reproduce every
figure has been assembled as a repository deposit, comprising the climatological monthly means for
the five experiments, the 100-year monthly transformation time series, the Southern Annular Mode
composite fields, and the ocean mesh files. The DOI will be inserted on acceptance. Source data for
the line-graph figures are provided with the paper.

**Code availability.** A statement has been added. The analysis and plotting scripts, covering the
surface buoyancy flux decomposition, the transformation calculations including the new sea ice
sector comparison, the compositing, and all figures, are archived with the data deposit and are
also available at https://github.com/xshi-awi/aabw_5exps.

**Source data.** Provided for the line-graph figures, as a single spreadsheet with one sheet per
figure panel.

**ORCID.** The corresponding author has linked an ORCID to the submission system, and co-authors
have been asked to do the same.
