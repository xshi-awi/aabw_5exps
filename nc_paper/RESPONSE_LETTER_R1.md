# Response to Reviewers

**Manuscript:** Glacial–interglacial regime shift in Southern Ocean dense water formation
**Authors:** Xiaoxu Shi, Jiping Liu, Chaoyuan Yang, Gerrit Lohmann

---

We thank the editor and the four reviewers for their detailed and constructive comments, which
helped a lot improve the manuscript. A question is pointed out, namely the comparison between our
pre-industrial result and observation-based transformation estimates for the modern Southern
Ocean. We took that seriously and ran the diagnostic tests that Reviewers 2 and 3 proposed. The
result is more informative than we expected.

We firstly compare our result with the classic WMT observation from Pellichero et al. (2018). In
our manuscript, 42.7% of the studied area we integrated over lies outside the sea ice edge and is
therefore water region that Pellichero et al. did not include. After we apply WMT analysis over
the same domain and the same season as Pellichero et al. did, we got a WMT curve similar as
theirs, and freshwater component dominate the light water class (< 27.1 kg/m3) and the dense water
class (> 27.67 kg/m3), while the thermal signature is confined to a band at intermediate water
densities. Thus the driving component for WMT is actually mixed. The second reason is that
Pellichero et al. worked from a different kind of quantity than ours. Their estimate rests on Argo
floats, ship-based profiles and instrumented marine mammals, the last being the only platform that
samples under winter sea ice, with the ice edge taken from SMMR and SSM/I passive microwave data
for 2000 to 2015 and the geostrophic term from AVISO mean dynamic topography for 1993 to 2012.
They do not measure the surface fluxes at all. They solve the mixed layer heat and salt budgets
with the flux as the unknown, so what they recover is the surface flux together with the diapycnal
mixing term, which that budget cannot separate, whereas our model diagnoses the surface flux
alone. The uncertainties they state are the mixed layer detection method, the instrumental errors
in pressure, temperature and conductivity, and the spatial sampling, which is sparsest in
precisely the ice-covered region that matters most here. Some difference is therefore unavoidable
before any model bias is considered. Their freshwater term, on the other hand, is the same
quantity as ours. They define it as the flux from evaporation, precipitation and ice formation or
melt, so it carries the sea ice contribution exactly as our total freshwater term does. Note that
Southern Ocean is more warm and too fresh near Antarctica compared to observation, a bias shared
by most coupled models, and this pushes our WMT curve toward lighter density than the Pellichero
et al. describe, which is to say our whole curve is displaced toward the light end relative to
theirs. We now state this plainly. What the paper claims, however, is the relative change between
climate states rather than the absolute values. On that basis we consider the regime shift robust,
and we have bounded the rest of our claims accordingly.

We then compare our result with the ORAS5 reanalysis for 1999 to 2018, the Argo-rich part of the
record where the assimilation constrains the Southern Ocean most strongly. ORAS5 carries no sea
ice thermodynamic term in its freshwater flux, so we removed the sea ice contribution from our own
flux as well and compared the same two terms on each side, heat and evaporation minus
precipitation minus runoff. Over the sea ice zone in the annual mean, the same domain as
Pellichero et al., the two agree closely: our densification branch peaks at +5.0 Sv against +4.4
Sv in ORAS5, one density class apart, and both show the light-end buoyancy gain, ours reaching
-8.0 Sv against -12.9 Sv. Over the domain and season our paper uses, south of 60°S in JJA, both
curves are positive almost everywhere and peak in the same density class, but ours is the stronger
of the two, +62.9 against +36.8 Sv. We read the agreement in sign, shape and peak location as
support for the diagnostic, and the winter amplitude difference as a consequence of our deeper and
more extensive winter mixed layers, which expose more water to the atmosphere than the reanalysis
does. In the densest classes neither curve sustains transformation, which is expected once the
brine input is removed from both sides.

The ORAS5 comparison is informative but it is not independent validation, and the revised text
says so. ORAS5 is a model constrained by observations rather than a measurement, and the two
fields this calculation needs are the two that are least constrained, since no observing system
measures an air-sea flux directly. That is the same objection Pellichero et al. raise when they
decline to use reanalysis fluxes, writing that in the Southern Ocean such products are poorly
constrained and essentially unusable in the seasonally ice-covered region.

For convenience the principal new figures are reproduced in this letter, so that the reviewers can
assess them conveniently.

Reviewer comments are reproduced in full below in black, each followed by our response in blue; text quoted from the revised manuscript is given in blue italics. Line
numbers in our responses refer to the revised manuscript unless stated otherwise. All changes are
marked in the accompanying tracked-changes file.

---

# Response to comments by Reviewer #1

> *The manuscript presents a comprehensive investigation of Southern Ocean dense water formation
> across a range of glacial and interglacial climates using the AWI-ESM2 Earth system model. By
> applying a water mass transformation (WMT) framework, the authors successfully separate the
> relative roles of thermal and haline buoyancy forcing and demonstrate a compelling regime
> shift between interglacial and glacial climates. The manuscript is well structured, the
> analyses are technically rigorous, and the decomposition of the different buoyancy
> contributions provides valuable physical insight. The use of WMT diagnostics allows the
> underlying mechanisms to be quantified rather than inferred from changes in circulation alone.
> Overall, I found the study scientifically interesting and suitable for publication after major
> revisions. My comments below are primarily intended to strengthen the robustness of the
> conclusions and improve the clarity of the presentation.*

We thank the reviewer for the constructive general comments and a thorough list of specific
comments.

## General comments

### Model evaluation and confidence in simulated dense water formation

> *The conclusions of this study rely entirely on a single Earth system model. While I do not
> consider this to be a major limitation by itself, I think the manuscript would benefit from a
> more thorough discussion of how well AWI-ESM2 represents present-day Southern Ocean dense
> water formation and Antarctic Bottom Water formation. The manuscript briefly states that the
> PI simulation produces comparable WMT magnitudes to observational estimates, but Figure A4
> only shows the model results, making it difficult for the reader to evaluate this statement.
> Given the importance of the model for all subsequent paleoclimate interpretations, I encourage
> the authors to provide stronger evidence supporting its performance. This could include a
> direct comparison with observational WMT estimates, or a more detailed discussion of known
> model strengths and biases in Southern Ocean stratification, sea ice, mixed-layer depth, and
> Antarctic Bottom Water formation. Even if such comparisons already exist in previous AWI-ESM2
> publications, briefly summarising them here would substantially increase confidence in the
> conclusions.*

Authors' response:

We agree with the reviewer. This concern also overlaps with those from Reviewers 2 and 3. Our
response has three parts.

The reviewer is right that Figure A4 showed only model curves. We now place our pre-industrial
transformation beside the observation-based estimate of Pellichero et al. (2018). The result of
Pellichero et al. comes from Argo floats, ship-based observations and instrumented marine mammals,
the only platform that samples under winter sea ice; the ice edge comes from SMMR and SSM/I
passive microwave data for 2000-2015, and the geostrophic term from AVISO mean dynamic topography
for 1993-2012. We cannot reproduce that calculation, because it rests on a seal-tag mixed layer
database those authors assembled and we do not have. We have therefore compared our figure
against their published figure.

To match Pellichero et al. (2018), we recomputed our transformation over the same sector enclosed
by the September 15% sea ice concentration, and took the annual mean instead of the austral winter
mean used elsewhere in our manuscript.

![](letter_figs/figR20_wmt_merged.png)

**Figure R1.1 (new analysis).** Transformation rate against neutral density γ, in three
comparisons. Top row, the seasonal sea ice zone in the annual mean, with all terms present: (a)
our pre-industrial state, decomposed into the heat, freshwater and sea ice contributions, and (b)
the observational estimate of Pellichero et al. (2018), their Fig. 3a, reproduced for comparison.
Pink shading in (a) marks where the heat term carries the total and blue shading where the
freshwater term does, classified point by point along the drawn curve rather than by density bin,
which places the thermally driven band at γ = 27.10–27.68 kg m⁻³. Middle row, the same domain and season
against the ORAS5 reanalysis for 1999-2018: (c) our state with the sea ice contribution removed
and (d) ORAS5, which carries heat and evaporation minus precipitation minus runoff only. Bottom
row, the domain and season the paper itself uses, south of 60°S in JJA: (e) ours with the sea ice
contribution removed and (f) ORAS5. Rows two and three are unshaded, because with the sea ice term
absent from both sides a heat-against-freshwater classification would describe an incomplete
freshwater flux and read as a physical result rather than a missing field. Our simulated surface
water is lighter than observed, so equal γ does not identify the same water mass across panels.

As seen in Fig. R1.1a-b, the two curves share their structure. Both show buoyancy gain at the
lighter classes, a zero crossing, and densification at the heavier classes, with the freshwater
term carrying the light end and both curves converging towards zero at the dense end. Freshwater
component dominate the light water class (< 27.1 kg/m3) and the dense end (> 27.67 kg/m3), shown
by the blue shading in Fig. R1.1a, while the thermal signature is confined to a band at
intermediate water densities, the pink shading in Fig. R1.1a. In our pre-industrial state the
transformation reaches a minimum of -9.2 Sv, crosses zero, and peaks at +6.6 Sv; Pellichero et al.
report a minimum near -22 Sv and a densification peak of 5±5 Sv. The sign structure and the order
of magnitude of the densification branch therefore agree.

Two differences should be noted, and they are of different kinds. Our light-end buoyancy gain is
weaker than theirs, and our whole curve sits at lighter densities, because the simulated Southern
Ocean surface is fresher than observed; both follow from the same bias, which we quantify under
model limitations below.

It should be noted that the difference could be definitional, and it sets a floor on how closely
the two can ever agree. Pellichero et al. solve the mixed layer heat and salt budgets for the
surface fluxes as the unknown, using the observed seasonal cycle of mixed layer temperature,
salinity and depth together with Ekman and geostrophic advection and entrainment. Thus, what
Pellichero et al. solve for is the surface flux together with the diapycnal mixing term, because
the mixed layer budget cannot separate them, whereas our model diagnoses the surface flux alone.
Some discrepancy is therefore expected before any model bias is considered, and the agreement in
shape should be read in that light. Their freshwater term, by contrast, is the same quantity as
ours, since they define it as the flux from evaporation, precipitation and ice formation or melt,
so the sea ice contribution is included on both sides of that particular comparison.

The reviewer, and Reviewer 3, also asked whether we could compute an observational transformation
ourselves from a reanalysis rather than citing a published one. A transformation budget needs the
net surface heat flux, the net surface freshwater flux, and the sea ice thermodynamic salt flux.
We checked the Copernicus ocean reanalyses, namely GLORYS12, the GREP ensemble and the
analysis-forecast product, which carry temperature, salinity, velocity and sea ice state but no
surface flux fields at all. ORAS5 does carry a net heat flux and a net water flux, but excluding
the sea ice term: we tested its water flux inside the September ice zone and found no freezing
signal, -1.2e-5 kg m-2 s-1 at peak freezing against -1.6e-5 at peak melt, so the field is
evaporation minus precipitation minus runoff. Therefore, we calculate transformation based on
ORAS5, and for consistency, we also remove the sea ice term from our model, then compare our model
results with ORAS5.

The results are shown in Fig. R1.1c-f. Over the sea ice zone in the annual mean, panels (c) and
(d), the agreement is close. Our densification branch peaks at +5.0 Sv and the ORAS5 branch at
+4.4 Sv, one density class apart at γ 27.49 against 27.28, and both show the light-end buoyancy
gain, ours reaching a minimum of -8.0 Sv against -12.9 Sv in the reanalysis. The sign structure,
the position of the zero crossing and the magnitude of the densification peak therefore agree on
a domain neither calculation was tuned to.

Over the domain and season the paper itself uses, south of 60°S in JJA, panels (e) and (f), the
two agree in shape but not in amplitude. Both are positive almost everywhere, since in winter the
whole sector is losing buoyancy, and both peak in the same density class at γ 27.28 and 27.08
respectively. Ours is the stronger, +62.9 Sv against +36.8 Sv. We do not attribute that gap to the
reanalysis being correct. Our winter mixed layers south of 60°S are deeper and more extensive than
in ORAS5, which exposes more water to the atmosphere and so supports more transformation, and the
same bias is visible directly in the mixed layer depth comparison shown later in this response. We
therefore read the winter comparison as agreement in sign, shape and density class, with an
amplitude difference that follows from a mixed layer bias we document rather than from the
diagnostic itself.

In the densest classes, above γ 27.7, neither curve sustains much transformation once the sea ice
term is removed from both sides, which is the expected outcome: that is the class where brine
rejection does the work, and it is exactly the term ORAS5 cannot supply. With the term restored,
our own dense-class transformation is positive, which is the result the paper rests on and the one
no available reanalysis can be used to check.

Two caveats bound what this comparison can support. ORAS5 is a model constrained by observations
rather than a measurement, and the two fields the calculation needs are the two that are least
constrained, since no observing system measures an air-sea flux directly; its heat flux is
computed from ERA-Interim forcing and then adjusted by sea surface temperature restoring, not
fitted to any flux observation. And Pellichero et al. themselves reject this approach for this
region, writing that ocean buoyancy fluxes are commonly obtained from atmospheric reanalysis
datasets but that *"in the Southern Ocean such reanalysis products are poorly constrained, and
essentially unusable in the seasonally sea-ice covered region."* We therefore present the ORAS5
comparison as a consistency check on the diagnostic rather than as independent validation, and we
continue to treat their published result as the observational reference.

Another point from the reviewer is about the discussion of known model strengths and biases in
Southern Ocean stratification, sea ice, mixed-layer depth, and Antarctic Bottom Water formation.
Since the second last comment of the reviewer specifically deals with "Model limitations", we refer
to our response to that comment.

### Boundary conditions

> *The description of the prescribed boundary conditions for the paleoclimate simulations could
> be expanded. In particular, the glacial simulations use the GLAC-1D ice-sheet reconstruction,
> but it is not entirely clear how the ice-sheet changes are implemented in the model (e.g.,
> topography, land-sea mask, ice-sheet extent, bathymetry, or other boundary conditions). As
> Antarctic dense water formation is highly sensitive to such factors, providing additional
> details on the experimental setup would help readers assess the robustness of the simulations.
> This is not a request for additional simulations, but rather for a clearer discussion of the
> robustness of the conclusions and the extent to which they may depend on the chosen model
> configuration and forcing datasets. A summary table listing the prescribed greenhouse gas
> concentrations, orbital parameters, and ice-sheet boundary conditions for each experiment
> would also improve the readability of the Methods section.*

Following the reviewer's suggestion, we have provided a summary table (Table R1, manuscript Table
1), which lists the greenhouse gas concentrations, orbital parameters and ice sheet configuration
for all five experiments. The values are those actually used in the runs rather than the protocol
defaults.

| | CO₂ (ppm) | CH₄ (ppb) | N₂O (ppb) | Eccentricity | Obliquity (°) | Perihelion (°) | Ice sheet / bathymetry |
|---|---|---|---|---|---|---|---|
| PI | 284.3 | 808.2 | 273.0 | 1850 CE | 1850 CE | 1850 CE | modern |
| MH | 264.4 | 597.0 | 262.0 | 0.018682 | 24.105 | 180.87 | modern |
| LIG | 275.0 | 685.0 | 255.0 | 0.039378 | 24.040 | 95.41 | modern |
| LGM | 190.0 | 375.0 | 200.0 | 0.018994 | 22.949 | 294.42 | GLAC-1D 21 ka |
| MIS3 | 210.5 | 556.2 | 247.4 | 0.013676 | 23.2591 | 25.99 | GLAC-1D 38 ka |

**Table R1.** Boundary conditions prescribed in the five equilibrium
simulations. Orbital parameters for PI are those of 1850 CE. Greenhouse gas concentrations are
volume mixing ratios.

We also expanded the text describing the boundary conditions used in the experimental design,
especially the implementation of the GLAC-1D ice-sheet reconstruction. Now reads:

> *"We perform five equilibrium simulations spanning distinct climate states including three interglacial periods, i.e., pre-industrial (PI), mid-Holocene (MH), and last interglacial (LIG), and two glacial periods, i.e., the Last Glacial Maximum (LGM) and Marine Isotope Stage 3 (MIS3). The boundary conditions are configured following the criteria of PMIP4 and are summarised in Table 1. Orbital parameters are calculated following , and the greenhouse gas concentrations are taken from multi-archive reconstructions from ice core records and recent measurements of firn air and atmospheric samples . For LGM and MIS3, we fix the boundary conditions at 21 ka and 38 ka respectively, with the topography and ice-sheet properties deriving from the GLAC1D reconstruction . The GLAC-1D reconstruction enters the model through several boundary fields at once, and it is clearest to list them by the component that uses them. Specifically, the atmosphere component receives the ice sheet extent and surface elevation as orography, together with the land-sea mask implied by the lowered sea level. The land surface receives the same topography, from which river routing and vegetation distribution are derived. The ocean component receives bathymetry and a modified coastline. Because the ocean component uses an unstructured mesh, a changed coastline cannot be imposed on the existing grid and a dedicated mesh must be generated for each glacial period; we use meshes constructed for 21 ka and 38 ka respectively, while PI, MH and LIG share the modern-geometry mesh. Global mean ocean salinity is raised in the glacial experiments to account for the water stored in the ice sheets. No prescribed ice sheet meltwater flux is applied in any of the equilibrium simulations, and the model does not include ice shelf cavities, so basal melt beneath floating ice shelves is not represented."*

### Clarifying several interpretations

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

In the revised manuscript the moderated passage now reads:

> *"Whether this persistent coupling could be exploited for proxy reconstruction is less clear than we previously suggested. Recovering an interannual-to-decadal signal from marine sediments would require accumulation rates and chronological precision that are rarely available around the Antarctic margin, where sedimentation is slow and bioturbation mixes the record over intervals far longer than the variability of interest. We therefore present the persistence of the coupling as a property of the simulated system rather than as a practical route to a marine Southern Annular Mode reconstruction."*

### Model limitations

> *Model limitations: The manuscript convincingly demonstrates that sea ice drives the transition
> from a thermally dominated to a haline-dominated dense water formation regime. However, the
> discussion would benefit from a brief consideration of the known biases and limitations of
> AWI-ESM2, particularly regarding Southern Ocean sea ice, mixed-layer depths, and Antarctic Bottom
> Water formation. Discussing how these model characteristics may influence the proposed mechanism
> would help readers assess the robustness and broader applicability of the conclusions.*

Authors' response:

This comment asks specifically about bias, where the general comment above asked about evaluation,
so we treat the two separately. We take the three quantities the reviewer names in turn, and in
each case compare the pre-industrial simulation against observations rather than against the other
climate states.

Before doing so, one limitation underlies all three and is best stated once. We diagnosed where
the model actually forms its dense water. In the pre-industrial simulation the Weddell sector
accounts for 70 per cent of the area with a winter mixed layer deeper than 400 m south of 55°S and
97 per cent of the area deeper than 600 m, centred near 68°S in the interior of the gyre rather
than over the shelf. This is open-ocean convection, not the shelf overflow pathway that operates
in the real ocean. It is a limitation shared across the current model generation rather than
particular to AWI-ESM2: Heuzé (2021) finds that 28 of 35 CMIP6 models form deep and bottom water
by open-ocean convection that is too deep, too frequent or too extensive, with no model showing
convincing shelf export in the Weddell Sea. We also now state that this configuration has no ice
shelf cavities, so basal melt beneath floating ice is absent by construction.

Thanks for the comment. We now state the model limitation explicitly. We diagnosed where the model
actually forms its dense water. In the pre-industrial simulation, the Weddell sector accounts for
70% of the area with a winter mixed layer deeper than 400 m south of 55°S, centred near 68°S in
the interior of the gyre (see Figure R1.2). We note that this is a limitation shared across the
current model generation. Heuzé (2021) finds that 28 of 35 CMIP6 models form deep and bottom water
by open-ocean convection that is too deep, too frequent, or too extensive, with no model showing
convincing shelf export in the Weddell Sea. We also now state that this configuration has no ice
shelf cavities, so basal melt beneath floating ice is absent by construction.

The broad pattern is reproduced. Both the model and the reanalysis keep shallow mixed layers over
most of the open Southern Ocean, deepen towards the Antarctic margin, and place the deepest water
of the sector in the Atlantic sector. Where they part company is the Weddell gyre interior: the
model opens a patch deeper than 400 m that the reanalysis does not have at all, and its Weddell
sector mean is 128 m against 71 m. This is the open-ocean convection described above, and the
comparison localises it rather than leaving it as an assertion.

![](letter_figs/figR17_mld_pi_vs_obs.png)

**Figure R1.2 (new analysis).** Winter mixed layer depth south of 60°S. (a) The pre-industrial
simulation. (b) The ORAS5 ocean reanalysis, JJA climatology over 1990-2009. The large-scale
pattern agrees: both are shallow over most of the open Southern Ocean, both deepen towards the
Antarctic margin, and both show the deepest water of the sector in the Atlantic sector. The
difference is where it matters for this study. The model places a large patch deeper than 400 m in
the interior of the Weddell gyre, covering 1.9 per cent of the area south of 60°S against 0.0 per
cent in the reanalysis, and its Weddell sector mean is 128 m against 71 m. Area-weighted over the
whole sector the model gives 100 m against 78 m, so the bias is concentrated in the convecting
patch rather than spread evenly. We note that the two use different mixed layer definitions, a
0.01 kg m⁻³ density threshold in ORAS5 and a bulk Richardson number criterion in the model, so the
point-by-point difference is not purely model error.

In addition, the published evaluation reports that Antarctic sea ice in this configuration is too
thin especially in the central Weddell Sea in September, and that the Southern Ocean carries a
pronounced warm bias of 1.43 K. The abyssal overturning cell is reported at about 10 Sv,
comparable to our pre-industrial value.

That cell is produced by the wrong pathway, as described above, so we present our results as a
statement about how the partitioning of surface buoyancy forcing responds to changing ice cover,
and explicitly not as a quantitative reconstruction of bottom water formation rates.

Taken together these are systematic biases. That matters for how the results should be read.
Because the same bias is present in all five experiments, it largely cancels in the differences
between them, which is what the paper actually reports. We make no claim about the absolute
transformation rate or the absolute density at which it occurs; we claim that the balance between
the thermal and haline pathways shifts between climate states, and a bias common to every state
cannot generate that shift.

The corresponding passage added to the Discussion reads:

> *"Independent evaluations of this model provide context for these limitations, and we summarise the relevant points here rather than leaving them to be sought in the model description literature. Sidorenko et al. (2019) evaluate the coupled configuration against observations and report that the abyssal overturning cell associated with Antarctic Bottom Water is reproduced with a maximum of about 10 Sv, comparable to the pre-industrial value obtained here. The same evaluation identifies the Southern Ocean as the region of the most pronounced warm bias of 1.43 K against the PHC climatology at the resolution used here, and notes that Antarctic sea ice is too thin, hardly reaching 0.25 m in the central Weddell Sea in September. Their overall assessment is that although biases in the representation of water mass properties and ventilation mechanisms are present, the model still produces a reasonable density distribution that maintains realistic transports. A warm and thin-ice Southern Ocean is precisely the state in which open-ocean convection is favoured over shelf processes, so these documented biases are consistent with the convection behaviour we diagnose above."*

### Latitude and longitude labels, and subpanel references

> *Please consider adding latitude and longitude labels to all map figures where appropriate. In
> addition, referring to specific figure subpanels throughout the text, rather than only the
> full figure number, would make the discussion easier to follow.*

Subpanel references have been added throughout the Results and Discussion. On the map labels, the
polar stereographic panels now carry labelled latitude circles, and the ventilation age maps carry
labelled meridians as well. On the fifteen-panel composite figures we label the latitude circles
only, because a full graticule with meridian labels on panels of that size was illegible in trial
versions and crowded the panel margins. We are happy to add complete labelling if the reviewer
prefers, or to enlarge those figures to accommodate it. As a concrete example, the ventilation age
maps now carry labelled meridians at 60°W, 0°, 60°E and 120°E as well as labelled latitude
circles, and the mixed layer and composite figures carry labelled latitude circles; Figure R1.3 in
this letter shows the result. The same treatment has been applied to every map figure in the
revised manuscript and supplement.

![](letter_figs/fig_age_labels-1.png)

**Figure R1.3 (revised manuscript figure).** The ventilation age figure as an example of the
labelling now applied throughout. Meridians are labelled just outside the map boundary and
latitude circles along the 150°E meridian, at 60°S, 70°S and 80°S. Labels are placed only where
there is white space, so that those of neighbouring panels do not run into one another and nothing
is written over the data. The mixed
layer, wind stress and composite figures have been treated the same way; please refer to the
revised manuscript and supplement for the complete set.

## Specific comments

> **1.** LIG, LGM and MIS should be capitalised.

Thank you for catching this. The three abbreviations are now capitalised consistently throughout
the manuscript, the figure captions and the supplementary, and MIS is written as MIS3 everywhere
so that it matches the column headings of the figures.

> **2.** Line 21-22: The transition to SAM is quite abrupt.

The abstract now introduces the Southern Annular Mode through the mechanism that connects it to the mean state, noting that because the balance between the thermal and haline pathways is set by how much of the surface is ice covered, it also governs the response to internal atmospheric variability.

In the revised manuscript this now reads:

> *"The balance between these two pathways is set by sea ice extent, which is modulated by internal atmospheric variability. Further analysis suggests that the Southern Annular Mode modulates transformation by 3-8 Sv between its positive and negative phases in every climate state, working through heat loss when the ocean is open and through brine rejection when it is ice covered."*

> **3.** Line 24: "same mechanisms" sounds vague her without additional context. The abstract should be understandable on its own without requiring the reader to refer to the main text.

Replaced with an explicit statement of the two pathways, so the abstract no longer relies on the
reader having seen the main text.

In the revised manuscript this now reads:

> *"The balance between these two pathways is set by sea ice extent, which is modulated by internal atmospheric variability. Further analysis suggests that the Southern Annular Mode modulates transformation by 3-8 Sv between its positive and negative phases in every climate state, working through heat loss when the ocean is open and through brine rejection when it is ice covered."*

> **4.** Lines 25-27: Similar to Comment 2 above, additional context is needed before introducing the deep ocean ventilation results. The motivation for analysing ventilation ages only becomes clear later in the manuscript, whereas the abstract should be self-contained.

The abstract now bridges from surface forcing to ventilation via the overturning circulation, stating that weaker glacial transformation coincides with a collapse of the simulated abyssal cell from about 10 to 2 Sv, before the ages are given. This also incorporates the new overturning analysis requested by Reviewer 3.

In the revised manuscript this now reads:

> *"Weaker glacial transformation coincides with a collapse of the simulated abyssal overturning cell from about 10 to 2 Sv and with older abyssal water. Ideal ages at 4000 m remain below 500 years during interglacials but exceed 750 years basin-wide during glacials, reaching 1500 years in the Pacific."*

> **5.** Lines 102-105: SAM is introduced only briefly here. Given its importance throughout the Results and Discussion, please consider providing a slightly more detailed introduction.

The introduction now defines the mode as the leading mode of extratropical Southern Hemisphere variability, describes what its positive phase does to the westerlies and hence to Ekman divergence, air-sea heat exchange and sea ice export, and cites the observational literature linking it to bottom water changes.

In the revised manuscript this now reads:

> *"The Southern Annular Mode is the leading mode of extratropical Southern Hemisphere circulation variability, describing a meridional redistribution of atmospheric mass between mid-latitudes and Antarctica . Its positive phase strengthens the circumpolar westerlies and displaces them poleward, which changes Ekman divergence, air-sea heat exchange, and the northward export of sea ice. In the present-day ocean these adjustments modulate mixed-layer depth and surface buoyancy forcing , and they have been linked to observed changes in bottom water properties . Whether the mode remains an effective modulator of dense water formation once the mean state changes, in particular once an expanded ice cover separates the ocean from the atmosphere, is not known."*

> **6.** Line 125: typo: 'more'.

Corrected ("moer" → "more").

> **7.** Line 127: Figure numbering; A3 can't be called before A1 and A2.

The appendix figures have been reordered so that they are cited in numerical order.

> **8.** Line 129: 'Here,'.

Corrected.

> **9.** Line 140: As a result, the deep MLD signals, characteristic of open-ocean convection, largely…. Also, "vanish" is perhaps not the most scientific wording.

Rephrased to "are strongly reduced and become confined to narrow coastal bands and polynyas", and we now quantify the reduction rather than describing it qualitatively.

In the revised manuscript this now reads:

> *"The reduction is substantial rather than complete. Summed south of 55°S, the area with a winter mixed layer deeper than 400 m falls from 7.510^11 m^2 in PI to 1.710^11 m^2 in LGM and 1.610^11 m^2 in MIS3, and the maximum mixed-layer depth decreases from 991 m to 819 m and 645 m. Mixed layers deeper than 600 m, which cover 2.410^11 m^2 in PI, essentially disappear in both glacial states. The residual glacial signals in coastal regions are therefore genuinely shallower than their interglacial counterparts, consistent with dense water production that is confined to narrow coastal bands rather than distributed across the open gyres."*

> **10.** Line 141: These signals are not particularly visible even in the coastal regions (the darker blue shading in LGM is at ~200m, so considerably shallower than the interglacials).

This is a fair objection and we have addressed it with numbers. Summed south of 55°S, the area with a winter mixed layer deeper than 400 m falls from 7.5×10¹¹ m² in PI to 1.7×10¹¹ m² in LGM and 1.6×10¹¹ m² in MIS3, the maximum mixed-layer depth decreases from 991 m to 819 m and 645 m, and mixed layers deeper than 600 m essentially disappear in both glacial states. The text now says explicitly that the residual glacial signals are genuinely shallower than their interglacial counterparts. Figure R1.2 below shows the glacial fields on a scale that resolves them, and the reviewer's reading of the shading is correct and is now reflected in the text.

The reviewer's observation also pointed to a presentation problem, which we have fixed. On the
shared 0–400 m scale of the original figure the glacial panels are almost featureless, because the
99th percentile of the glacial winter mixed layer south of 55°S is only 280 m (LGM) and 296 m
(MIS3). Replotting those two states on their own 0–300 m scale makes the coastal cells visible.

![](letter_figs/figR7_mld_glacial_zoom.png)

**Figure R1.4 (new supplementary panel).** (a) LGM and (b) MIS3 winter mixed layer depth south of
60°S, on a 0–300 m scale. The tightened scale resolves discrete
deep cells along the coast, in the Weddell, Prydz Bay and Ross sectors, which is where the glacial
dense water is produced. This is the figure we now use in the supplement.

> **11.** Line 134-141: Fig.1 i-l subpanels are not discussed at all.

A paragraph discussing the surface density panels has been added. It notes that the glacial density anomalies follow the salinity field rather than the temperature field, because the thermal expansion coefficient is small at the near-freezing temperatures south of the ice edge, and points out the density-compensated LIG anomaly and the weak negative LGM anomalies in the Atlantic-Indian sector.


In the revised manuscript this now reads:

> *"The surface density anomalies (Fig. 1i-l) follow the salinity field rather than the temperature field in the glacial states. Cooling and salinification both act to densify the surface, but at the near-freezing temperatures that prevail south of the ice edge the thermal expansion coefficient is small, so the haline term sets both the pattern and the magnitude. In LIG the warm and salty Bellingshausen-Amundsen anomaly is close to density compensated (Fig. 1j), and the LGM retains weak negative density anomalies in parts of the Atlantic-Indian sector where surface freshening outweighs cooling."*

> **12.** Line 144-146: The LIG also exhibits similar behaviour, although the changes are smaller in some basins. Likewise, the LGM shows some negative anomalies in the Atlantic–Indian sector.

Both points are now stated in the text; see the response to comment 11.

> **13.** Line 146-148: An explanation of why the wind stress changes do not lead to increased sea ice in the Ross Sea would be useful. Is this due to the presence of relatively warm water despite enhanced salinity, density, and Ekman transport in this region?

We checked this directly, and the answer turned out to be more interesting than the question assumed, so we set it out in full.

The mechanism the reviewer proposes is real, but it operates in the last interglacial rather than in
the glacial states the comment refers to. Averaged between 60 and 75°S in winter, the glacial sea
ice response is in fact fairly uniform around the continent: concentration rises by 0.62 in the
Ross sector, 0.70 in the Weddell sector and 0.53 in the Adélie sector at the Last Glacial Maximum,
and all three sectors cool to within about 0.1 K of the surface freezing point, so ice growth is
not limited by surface heat content anywhere.

![](letter_figs/figR9_ross_sector.png)

**Figure R1.5 (new analysis).** (a) Winter surface temperature above the local freezing point and
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

> **14.** Line 167: It would be helpful to quantify the thermal contribution, similar to the quantification provided for the haline contribution in line 157.

Added. The thermal term accounts for 70–85% of the total at the transformation maximum in the interglacials, against 15–30% for the haline term, and falls to 12–17% in the glacial states.


In the revised manuscript this now reads:

> *"This transformation is dominated by surface heat loss (red line), which accounts for 70-85% of the total at the transformation maximum, with limited sea ice extent restricting the haline contribution from brine rejection to a secondary role (15-30%; green line). Other freshwater fluxes (blue lines) oppose densification through net precipitation and river runoff. We note that this thermally dominated maximum occurs at densities that correspond to intermediate and mode waters rather than to bottom water. In the densest classes that the simulated surface fluxes reach (σ₂ ≥ 37.0 kg m⁻³), the PI transformation is already haline dominated, with the sea ice term contributing 0.79 Sv against 0.06 Sv from heat in the annual mean. The thermal dominance reported here is therefore a property of the integral over all density classes, and we quantify its sensitivity to the choice of integration domain and density range in the Discussion."*

> **15.** Fig. 2 Would it be possible to share a figure with the same y-axis limits for all panels? This may not be necessary for the final manuscript, but it would make it easier to compare the magnitude of the changes across the different climate states.

Yes. Supplementary Fig. S5 now shows the same transformation curves on common vertical and horizontal axes, so magnitudes and density positions can be compared directly across climate states and regions. We have kept the independently scaled version in the main text because the shape of the individual curves is otherwise hard to read, and the main text now points to the common-axis version explicitly.

![](letter_figs/figS_common_axis-1.png)

**Figure R1.6 (new Supplementary Fig. S5).** Winter transformation for the four sectors and five
climate states on a common vertical and horizontal axis. Plotted this way the glacial weakening of
the transformation maximum and its shift to denser classes are directly comparable between panels,
which the independently scaled main-text version does not allow.

> **16.** Line 174: "Section 4?"If this is intended to refer to the Methods section.

Corrected to refer to the Methods section by name.

> **17.** Fig.3 Please consider adding the climate-state labels at the top of each column, as in the other figures.

Added, here and on the other multi-panel figures; Reviewer 3 made the same request.

![](letter_figs/fig03_labels-1.png)

**Figure R1.7 (revised manuscript Fig. 3).** The surface density tendency figure with the
climate-state headings added, so that the columns are identified in the same way as in the other
figures, and with latitude and longitude now labelled. Every panel carries the latitude circles, at
50°S, 60°S, 70°S and 80°S along 150°E, and the 180° meridian. The remaining meridians are given on
the outer columns only, 60°W and 120°W on the left and 60°E and 120°E on the right, so that no
label falls between two panels; all panels share the same projection and extent. All labels sit
outside the frame. The freshwater counterpart, Fig. 4, has been treated the same way.

> **18.** Line 180/181: The LIG is first described as "showing modest changes", but later as "exhibiting a pronounced increase". These descriptions appear inconsistent.

The two statements referred to different quantities, which was not clear. The changes are modest in the circumpolar integral and pronounced locally in the Ross Sea sector. The text now says so explicitly.

> **19.** Could the authors briefly explain what is meant by thermal density tendency and sea-ice-driven density tendency in the methods.

Definitions have been added to the Methods, including the sign convention and the physical meaning of brine rejection versus melt.

In the revised manuscript this now reads:

> *"We refer throughout to the thermal density tendency, meaning the surface density change produced by the net heat flux acting through the thermal expansion coefficient, that is the -≤ft(/C_p)Q_net term, and to the sea-ice-driven density tendency, meaning the part of the haline term S F_fw that is carried by the thermodynamic growth and melt of sea ice. A positive value denotes densification of the surface water in both cases. Brine released during ice growth increases surface salinity and therefore density, whereas melting releases freshwater and reduces it."*

> **20.** Line 209: Please add the appropriate reference(s) here to support this statement.

We now cite Cerovečki et al. (2013) and Abernathey et al. (2016) in support of the statement that turbulent rather than radiative fluxes control the surface buoyancy budget.

> **21.** Line216-246: Please refer to the relevant subpanels of Figure A5.

Done, and the same treatment has been applied throughout the Results and Discussion rather than
only in the passage the reviewer names. As an example of the form these references now take:

> *"In PI, strong thermal densification (red shading) is concentrated in the open ocean (Fig. 3a), with maximal values reaching 5 10^-6 kg m⁻³ s⁻¹ in the circumpolar band between 50-65°S. This thermal densification is overwhelmingly controlled by turbulent fluxes (sensible and latent heat), with radiative fluxes playing a minor role in the buoyancy budget (Fig. 3f-j vs. Fig. 3k-o), consistent with observationally based surface buoyancy budgets for the Southern Ocean ."*

> **22.** Line 249, 255: The phrase "across at high latitudes" is somewhat unclear. Does this refer specifically to the Antarctic coastal region or the Southern Ocean south of a particular latitude? Please consider rephrasing for clarity.

Both instances have been rewritten. They now read "over the ice-free parts of the Southern Ocean south of about 50°S" and "south of the ice edge" respectively.

> **23.** Could the authors clarify where the primary deep water formation regions occur in this model? Identifying these regions would also help assess whether the SAM-related changes shown in Figure 5 are directly influencing the main regions of dense water formation, as relatively little signal is apparent over the Weddell and Ross Seas.

We have added this diagnosis to the Results, and Figure R1.2 above shows it directly: the winter mixed layer depth for all five states. In the pre-industrial state the Weddell sector accounts for 70% of the area with winter mixed layers deeper than 400 m south of 55°S and 97% of the area deeper than 600 m, centred near 68°S in the gyre interior. The formation region is therefore the open Weddell gyre rather than the shelf, which is directly relevant to the model limitation discussed under model evaluation above. This also explains the pattern the reviewer noticed: the Southern Annular Mode heat flux signal is strongest over open water, and in the glacial states the coastal regions where dense water is produced are precisely those insulated by ice, so the surface flux anomalies there are small even though the transformation response is not.

> **24.** Lines 261-263: Please consider expanding on the physical mechanism linking positive SAM to freshwater loss and intensified brine rejection by additional explanation and supporting references.

Expanded, and we have added the two references the reviewer asks for. The passage now states the
mechanism as a chain rather than asserting the endpoint, and cites the observational work on
Antarctic Bottom Water production in the Cape Darnley polynya and on the sea ice contribution to
Southern Ocean transformation.

In the revised manuscript this now reads:

> *"This pattern is controlled by sea ice processes, and the chain that produces it runs as follows. Strengthened and poleward-shifted westerlies increase the offshore component of the wind stress at the coast, which drives sea ice away from the continental margin. The divergence opens and widens coastal polynyas, exposing water already at the freezing point directly to a cold atmosphere. New ice therefore forms rapidly in those openings, and because ice rejects most of its salt as it grows, the salt is delivered to the surface ocean precisely where the water is coldest and densest. The ice exported offshore carries the compensating freshwater north, where it melts and freshens the surface. The result is the observed dipole, with salinification at the coast and freshening offshore, and it is the same mechanism identified in observations of Antarctic Bottom Water production in the Cape Darnley polynya and in the sea ice contribution to Southern Ocean water mass transformation more generally ."*

> **25.** Line 291, 294: Please check citation format.

Corrected.

> **26.** Line 301:Figure A4 appears to show only model results. Please consider including the observational estimates in the figure (or referring to them directly) to support this statement. As noted above, it would also improve readability if the appendix figures were cited in numerical order.

We have added the quantitative comparison to the text and to the new Fig. 8, reproduced as Figure R1.1 above, which shows our transformation recomputed over the Pellichero domain alongside the standard domain. We report their published values directly in the Discussion, including the 5 ± 5 Sv transformation into denser classes and the stated factor of 2–5 by which the freshwater term exceeds the heat term in their sector, and we compare our numbers against them. We chose to make the comparison quantitatively in the text and in a dedicated figure rather than overplotting their curve, because their analysis is in neutral density over a different density range and an overlay would misleadingly imply a point-by-point correspondence that does not exist; the density-range mismatch is itself one of our findings. The appendix figures are now also cited in numerical order. We note that this comment and the general comment on model evaluation above concern the same comparison, and we have kept the two answers consistent: the quantitative comparison with the published observational values is given once in the Discussion, and both replies point to the same new figure.

> **27.** Lines 308-321: The discussion quantifies the interglacial changes (e.g., lines 315–316), but similar quantification is not provided for the glacial changes. For consistency, please consider including quantitative estimates for both.

Added. The glacial transformation maximum weakens to below 60 Sv and moves to σ₂ = 37.0–37.5 kg m⁻³, with the thermal share falling to 12–17%.


In the revised manuscript this now reads:

> *"The winter transformation maximum weakens to below 60 Sv and moves to σ₂ = 37.0-37.5 kg m⁻³, and the thermal share of the transformation falls to 12-17% while the sea ice term becomes dominant. Extensive ice cover insulates the ocean from atmospheric heat exchange, and simultaneously intensifies haline forcing through enhanced brine rejection concentrated in coastal regions."*

> **28.** Lines 326 appear to repeat the statement made in line 323. In addition, the general statement in lines 323–325 does not appear to hold equally across all climate states. For example, the Ross Sea and Weddell Sea exhibit different behaviour for the peak-density classes.

The repetition has been removed by merging the two sentences. We have also added the qualification the reviewer asks for, noting that the Ross Sea departs from the circumpolar picture and that in the glacial states the signal is carried largely by the Weddell sector.

In the revised manuscript this now reads:

> *"This circumpolar picture does not hold uniformly in every sector. The Ross Sea in particular departs from it, showing weaker and at some densities opposing responses, and in the glacial states the circumpolar signal is carried largely by the Weddell sector while the Ross and Adelie sectors contribute little."*

> **29.** Line 339: Please provide a reference for the sediment-record evidence discussed here.

On reflection we think the passage was speculative rather than supported, and Reviewer 3 raised the same objection from the opposite direction, noting that accumulation rates around the Antarctic margin are very low. We have replaced the proposal with a statement of the difficulty, and now present the persistence of the coupling as a property of the simulated system rather than as a practical route to a marine reconstruction.

In the revised manuscript this now reads:

> *"Whether this persistent coupling could be exploited for proxy reconstruction is less clear than we previously suggested. Recovering an interannual-to-decadal signal from marine sediments would require accumulation rates and chronological precision that are rarely available around the Antarctic margin, where sedimentation is slow and bioturbation mixes the record over intervals far longer than the variability of interest. We therefore present the persistence of the coupling as a property of the simulated system rather than as a practical route to a marine Southern Annular Mode reconstruction."*

> **30.** Section 4.1: Please define the full names of the abbreviated components.

Done, and the definitions are now given where the figures use the terms rather than only in the
flux description.

In the revised manuscript this now reads:

> *"In the figures the radiative component denotes the sum of the shortwave and longwave fluxes, and the turbulent component denotes the sum of the latent and sensible heat fluxes."*

In the revised manuscript this now reads:

> *"In the figures the radiative component denotes the sum of the shortwave and longwave fluxes, and the turbulent component denotes the sum of the latent and sensible heat fluxes."*

> **31.** Line 388: Please check the citation formatting.

Corrected.

> **32.** Lines 394-403: Please consider including a summary table listing the prescribed boundary conditions for each experiment (e.g., greenhouse gas concentrations, orbital parameters, ice-sheet reconstruction, and any other relevant forcings). This would make the experimental setup considerably easier to follow.

Added as Table 1; please see our response to the second general comment.

---
