#!/usr/bin/env python
"""
Apply the revision edits to nc-main.tex -> build/revised.tex.

Each edit is (old, new, label). The script asserts that every `old` string is
found exactly once, so a silent mismatch cannot slip through.
"""
import sys
from pathlib import Path

SRC = Path('nc-main.tex')
DST = Path('build/revised.tex')

s = SRC.read_text()
edits = []


def E(old, new, label):
    edits.append((old, new, label))


# =====================================================================
# ABSTRACT  — R1#2,3,4 (SAM transition abrupt, "same mechanisms" vague,
# ventilation needs context), R3 L19 ("in the model"), R3 L25 (abyssal SO)
# =====================================================================
E(r"""Winter dense water formation in the Southern Ocean drives global overturning and regulates deep-ocean heat and carbon storage, but how surface buoyancy forcing controls this process across past climate states remains unclear. Using water mass transformation analysis in AWI-ESM simulations of the pre-industrial, mid-Holocene, last interglacial, last glacial maximum, and marine isotope stage 3, we identify a sea-ice-mediated regime shift between climate states. Interglacial dense water formation is thermally driven by turbulent heat loss over the open ocean, whereas glacial formation is haline-driven by brine rejection in coastal polynyas as expanded sea ice insulates the open ocean. The Southern Annular Mode systematically modulates this process across all states, enhancing dense water formation by 3–8 Sv during positive phases through the same mechanisms that govern the mean state. These surface forcing changes leave a clear imprint on deep-ocean ventilation. Simulated ages at 4000 m remain below 500 years during interglacials but exceed 750 years basin-wide during glacials, with Pacific ages reaching 1500 years. This directly links the sea-ice-mediated regime shift to prolonged abyssal isolation and enhanced glacial carbon storage. Overall, our results provide a mechanistic framework for paleo Southern Ocean circulation.""",
  r"""Winter dense water formation in the Southern Ocean drives global overturning and regulates deep-ocean heat and carbon storage, but how surface buoyancy forcing controls this process across past climate states remains unclear. Using water mass transformation analysis in AWI-ESM simulations of the pre-industrial, mid-Holocene, last interglacial, last glacial maximum, and marine isotope stage 3, we identify a sea-ice-mediated regime shift between climate states. In the model, interglacial dense water formation is driven mainly by turbulent heat loss over the open ocean, whereas glacial formation is haline-driven by brine rejection as expanded sea ice insulates the ocean from the atmosphere. Because the balance between these two pathways is set by how much of the surface is ice covered, it also governs the response to internal atmospheric variability. The Southern Annular Mode modulates transformation by 3--8 Sv between its positive and negative phases in every climate state, working through heat loss when the ocean is open and through brine rejection when it is ice covered. Weaker glacial transformation coincides with a collapse of the simulated abyssal overturning cell from about 10 to 2 Sv and with older abyssal water. Ideal ages at 4000 m remain below 500 years during interglacials but exceed 750 years basin-wide during glacials, reaching 1500 years in the Pacific. Our results provide a mechanistic framework for interpreting past Southern Ocean circulation, while also showing that the simulated pre-industrial state transforms water at lighter densities than observations indicate.""",
  'abstract')

# =====================================================================
# INTRO — R2/R3: define acronyms at first use in Results; R1#5 & R3: expand SAM intro
# =====================================================================
E(r"""Beyond the mean state, internal atmospheric variability, e.g., the Southern Annular Mode (SAM), modulates present-day Southern Ocean mixed-layer depth and air-sea heat exchange \cite{Sallee2010}, but whether it remains a robust modulator of dense water formation under different climate boundary conditions is not known.""",
  r"""Beyond the mean state, dense water formation also responds to internal atmospheric variability. The Southern Annular Mode is the leading mode of extratropical Southern Hemisphere circulation variability, describing a meridional redistribution of atmospheric mass between mid-latitudes and Antarctica \cite{Marshall2003,Thompson2000}. Its positive phase strengthens the circumpolar westerlies and displaces them poleward, which changes Ekman divergence, air-sea heat exchange, and the northward export of sea ice. In the present-day ocean these adjustments modulate mixed-layer depth and surface buoyancy forcing \cite{Sallee2010}, and they have been linked to observed changes in bottom water properties \cite{Zhou2023,Silvano2020}. Whether the mode remains an effective modulator of dense water formation once the mean state changes, in particular once an expanded ice cover separates the ocean from the atmosphere, is not known.""",
  'sam-intro')

# Chen et al. 2025 must appear in the Introduction as prior work on the same model family
E(r"""Indeed, existing paleoclimate modeling studies have largely diagnosed circulation changes through overturning streamfunctions, water mass volumes, or deep-ocean tracer distributions \cite{Weber2007,Wainer2012}, rather than by explicitly decomposing the surface buoyancy fluxes that drive those changes.""",
  r"""Indeed, existing paleoclimate modeling studies have largely diagnosed circulation changes through overturning streamfunctions, water mass volumes, or deep-ocean tracer distributions \cite{Weber2007,Wainer2012}, rather than by explicitly decomposing the surface buoyancy fluxes that drive those changes. A recent study using the same ocean model in a forced, ocean-only configuration attributes the large glacial volume of Atlantic bottom water to enhanced sea ice export and to weaker mixing between northern- and southern-sourced deep water \cite{Chen2025}. That result identifies sea ice as the central agent from the distribution of water masses and from the overturning streamfunction. It does not measure the surface transformation rates themselves, which is what a decomposed buoyancy budget provides and what allows the thermal and haline pathways to be compared directly and across several climate states.""",
  'chen-intro')

# =====================================================================
# RESULTS 2.1 — R1#6 typo, R1#9 "vanish", R1#10 LGM MLD shallower,
# R1#11 Fig 1i-l undiscussed, R1#12 LIG/LGM nuance, R1#13 Ross Sea, R3 westerlies
# =====================================================================
E(r"""pronounced warming (moer than 2 °C)""",
  r"""pronounced warming (more than 2 °C)""", 'typo-moer')

E(r"""As a result, the deep MLD signals characteristic of open-ocean convection largely vanish and are confined to narrow coastal bands and polynyas (Fig. \ref{reso}).""",
  r"""As a result, the deep mixed layers characteristic of open-ocean convection are strongly reduced and become confined to narrow coastal bands and polynyas (Fig. \ref{reso}). The reduction is substantial rather than complete. Summed south of 55°S, the area with a winter mixed layer deeper than 400 m falls from $7.5\times10^{11}$~m$^2$ in PI to $1.7\times10^{11}$~m$^2$ in LGM and $1.6\times10^{11}$~m$^2$ in MIS3, and the maximum mixed-layer depth decreases from 991 m to 819 m and 768 m. Mixed layers deeper than 600 m, which cover $2.4\times10^{11}$~m$^2$ in PI, essentially disappear in both glacial states. The residual glacial signals in coastal regions are therefore genuinely shallower than their interglacial counterparts, consistent with dense water production that is confined to narrow coastal bands rather than distributed across the open gyres.""",
  'mld-quantified')

E(r"""These thermodynamic changes are accompanied by shifts in atmospheric forcing (Fig. \ref{sst}m-p). While the MH shows moderate changes in zonal wind, the LIG displays localized intensification (Fig. \ref{sst}n). Both LGM and MIS3 exhibit strong positive anomalies in zonal wind speed and wind stress ($>0.04$~N/m$2$) across the circumpolar belt (Fig. \ref{sst}o,p; Fig. \ref{wind}g,h). This intensified wind stress enhances Ekman transport and facilitates the northward expansion of sea ice, fundamentally altering the air-sea interaction interface.""",
  r"""These thermodynamic changes are accompanied by shifts in atmospheric forcing (Fig. \ref{sst}m-p). While the MH shows moderate changes in zonal wind, the LIG displays localized intensification (Fig. \ref{sst}n). Both LGM and MIS3 exhibit strong positive anomalies in zonal wind speed and wind stress ($>0.04$~N/m$^2$) across the circumpolar belt (Fig. \ref{sst}o,p; Fig. \ref{wind}g,h). This intensified wind stress enhances Ekman transport and facilitates the northward expansion of sea ice, fundamentally altering the air-sea interaction interface.

The simulated westerlies strengthen in both the glacial and the last interglacial states, for different reasons. Measured as the latitude and magnitude of the maximum winter zonal-mean 10 m zonal wind, the jet sits at 45.7°S with 5.83~m~s$^{-1}$ in PI and is unchanged in MH. It shifts to 47.6°S and 6.92~m~s$^{-1}$ in LIG, following the orbital configuration of that period, and to 49.4°S with 6.30 and 6.85~m~s$^{-1}$ in LGM and MIS3, following the steepened meridional temperature gradient imposed by the expanded ice sheets and sea ice. That MH remains indistinguishable from PI argues against residual model drift as the origin of these changes. We note that the simulated glacial jet is displaced poleward, whereas proxy reconstructions infer an equatorward shift of about 4.8° and a weakening of roughly 25\% at the Last Glacial Maximum relative to the mid-Holocene \cite{Gray2023}. This disagreement is shared across the PMIP ensemble and we return to it, and to its consequences for our conclusions, in the Discussion.""",
  'westerlies')

# Discuss Fig 1 i-l (density panels), R1#11
E(r"""The combination of extreme cooling and salinification leads to a pronounced increase in surface density across the entire Southern Ocean (Fig. \ref{wind}c,d), particularly in the Pacific sector.""",
  r"""The combination of extreme cooling and salinification leads to a pronounced increase in surface density across the entire Southern Ocean (Fig. \ref{wind}c,d), particularly in the Pacific sector. The surface density anomalies (Fig. \ref{sst}i-l) follow the salinity field rather than the temperature field in the glacial states. Cooling and salinification both act to densify the surface, but at the near-freezing temperatures that prevail south of the ice edge the thermal expansion coefficient is small, so the haline term sets both the pattern and the magnitude. In LIG the warm and salty Bellingshausen-Amundsen anomaly is close to density compensated (Fig. \ref{sst}j), and the LGM retains weak negative density anomalies in parts of the Atlantic-Indian sector where surface freshening outweighs cooling.""",
  'fig1-il')

# R1#12 LIG/LGM sea ice nuance and R1#13 Ross Sea explanation
E(r"""In the interglacial states (PI, MH, LIG),  deep mixed layers ($>$400 m) are observed in the Weddell and Ross Sea gyres (Fig. \ref{mld}a-c), indicating active open-ocean convection.""",
  r"""In the interglacial states (PI, MH, LIG), deep mixed layers ($>$400 m) are observed in the Weddell and Ross Sea gyres (Fig. \ref{mld}a-c), indicating active open-ocean convection. In PI this convection is concentrated in the Weddell sector, which accounts for 70\% of the area with a winter mixed layer deeper than 400 m south of 55°S and for 97\% of the area deeper than 600 m, centred near 68°S in the interior of the gyre rather than over the continental shelf.""",
  'convection-location')

# =====================================================================
# RESULTS 2.2 — R1#14 quantify thermal contribution; R1#15 axis note;
# density-class caveat (the key new framing)
# =====================================================================
E(r"""This transformation is dominated by surface heat loss (red line), with limited sea ice extent restricting the haline contribution from brine rejection to a secondary role (15-30\%; green line). Other freshwater fluxes (blue lines) oppose densification through net precipitation and river runoff.""",
  r"""This transformation is dominated by surface heat loss (red line), which accounts for 70--85\% of the total at the transformation maximum, with limited sea ice extent restricting the haline contribution from brine rejection to a secondary role (15--30\%; green line). Other freshwater fluxes (blue lines) oppose densification through net precipitation and river runoff. We note that this thermally dominated maximum occurs at densities that correspond to intermediate and mode waters rather than to bottom water. In the densest classes that the simulated surface fluxes reach ($\sigma_2 \ge 37.0$~kg~m$^{-3}$), the PI transformation is already haline dominated, with the sea ice term contributing 0.79~Sv against 0.06~Sv from heat in the annual mean. The thermal dominance reported here is therefore a property of the integral over all density classes, and we quantify its sensitivity to the choice of integration domain and density range in the Discussion.""",
  'wmt-thermal-quantified')

E(r"""This interglacial thermal-dominance versus glacial haline-dominance signature is consistent across regional sectors, including the Ross Sea, Weddell Sea, and Ad\'{e}lie regions.""",
  r"""This interglacial thermal-dominance versus glacial haline-dominance signature is consistent across regional sectors, including the Ross Sea, Weddell Sea, and Ad\'{e}lie regions. Note that the vertical axes in Fig. \ref{wmt_jja} are scaled independently for each panel to make the shape of the transformation curves legible; a version with a common vertical axis, which allows the magnitudes to be compared directly between climate states, is provided as Supplementary Fig. \ref{s_common_axis}.""",
  'wmt-regional-axis')

# =====================================================================
# RESULTS 2.4 — R1#22 "across at high latitudes"; R3 where does SAM act
# =====================================================================
E(r"""In contrast, glacial periods (LGM and MIS3) show only weak SAM-driven changes across at high-latitudes, as extensive sea ice coverage limits direct atmosphere-ocean exchange.""",
  r"""In contrast, glacial periods (LGM and MIS3) show only weak SAM-driven heat flux changes south of the ice edge, because the extensive sea ice cover limits direct atmosphere-ocean exchange.""",
  'across-at-1')

E(r"""Together, these results demonstrate that SAM systematically modulates WMT rates across all the 5 simulated climate states through two distinct contributors: thermal forcing, which dominates during interglacials via turbulent heat fluxes, and haline forcing, which dominates during glacials via sea ice processes.""",
  r"""Together, these results demonstrate that the Southern Annular Mode systematically modulates transformation rates in all five simulated climate states through two distinct contributors: thermal forcing, which dominates during interglacials via turbulent heat fluxes, and haline forcing, which dominates during glacials via sea ice processes.

The heat and freshwater composites are shown in their native units in Fig. \ref{sam_heat} and Fig. \ref{sam_fwf}, which makes the two forcings difficult to compare directly. Expressed as equivalent surface density tendencies, the freshwater contribution is the larger of the two wherever sea ice is present, and the heat contribution is larger only in the ice-free sector, which is the same partition that governs the mean state.""",
  'sam-summary')

# =====================================================================
# RESULTS — new subsection on overturning, before ventilation age (R3)
# =====================================================================
E(r"""Having examined surface water mass transformation, we next assess deep ocean ventilation through ideal age distributions at 4000 m depth.""",
  r"""\subsection{Overturning circulation and deep ocean ventilation}

The surface transformation changes described above should be reflected in the overturning circulation itself. We therefore examine the global overturning streamfunction diagnosed online by the model (Fig. \ref{moc}). In the interglacial states the abyssal cell, which represents the northward spreading of southern-sourced bottom water, reaches $-10.0$~Sv in PI, $-10.1$~Sv in MH and $-9.2$~Sv in LIG. In the glacial states it weakens to $-2.4$~Sv in LGM and $-1.6$~Sv in MIS3, a reduction of roughly 75 to 85\%. The upper cell associated with northern-sourced deep water changes comparatively little, from 19.1~Sv in PI to 20.0~Sv in LGM and 21.7~Sv in MIS3, so the glacial reorganisation in these simulations is concentrated in the abyssal cell and expresses a contraction of the southern-sourced cell rather than a wholesale collapse of the overturning.

Having examined surface water mass transformation and the overturning response, we next assess deep ocean ventilation through ideal age distributions.""",
  'moc-subsection')

# Global age discussion (R3 L25, L214)
E(r"""This represents a significant global increase in deep ocean isolation or a slowdown in overturning in glacial times, consistent with radiocarbon proxy reconstructions \cite{Skinner2017}.""",
  r"""This represents a significant global increase in deep ocean isolation or a slowdown in overturning in glacial times, consistent with radiocarbon proxy reconstructions \cite{Skinner2017}.

Because a single depth level gives an incomplete picture of ventilation, we also examine ideal age throughout the global ocean (Fig. \ref{age_global}). Averaged below 2000 m, the ageing is strongly basin dependent. The Atlantic increases only modestly, from 377 years in PI to 439 years in LGM and 508 years in MIS3, because it remains ventilated from the north. The Southern Ocean more than doubles, from 381 to 789 and 844 years, and the Pacific, already the oldest basin in the interglacials at 847 years, reaches 1291 and 1314 years. The zonal-mean sections (Fig. \ref{age_global}f-j) show that the glacial ageing is concentrated below roughly 2000 m and is largest in the deep Pacific, while the upper kilometre changes comparatively little. The contrast between a mildly ageing Atlantic and a strongly ageing Southern Ocean and Pacific is consistent with the abyssal cell weakening while the upper cell is maintained.

We caution that ideal age in these simulations should be read as a lower bound. Each experiment was integrated for 1000 years, which is shorter than the oldest simulated ages, so the glacial age fields are not fully equilibrated and the true equilibrium contrast between climate states would probably be larger. Achieving equilibrated deep-ocean age tracers requires multi-millennial integrations \cite{Millet2025}.""",
  'global-age')

# =====================================================================
# DISCUSSION — the central rewrite (R2 major 1 and 2, R3 main critique)
# =====================================================================
E(r"""For the PI state, our model reproduces comparable annual-mean WMT magnitudes to modern observations (Fig. \ref{s2}). However, a key difference is found in the forcing balance. While Pellichero et al. and Bailey et al. \cite{Pellichero2018,Bailey2023} show haline dominance in the modern seasonal ice zone, our PI results show stronger thermal contributions, but for WMT integrated over the entire Southern Ocean. This discrepancy likely reflect model biases or regional differences in analysis domains.""",
  r"""For the PI state, our model reproduces comparable annual-mean transformation magnitudes to modern observations (Fig. \ref{s2}). The forcing balance, however, differs. Pellichero et al. \cite{Pellichero2018} and Bailey et al. \cite{Bailey2023} find haline dominance in the modern seasonal ice zone, whereas our PI integral is thermally dominated. Because this difference concerns the central claim of the present study, we examined it directly rather than attributing it to unspecified model bias.

Two factors contribute, and they can be separated. The first is the integration domain. Pellichero et al. analyse only the region enclosed by the September sea ice edge, whereas we integrate over everything south of 60°S. In our PI simulation, 42.7\% of that area lies outside the September 15\% ice contour and is water they do not analyse. It is also the part of the domain where the ocean is exposed to the atmosphere all year and where thermal forcing is strongest, and taken alone it is 89--91\% thermally driven in the dense classes. Recomputing the transformation over a sea ice sector defined exactly as theirs, which reproduces their domain to within 2.5\% in area, moves the partition in the direction they report. The thermal share of the dense-class transformation falls from 69\% to 61\% and the sea ice contribution at the transformation maximum roughly doubles, from 0.86 to 1.84~Sv (Fig. \ref{domain}).

The second factor is the density range, and it accounts for most of the remaining difference. Converting our simulated surface properties inside the September ice zone to the neutral density coordinate used by Pellichero et al., our thermally dominated transformation maximum sits near $\gamma_n \approx 27.3$~kg~m$^{-3}$, whereas the haline-dominated lower cell they describe occupies $\gamma_n = 27.9$--$28.8$~kg~m$^{-3}$. Our surface transformation barely reaches the lighter edge of that range. In the densest classes our fluxes do populate ($\sigma_2 \ge 37.0$~kg~m$^{-3}$), the PI transformation is haline dominated, with 0.79~Sv from sea ice against 0.06~Sv from heat. The two analyses are therefore largely describing different water masses, ours weighted toward mode and intermediate densities and theirs toward bottom water densities, and where they overlap they agree in sign. This also indicates that the interglacial thermal pathway identified here is more relevant to the formation of intermediate and mode waters than to bottom water proper.

A residual discrepancy remains, and it reflects a genuine limitation of the model rather than a difference in bookkeeping. We can exclude one candidate explanation: the simulated ice-covered sector is not warm biased in a way that would inflate the thermal term. Inside the September ice zone the area-weighted winter surface temperature is $-1.08$~°C, the median is $-1.71$~°C, 55\% of the area lies within 0.5~°C of the local freezing point, and the corresponding thermal expansion coefficient is $3.5\times10^{-5}$~K$^{-1}$. The near-freezing, weakly thermally sensitive regime invoked by Pellichero et al. is therefore reproduced. What differs is where dense water is made. In the model, PI dense water forms by open-ocean convection in the interior of the Weddell gyre near 68°S, not by the sequence of ice shelf melt, dense shelf water formation in coastal polynyas, and downslope overflow that operates in the real ocean \cite{Orsi1999,ToggweilerSamuels1995}. Because open-ocean convection exposes the water column to the atmosphere over a large area, it recruits an excessive thermal contribution and produces water that is not dense enough, which is precisely the offset in density class documented above.

This bias is shared across the current generation of climate models rather than particular to AWI-ESM2. In the CMIP6 ensemble, 28 of 35 models form deep and bottom water by open-ocean deep convection that is too deep, too frequent, or too extensive, and no model shows convincing shelf export in the Weddell Sea \cite{Heuze2021}; the same behaviour was documented in CMIP5 \cite{Heuze2013}. The present configuration also has no ice shelf cavities, so basal melt beneath floating ice and the associated production of near-freezing shelf water are absent by construction. We state these limitations plainly because they bound what can be claimed. Our results should be read as a statement about how the partitioning of surface buoyancy forcing between heat and freshwater responds to changing sea ice cover, which is a robust property of the simulations, and not as a quantitative reconstruction of past bottom water formation rates.

It is reasonable to ask whether the regime shift is then simply an artefact of a biased interglacial end member, with the glacial state being the more realistic of the two. Two results argue against that reading. First, the shift survives the domain test. Recomputed over the sea ice sector alone, the glacial states remain 15\% and 17\% thermally driven while the interglacials remain 53--88\% thermally driven (Fig. \ref{domain}), so the contrast is not created by including open water in the interglacial integrals. Second, the mechanism is the ice cover itself rather than the convection style. Insulation of the surface from the atmosphere and concentration of brine rejection follow from the areal expansion of sea ice, and both would operate in a model that formed its dense water on the shelf. What such a model would change is the density and the geographic origin of the resulting water, not the direction of the shift in the surface buoyancy budget.""",
  'discussion-pellichero')

# Chen et al. discussion, R2 major #2
E(r"""Though our analysis focuses on surface forcing rather than bottom water formation directly, these results have important implications for understanding how deep water formation mechanisms may respond to climate change (Fig. \ref{schematic}). Our finding that glacial sea ice acts as an amplifier of haline forcing aligns with the sea ice pump hypothesis \cite{Abernathey2016}.""",
  r"""Our conclusions about the glacial state are consistent with recent work using the same ocean model in a forced configuration. Chen et al. \cite{Chen2025} attribute the large glacial volume of Atlantic bottom water to a substantial increase in sea ice export toward lower latitudes, which supplies dense shelf water year round, and to weaker mixing between northern- and southern-sourced water, which allows that water to retain the properties of its origin. The agreement is worth stating explicitly, and the two studies are complementary rather than overlapping. Chen et al. infer the role of sea ice from water mass volumes, the overturning streamfunction, and mixed-layer and sea ice fields, in ocean-only simulations of two states with a prescribed atmosphere. The transformation framework applied here measures the surface buoyancy forcing directly and attributes it to individual flux components in sverdrups, so the contribution of sea ice is quantified rather than inferred from covarying fields. Running the analysis across five coupled climate states, in which sea ice and the atmospheric fluxes evolve together, is also what makes the change of regime visible, since a shift between two forcing regimes cannot be identified from a single glacial state. The present analysis therefore tests and quantifies the mechanism that Chen et al. proposed.

Though our analysis focuses on surface forcing rather than bottom water formation directly, these results have important implications for understanding how deep water formation mechanisms may respond to climate change (Fig. \ref{schematic}). Our finding that glacial sea ice acts as an amplifier of haline forcing aligns with the sea ice pump hypothesis \cite{Abernathey2016}.""",
  'discussion-chen')

# Glacial sea ice / convection: Lhardy; and the wind bias discussion
E(r"""SAM emerges as a persistent modulator of ventilation across all climate states.""",
  r"""The ability of the model to maintain an extensive glacial winter ice cover, and thereby to suppress open-ocean convection, is not a general feature of paleoclimate simulations. Many PMIP models underestimate glacial Southern Ocean sea ice and its seasonality, and colder glacial forcing on its own tends to intensify open-ocean convection rather than suppress it, which degrades the properties of the simulated bottom water; in one systematic study the only configuration that avoided an unrealistically deep northern-sourced cell was the one in which brine sinking along the Antarctic margin was parameterised explicitly \cite{Lhardy2021}. In our simulations the glacial ice cover expands sufficiently to insulate the interior gyres, and the area of deep winter mixed layers contracts to roughly one fifth of its PI value. This behaviour is what produces the haline-dominated glacial regime reported here, and it means our glacial result depends on the simulated sea ice expansion being realistic. Proxy-based reconstructions indicate extensive glacial winter ice, which supports the direction of the simulated change, but the quantitative agreement remains uncertain.

A related limitation concerns the simulated glacial winds. As noted above, the westerlies in our glacial experiments shift poleward and strengthen, whereas proxy reconstructions infer an equatorward shift of about 4.8° and a weakening of about 25\% at the Last Glacial Maximum relative to the mid-Holocene \cite{Gray2023}. The inferred shift exceeds that produced by any member of the PMIP3/4 ensemble, so this is a systematic shortcoming of current models rather than a defect specific to AWI-ESM2, but it does affect how our wind-related results should be read. Two considerations limit the consequences for our conclusions. The mean-state regime shift is driven by the areal extent of sea ice and by brine rejection, not by the position of the jet, and Gray et al. \cite{Gray2023} find no correlation across the model ensemble between Antarctic sea ice extent and either the latitude of the westerlies or the position of the surface temperature front. A bias in jet position therefore does not automatically propagate into the sea ice field on which our main result depends. The Southern Annular Mode results, by contrast, describe the response to a wind perturbation superimposed on a background state whose winds are biased, and they should be read as a statement about the mechanism by which wind variability modulates transformation rather than as a reconstruction of glacial wind-driven variability.

SAM emerges as a persistent modulator of ventilation across all climate states.""",
  'discussion-lhardy-winds')

# R1#27: quantify glacial changes alongside interglacial ones
E(r"""Glacial conditions fundamentally restructure this forcing balance, extensive ice cover insulates the ocean from atmospheric heat exchange, while simultaneously intensifying haline forcing through enhanced brine rejection concentrated in coastal regions.""",
  r"""Glacial conditions fundamentally restructure this forcing balance. The winter transformation maximum weakens to below 60~Sv and moves to $\sigma_2 = 37.0$--$37.5$~kg~m$^{-3}$, and the thermal share of the transformation falls to 12--17\% while the sea ice term becomes dominant. Extensive ice cover insulates the ocean from atmospheric heat exchange, and simultaneously intensifies haline forcing through enhanced brine rejection concentrated in coastal regions.""",
  'glacial-quantified')

# Ventilation age paragraph: spin-up caveat (R3/Millet)
E(r"""While this study is based on a single model, the mechanisms identified here provide a framework for systematic inter-model comparison.""",
  r"""Two caveats apply to this comparison. The ideal age fields are not fully equilibrated after 1000 years of integration, so the simulated glacial ages, and the glacial-interglacial contrast, are best read as lower bounds; equilibrating deep-ocean age tracers requires multi-millennial simulations \cite{Millet2025}. Ideal age also responds to mixing as well as to advection, so it measures the combined effect of a weaker abyssal cell and altered interior mixing rather than the overturning rate alone. The overturning streamfunction shown in Fig. \ref{moc} provides the more direct measure, and the two diagnostics agree in indicating a substantially weaker glacial abyssal cell.

While this study is based on a single model, the mechanisms identified here provide a framework for systematic inter-model comparison.""",
  'age-caveats')

# =====================================================================
# METHODS — boundary condition table (R1 general, R1#32), and domain note
# =====================================================================
E(r"""We perform five equilibrium simulations spanning distinct climate states including three interglacial periods, i.e., pre-industrial (PI), mid-Holocene (MH), and last interglacial (LIG), and two glacial periods, i.e., the Last Glacial Maximum (LGM) and Marine Isotope Stage 3 (MIS3). The boundary conditions are configured following the criteria of PMIP4 \cite{otto2017pmip4}.""",
  r"""We perform five equilibrium simulations spanning distinct climate states including three interglacial periods, i.e., pre-industrial (PI), mid-Holocene (MH), and last interglacial (LIG), and two glacial periods, i.e., the Last Glacial Maximum (LGM) and Marine Isotope Stage 3 (MIS3). The boundary conditions are configured following the criteria of PMIP4 \cite{otto2017pmip4} and are summarised in Table \ref{tab:bc}.""",
  'methods-table-ref')

E(r"""For LGM and MIS3, we fix the boundary conditions at 21 ka and 38 ka respectively, with the topography and ice-sheet properties deriving from the GLAC1D reconstruction \cite{tarasov2003greenland,tarasov2012data,briggs2014data}.""",
  r"""For LGM and MIS3, we fix the boundary conditions at 21 ka and 38 ka respectively, with the topography and ice-sheet properties deriving from the GLAC1D reconstruction \cite{tarasov2003greenland,tarasov2012data,briggs2014data}. The GLAC-1D reconstruction enters the model through several boundary fields simultaneously. It sets the ice sheet extent and surface elevation and the associated land-sea mask and orography used by the atmosphere and land-surface components, the river routing, and the vegetation distribution. It also determines the ocean bathymetry and coastline, which requires a dedicated unstructured ocean mesh for each glacial period; we use meshes constructed for 21 ka and 38 ka respectively, while PI, MH and LIG share the modern-geometry mesh. Global mean ocean salinity is raised in the glacial experiments to account for the water stored in the ice sheets. No prescribed ice sheet meltwater flux is applied in any of the equilibrium simulations, and the model configuration does not include ice shelf cavities, so basal melt beneath floating ice shelves is not represented.""",
  'methods-glac1d')

# Domain justification in methods (R2/R3)
E(r"""These regions encompass the primary Antarctic bottom water (AABW) formation sites and allow regional comparison of formation mechanisms. Regional transformations are calculated by applying spatial masks to the surface forcing fields before integration.""",
  r"""These regions encompass the primary Antarctic bottom water (AABW) formation sites and allow regional comparison of formation mechanisms. Regional transformations are calculated by applying spatial masks to the surface forcing fields before integration.

To compare with observational estimates that are defined over the ice-covered sector, we additionally repeat the calculation over a seasonal sea ice zone, defined as the region where the climatological September sea ice concentration exceeds 15\%, following the domain definition of Pellichero et al. \cite{Pellichero2018}, and over the complementary part of the domain south of 60°S that lies outside that contour. These auxiliary calculations use the climatological monthly means and are otherwise identical to the main calculation. For the density-class comparison with observational studies that use neutral density, simulated surface properties were converted with the TEOS-10 routines.

\begin{table}[htbp]
\centering
\caption{Boundary conditions prescribed in the five equilibrium simulations. Orbital parameters for PI are those of 1850~CE. Greenhouse gas concentrations are volume mixing ratios.\label{tab:bc}}
\begin{tabular}{lccccccc}
\toprule
 & CO$_2$ & CH$_4$ & N$_2$O & Eccentricity & Obliquity & Perihelion & Ice sheet / \\
 & (ppm) & (ppb) & (ppb) & & (°) & (°) & bathymetry \\
\midrule
PI   & 284.3 & 808.2 & 273.0 & 1850~CE  & 1850~CE & 1850~CE & modern \\
MH   & 264.4 & 597.0 & 262.0 & 0.018682 & 24.105  & 180.87  & modern \\
LIG  & 275.0 & 685.0 & 255.0 & 0.039378 & 24.040  & 95.41   & modern \\
LGM  & 190.0 & 375.0 & 200.0 & 0.018994 & 22.949  & 294.42  & GLAC-1D 21~ka \\
MIS3 & 210.5 & 556.2 & 247.4 & 0.013676 & 23.2591 & 25.99   & GLAC-1D 38~ka \\
\botrule
\end{tabular}
\end{table}""",
  'methods-domain-table')

# =====================================================================
# DATA / CODE AVAILABILITY  (editorial requirement)
# =====================================================================
E(r"""\section*{Data availability}
The data that support the findings of this study are available from the corresponding author upon reasonable request.""",
  r"""\section*{Data availability}
The model output required to reproduce every figure in this manuscript, comprising climatological monthly means for the five experiments, the 100-year monthly water mass transformation time series, the Southern Annular Mode composite fields, and the ocean mesh files, has been deposited in a public repository (DOI to be inserted upon acceptance). Source data for the line-graph figures are provided with this paper. The raw model output totals several tens of terabytes and is archived at the German Climate Computing Centre; it is available from the corresponding author on reasonable request.""",
  'data-avail')

E(r"""Analysis scripts specific to this study are available from the corresponding author upon reasonable request.""",
  r"""The analysis and plotting scripts specific to this study, covering the surface buoyancy flux decomposition, the transformation calculations including the sea ice sector comparison, the Southern Annular Mode compositing, and all figures, are archived together with the data deposit and are also available at \url{https://github.com/xshi-awi/aabw_5exps}.""",
  'code-avail')

# =====================================================================
# apply
# =====================================================================
missing = []
for old, new, label in edits:
    n = s.count(old)
    if n != 1:
        missing.append((label, n))
        continue
    s = s.replace(old, new, 1)

if missing:
    print('FAILED to apply uniquely:')
    for label, n in missing:
        print(f'  {label}: found {n} times')
    sys.exit(1)

DST.write_text(s)
print(f'applied {len(edits)} edits -> {DST}')
