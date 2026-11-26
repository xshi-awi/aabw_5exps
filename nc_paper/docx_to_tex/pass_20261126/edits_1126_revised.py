#!/usr/bin/env python3
"""2026-11-26 pass: manuscript edits that follow the user's 11-26 notes in response_new.tex
(TexPage bb5f6eb). Applies to nc_paper/build/revised.tex.

Every anchor must match exactly once; the script refuses to write otherwise.
Usage: python3 edits_1126_revised.py <revised.tex>
notbut_1126.json (same folder) holds the not-X-but-Y / rather-than rewrites (user rule of 11-26); entries whose
anchor was already rewritten by this script are skipped and listed.
"""
import json
import re
import sys

tex_path = sys.argv[1]
T = open(tex_path, encoding='utf8').read()


def rep(old, new, count=1):
    global T
    n = T.count(old)
    assert n == count, f'anchor found {n} times, expected {count}: {old[:90]!r}'
    T = T.replace(old, new)


def rep_para(start, new):
    """Replace the whole line (paragraph) that starts with `start`."""
    global T
    lines = T.split('\n')
    hit = [i for i, l in enumerate(lines) if l.startswith(start)]
    assert len(hit) == 1, f'paragraph start found {len(hit)} times: {start[:80]!r}'
    lines[hit[0]] = new
    T = '\n'.join(lines)


# ---------------------------------------------------------------- 2.3: meltwater sentence (letter L195 edit)
rep(r"""in these simulations. We note that this statement applies to the freshwater sources represented in our configuration. In the present-day ocean,""",
    r"""in these simulations. In the present-day ocean,""")

# ---------------------------------------------------------------- 2.2: in the model (R3 L152, user wording)
rep(r"""During interglacial periods, dense water formation is predominantly thermally driven. In the austral winter""",
    r"""In our model, dense water formation is predominantly thermally driven in the interglacial experiments. In the austral winter""")
rep(r"""We note that this thermally dominated maximum occurs at densities that correspond to intermediate and mode waters rather than to bottom water.""",
    r"""We note that this thermally dominated maximum lies at the densities of intermediate and mode waters, which are lighter than bottom water.""")

# ---------------------------------------------------------------- 2.4: push and pull, A10, state dependence
rep(r"""It simultaneously increases the coastal brine input, by 51 and 42~mSv in PI and LIG and by 24 and 31~mSv in LGM and MIS3,""",
    r"""It simultaneously increases the coastal brine input, by 51, 12 and 42~mSv in PI, the mid-Holocene and the Last Interglacial and by 24 and 31~mSv in LGM and MIS3,""")

rep_para(r"""The spatial patterns show why. Strengthened westerlies""",
         r"""The spatial patterns are shown in Supplementary Fig. \ref{sam_windice}. Under the positive phase the westerly wind stress strengthens over the whole circumpolar belt in every state (Supplementary Fig. \ref{sam_windice}a,c,e,g,i). The sea ice concentration changes mainly near the winter ice edge (Supplementary Fig. \ref{sam_windice}b,d,f,h,j). Gains and losses alternate between sectors along the edge, and the net winter ice area grows in all five states, by 0.07 to 0.57 million km$^2$. Near the coast the concentration hardly changes, because the winter pack there stays close to full cover. The extra brine release near the coast therefore shows up in the freshwater flux (Supplementary Fig. \ref{sam_pushpull}b) and leaves little trace in the ice concentration. In the glacial states the ice edge response lies further north, following the expanded ice edge, and it is about twice as large as in the interglacial states.""")

rep_para(r"""An asymmetry between the two routes emerges when the mean state changes.""",
         r"""The size of the two responses differs between the climate states, and it does not separate the interglacial from the glacial states. The Ekman response is 11.2, 8.9 and 12.8~Sv in PI, the mid-Holocene and the Last Interglacial and 8.0 and 10.1~Sv in LGM and MIS3 (Supplementary Fig. \ref{sam_pushpull}a). The brine response ranks the states in a similar order (Supplementary Fig. \ref{sam_pushpull}b), which gives the correlation noted above. What changes between interglacial and glacial states is the local balance of the anomalous forcing, which we describe next.""")

# ---------------------------------------------------------------- 3: ideal age paragraph (letter quote edited by the user)
rep_para(r"""Because a single depth level gives an incomplete picture of ventilation, we also examine ideal age""",
         r"""We now examine ideal age throughout the global ocean (Fig. \ref{age_global}). Averaged below 2000 m, the age is strongly basin dependent. The Atlantic increases only modestly, from 377 years in PI to 439 years in LGM and 508 years in MIS3, because it remains ventilated from the north. The Southern Ocean more than doubles, from 381 to 789 and 844 years, and the Pacific, already the oldest basin in the interglacials at 847 years, reaches 1291 and 1314 years. The zonal-mean sections (Fig. \ref{age_global}f-j) show that the glacial age increase is concentrated below roughly 2000 m and is largest in the deep Pacific, while the upper ocean changes comparatively little. The contrast between a mildly ageing Atlantic and a strongly ageing Southern Ocean and Pacific is consistent with the abyssal cell weakening while the upper cell is maintained.""")

# ---------------------------------------------------------------- Discussion: biased interglacial end member (user: no not/but, unclear)
rep_para(r"""It is reasonable to ask whether the regime shift is then simply an artefact""",
         r"""Because the interglacial dense water forms by open-ocean convection in the model, one may ask whether the regime shift simply reflects a biased interglacial state, with the glacial state being the more realistic of the two. Two results make this unlikely. The shift remains when the analysis is restricted to the sea ice zone. There the glacial states are 15\% and 17\% thermally driven and the interglacial states 53--70\% (Fig. \ref{domain}), so the contrast does not come from the open water that the interglacial integrals include. The shift also follows directly from the sea ice cover. Where ice covers the ocean, it shields the surface from heat loss to the atmosphere, while brine is still released where new ice grows. Both effects would also act in a model that forms its dense water on the continental shelf. Such a model would probably produce denser water in other places, and its surface buoyancy forcing would still move from heat towards brine as the ice cover expands. The glacial simulations are not free of bias either, and proxy data constrain them less well than the modern ocean. As is common in paleoclimate modelling, we therefore focus on the differences between each past state and PI, in which biases shared by both states partly cancel.""")

# ---------------------------------------------------------------- Discussion: SAM paragraph (state dependence, A10, sediments)
rep(r"""In its positive phase it enhances turbulent heat loss over open water and simultaneously drives sea ice divergence that promotes brine rejection near the coast, with composite anomalies""",
    r"""In its positive phase it enhances turbulent heat loss over open water and increases brine rejection near the coast, with composite anomalies""")
rep(r"""The two effects are not independent expressions of the same forcing but the two halves of a single mechanism. A positive anomaly""",
    r"""The two effects act together. A positive anomaly""")
rep(r"""would suggest. The dominant pathway shifts from thermal during interglacials to haline during glacials, mirroring the mean-state regime shift described above. This circumpolar picture""",
    r"""would suggest. This circumpolar picture""")
rep(r"""The robustness of this coupling across dramatically different background conditions indicates that atmosphere-ocean-sea ice interactions operate through consistent physical mechanisms regardless of mean climate state, although the balance between the two routes shifts. The Ekman response to a given phase anomaly is weaker in the glacial states than in the interglacials despite a stronger mean wind stress, because extensive ice cover damps the transmission of stress to the ocean, while the haline route strengthens. The net modulation is of comparable magnitude in every state, but it is delivered through a different pathway, which mirrors the mean-state regime shift.""",
    r"""The coupling holds in all five states, although their background conditions differ strongly. The size of the Ekman and brine responses does not separate the interglacial from the glacial states, and the transformation anomaly is of similar size in every state. What changes is the local balance of the anomalous forcing. Over open water the heat term dominates, and under sea ice the freshwater term dominates, so the expanded glacial ice cover shifts the response towards brine release in the same way as it shifts the mean state.""")
rep(r"""Whether this persistent coupling could be exploited for proxy reconstruction is less clear than we previously suggested.""",
    r"""Whether this coupling could be used for proxy reconstruction is unclear.""")
rep(r"""We therefore present the persistence of the coupling as a property of the simulated system rather than as a practical route to a marine Southern Annular Mode reconstruction.""",
    r"""We therefore present the persistence of the coupling as a property of the simulated system. We do not expect it to be recoverable from marine sediments.""")

# ---------------------------------------------------------------- A10 caption (the old dipole description is not what the panels show)
rep(r"""Under the positive phase the westerlies strengthen over the circumpolar belt in every state, and the sea ice response is a dipole with a loss near the coast, where polynyas open and brine is rejected, and a gain further north where the exported ice melts. The dipole is displaced northward and is substantially stronger in the glacial states, consistent with the shift of the transformation response from the thermal to the haline pathway. \label{sam_windice}}""",
    r"""Under the positive phase the westerlies strengthen over the circumpolar belt in every state. The sea ice concentration changes mainly near the winter ice edge, with gains and losses alternating between sectors and a net gain of ice area in every state. Near the coast the concentration hardly changes because the winter pack stays close to full cover there. The ice edge response lies further north and is about twice as large in the glacial states. \label{sam_windice}}""")

# ---------------------------------------------------------------- Supplementary Fig. A2: MLD colour scale to 1000 m in a-c (R3 L134)
rep(r"""\includegraphics[width=\textwidth]{figures/fig02_climate_mld_seaice_jja.pdf}""",
    r"""\includegraphics[width=\textwidth]{figures/fig02_climate_mld_seaice_jja_mld1000.pdf}""")
rep(r"""and (e,j) MIS3.  \label{mld}}""",
    r"""and (e,j) MIS3. The mixed layer depth colour scale runs to 1000 m in (a-c) and to 400 m in (d,e). \label{mld}}""")

# ==== PART2 (westerlies, PMIP, formation location, drift, MH/LIG spelled out) ====
# ---------------------------------------------------------------- 2.1 and Discussion: PMIP4 westerlies (explore/pmip_westerlies)
rep(r"""This disagreement is shared across the PMIP ensemble and we return to it, and to its consequences for our conclusions, in the Discussion.""",
    r"""The four PMIP4 models with Last Glacial Maximum output available to us \cite{Kageyama2021} move the winter 10 m wind maximum by less than 1.2° between LGM and PI and change its strength by 6\% or less. They do not reproduce the reconstructed shift either, and our simulated poleward shift is larger than in any of them. We return to this disagreement, and to its consequences for our conclusions, in the Discussion.""")

rep_para(r"""A related limitation concerns the simulated glacial winds.""",
         r"""A related limitation concerns the simulated glacial winds. The westerlies in our glacial experiments shift poleward and strengthen, whereas proxy reconstructions infer an equatorward shift of about 4.8° and a weakening of about 25\% at the Last Glacial Maximum relative to the mid-Holocene \cite{Gray2023}. Gray et al. \cite{Gray2023} found that the inferred shift exceeds that of any PMIP3/4 model. In the four PMIP4 models with Last Glacial Maximum output available to us, the winter wind maximum moves by less than 1.2° between LGM and PI, so current models in general do not reproduce the reconstructed change. Our model departs further in the opposite direction, with a poleward shift of 3.7°, which is two grid rows of the atmosphere model. This affects the two parts of our results differently. The mean-state regime shift depends on the area of sea ice and on brine rejection. Gray et al. \cite{Gray2023} found no correlation across the model ensemble between Antarctic sea ice extent and either the latitude of the westerlies or the position of the surface temperature front. A bias in the position of the westerlies therefore need not carry over into the sea ice field on which our main result rests. The Southern Annular Mode results describe how the model responds to a wind anomaly added to a background state whose mean winds are biased. We therefore read them as a description of the mechanism by which wind variability changes transformation, and we do not use them to reconstruct glacial wind-driven variability.""")

# ==== PART3 (westerly mechanism, formation location, drift) ====
# ---------------------------------------------------------------- 2.1: why the westerlies strengthen (explore/westerly_mechanism, R3 L144)
rep(r"""It shifts to 47.6°S and 6.92~m~s$^{-1}$ in LIG, following the orbital configuration of that period, and to 49.4°S with 6.30 and 6.85~m~s$^{-1}$ in LGM and MIS3, following the steepened meridional temperature gradient imposed by the expanded ice sheets and sea ice. That MH remains""",
    r"""It shifts to 47.6°S and 6.92~m~s$^{-1}$ in LIG and to 49.4°S with 6.30 and 6.85~m~s$^{-1}$ in LGM and MIS3. In the glacial states the change follows the expanded sea ice. The winter ice edge moves from 66°S in PI to 55--57°S, and the strongest near-surface temperature gradient moves with it from about 65°S to 60--62°S and becomes 24--29\% steeper. This gradient still lies 10--12° south of the wind maximum. The lower atmosphere therefore becomes more baroclinic on the poleward side of the westerlies, and the Eady growth rate between 850 and 500~hPa increases by 21--28\% between 45 and 65°S. The tropical upper troposphere also cools more than the surface, which weakens the temperature gradient aloft in mid-latitudes. Both changes favour stronger westerlies further south. In LIG the change comes from the orbital forcing. Winter insolation increases by 26--33~W~m$^{-2}$ between 30 and 45°S, the tropical upper troposphere warms by about 2~K, and the temperature gradient aloft steepens, while the sea ice edge retreats by about 1°. The westerlies therefore strengthen with a small poleward shift. Five equilibrium states cannot separate the roles of sea ice, ice sheets and greenhouse gases, which would require atmosphere-only sensitivity experiments. That MH remains""")

# ==== PART4 (formation location, drift, ice-shelf runoff) ====
# ---------------------------------------------------------------- Discussion + Supplementary Fig. A13: where each density class forms (fig_formation_location, R3 main comment)
def insert_after_para(start, new):
    global T
    lines = T.split('\n')
    hit = [i for i, l in enumerate(lines) if l.startswith(start)]
    assert len(hit) == 1, f'paragraph start found {len(hit)} times: {start[:80]!r}'
    lines[hit[0]:hit[0] + 1] = [lines[hit[0]], '', new]
    T = '\n'.join(lines)


insert_after_para(r"""A residual discrepancy remains""",
    r"""To see where each density class is produced, we split the winter transformation south of 60°S by region (Supplementary Fig. \ref{formation_loc}). In the interglacial states the class of the transformation maximum forms mainly over ice-free deep water, which carries 69\% of it, while the continental shelf and slope, where the bottom is shallower than 1000~m, carry 20--24\%. The densest classes that each interglacial state produces carry 3.0--7.4~Sv and form mainly on the shelf and slope (61--88\%). The rest of them forms in the open water of the Weddell sector. Water as dense as the simulated PI bottom water is produced mostly offshore, with 13\% from the shelf. The model therefore does form dense water near the coast, mostly where the winter pack ice is open. In the interglacial states this coastal water is denser and much smaller in volume than the water made offshore, and the simulated PI bottom water has the density of the offshore water. In the glacial states the class of the transformation maximum forms under sea ice offshore (89--91\%), and the densest classes form on the shelf (53--59\%) and under the ice further offshore. The glacial bottom water is denser than 99\% of the winter surface water, and the 0.7--1.0~Sv that reach its density all form on the shelf. Because the density of bottom water differs between the climate states, a single density threshold cannot isolate the source of bottom water across all five states. For this reason our analysis covers the transformation of all water south of 60°S. Surface transformation shows where water becomes dense, and it does not show where that water ends up.""")

rep(r"""\end{figure}

\end{appendices}""",
    r"""\end{figure}

\begin{figure}
\centering
\includegraphics[width=\textwidth]{figures/figS_formation_location.pdf}
\caption{Where the simulated winter (JJA) surface transformation south of 60°S takes place. (a) Surface density flux that transforms water into the densest classes of each state, that is, the densities beyond which the transformation stays below 10\% of its maximum (threshold below each panel), on a logarithmic scale. (b) Surface density flux that transforms water across the density class of the transformation maximum. (c) Transformation in the class of the maximum, split into the continental shelf and slope (bottom depth shallower than 1000 m), deep water with a winter mixed layer deeper than 400 m, deep water under sea ice and ice-free deep water. (d) The same split for the densest classes. (e) The densest classes split by distance from the coast, with the part within 200 km of the coast divided at a winter sea ice concentration of 0.7. Lines in (a,b) mark the 1000 m isobath, the 15\% winter sea ice edge, the 400 m winter mixed layer depth and 60°S. Units: kg m$^{-2}$ month$^{-1}$ in (a,b) and Sv in (c-e). \label{formation_loc}}
\end{figure}

\end{appendices}""")

# ---------------------------------------------------------------- Methods: SST and AMOC drift (explore/drift_surface_amoc, R3 drift comment)
rep(r"""A century-scale drift of 0.1~K is therefore two orders of magnitude smaller than the signal and cannot account for it.""",
    r"""A century-scale drift of 0.1~K is therefore more than an order of magnitude smaller than the signal and cannot account for it.""")
rep(r"""which argues against residual drift as the origin of the wind changes reported in the other paleoclimate states.""",
    r"""which argues against residual drift as the origin of the wind changes reported in the other paleoclimate states. The ocean surface and the Atlantic overturning drift little as well. Over the analysis period the trend of global mean sea surface temperature lies between $-0.03$ and $+0.06$~K~century$^{-1}$ in all five states, against glacial minus PI differences of $-1.8$ and $-1.5$~K. The trend of the Atlantic overturning, taken as the maximum of the Atlantic streamfunction between 500 and 4000~m depth and between 20 and 70°N, lies between $-0.6$ and $+1.3$~Sv~century$^{-1}$. This is comparable to its year-to-year variability of 1.0--1.5~Sv and small against the glacial strengthening of 4.6--5.3~Sv relative to PI. The trend of the abyssal cell lies between $-1.0$ and $+0.6$~Sv~century$^{-1}$, against a glacial weakening of about 9~Sv.""")

# ---------------------------------------------------------------- Methods and 2.3: how the ice sheet's freshwater reaches the ocean (R3 ice-shelf comment)
rep(r"""so basal melt beneath floating ice shelves is not represented.""",
    r"""so basal melt beneath floating ice shelves is not represented. Precipitation minus evaporation over the Antarctic ice sheet is routed to the coast by the hydrological discharge model and enters the ocean as liquid runoff, about 0.085~Sv south of 60°S in PI and 0.07~Sv in the glacial states. This runoff stands in for the freshwater that the ice sheet would otherwise lose by basal melt and iceberg calving.""")
rep(r"""Because the model has no ice shelf cavities and no prescribed meltwater flux, that term is absent here, and the freshwater budget we decompose is dominated by sea ice thermodynamics.""",
    r"""Because the model has no ice shelf cavities and no prescribed meltwater flux, basal melt and iceberg melt are not represented as such. The freshwater that leaves the ice sheet returns to the ocean as liquid runoff at the coast (Methods), and the freshwater budget we decompose is dominated by sea ice thermodynamics.""")


# ---------------------------------------------------------------- not-but sweep (user rule 11-26), reviewed list in notbut_1126.json
import os
_here = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(_here, 'notbut_1126.json'), encoding='utf8'))
skipped = []
for p in P['revised']:
    n = T.count(p['old'])
    if n == 1:
        T = T.replace(p['old'], p['new'])
    else:
        skipped.append((n, p['old'][:70]))
print(f'not-but sweep: {len(P["revised"]) - len(skipped)} applied, {len(skipped)} skipped (already rewritten above)')
for s_ in skipped:
    print('   skipped', s_)

# ---------------------------------------------------------------- MH and LIG spelled out in the text (R3 minor, user 11-26); keep last
sys.path.insert(0, _here)
import mhlig
T = mhlig.apply_to_tex(T)
assert not mhlig.leftovers(T), mhlig.leftovers(T)

open(tex_path, 'w', encoding='utf8').write(T)
print('revised edits OK')
