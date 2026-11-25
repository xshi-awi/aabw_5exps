#!/usr/bin/env python3
"""2026-11-25 (afternoon pass, FV): manuscript edits asked for in the user's % notes in
response_new.tex and by the manager (questions-14). Applies to nc_paper/build/revised.tex
and adds verified references to nc_paper/build/ref.bib.

Every anchor must match exactly once; the script refuses to write otherwise.
Usage: python3 edits_1125c_revised.py <revised.tex> <ref.bib>
"""
import re
import sys

tex_path, bib_path = sys.argv[1], sys.argv[2]
T = open(tex_path, encoding='utf8').read()
B = open(bib_path, encoding='utf8').read()


def rep(old, new, count=1):
    global T
    n = T.count(old)
    assert n == count, f'anchor found {n} times, expected {count}: {old[:90]!r}'
    T = T.replace(old, new)


# ---------------------------------------------------------------- Introduction: SAM paragraph
rep(r""" Beyond the mean state, dense water formation also responds to internal atmospheric variability. The Southern Annular Mode is the leading mode of extratropical Southern Hemisphere circulation variability, describing a meridional redistribution of atmospheric mass between mid-latitudes and Antarctica \cite{Marshall2003,Thompson2000}. Its positive phase strengthens the circumpolar westerlies and displaces them poleward, which changes Ekman divergence, air-sea heat exchange, and the northward export of sea ice. In the present-day ocean these adjustments modulate mixed-layer depth and surface buoyancy forcing \cite{Sallee2010}, and they have been linked to observed changes in bottom water properties \cite{Zhou2023,Silvano2020}. Whether the mode remains an effective modulator""",
    r"""

Beyond the mean state, dense water formation also responds to internal atmospheric variability. The Southern Annular Mode (SAM) is the leading mode of extratropical Southern Hemisphere circulation variability, describing a meridional redistribution of atmospheric mass between mid-latitudes and Antarctica \cite{Gong1999,Thompson2000,Marshall2003}. Its positive phase strengthens the circumpolar westerlies and displaces them poleward, which changes Ekman divergence, air-sea heat exchange, and the northward export of sea ice \cite{HallVisbeck2002,SenGupta2006}. In the present-day ocean these adjustments modulate mixed-layer depth and surface buoyancy forcing \cite{Sallee2010,SenGupta2006}. Surface buoyancy forcing in turn sets the rate at which surface waters are transformed into denser classes \cite{Cerovecki2013,Abernathey2016}. The SAM has also been linked to observed changes in bottom water properties \cite{Zhou2023,Silvano2020}. Whether the SAM remains an effective modulator""")

# ---------------------------------------------------------------- 2.1: months of Fig. 1, sea ice of Fig. 1i-l
rep(r"""during austral winter (JJA), the primary season for this process in the Southern Ocean. %""",
    r"""during austral winter, the primary season for this process in the Southern Ocean. The maps in Fig. \ref{sst} are averaged over June to September. %""")

rep(r"""where surface freshening outweighs cooling.


In the interglacial states (PI, MH, LIG), the model produces""",
    r"""where surface freshening outweighs cooling.

The sea ice anomalies (Fig. \ref{sst}i-l) follow the temperature changes. In MH the changes are small. In LIG the winter sea ice decreases strongly in the Ross sector, where the winter ice edge retreats south by about 5° of latitude beneath the warm and salty surface anomaly. In LGM and MIS3 the winter sea ice area expands substantially and the ice edge moves north by about 10--15° of latitude, most of all in the Weddell sector.


In the interglacial states (PI, MH, LIG), the model produces""")

# comment 9: the wording of the letter goes into the manuscript
rep(r"""As a result, the deep mixed layers characteristic of open-ocean convection are strongly reduced and become confined to narrow coastal bands and polynyas (Fig. \ref{reso}).""",
    r"""As a result, the mixed layer depth is strongly reduced and becomes confined to narrow coastal bands and polynyas (Fig. \ref{reso}).""")
rep(r"""genuinely shallower than their interglacial counterparts, consistent with""",
    r"""genuinely shallower than their interglacial counterparts (Supplementary Fig. \ref{s1}), consistent with""")

# comments 12 and 13: winds as in the letter (checked against the June-September winds of Fig. 1),
# sea ice by direction only and the LIG passage on the Ross sector only (user, 2026-11-25)
rep(r"""These thermodynamic changes are accompanied by shifts in atmospheric forcing (Fig. \ref{sst}m-p). While the MH shows moderate changes in zonal wind, the LIG displays localized intensification (Fig. \ref{sst}n). Both LGM and MIS3 exhibit strong positive anomalies in zonal wind speed and wind stress ($>0.04$~N/m$^2$) across the circumpolar belt (Fig. \ref{sst}o,p; Fig. \ref{wind}g,h).""",
    r"""These thermodynamic changes are accompanied by shifts in atmospheric forcing (Fig. \ref{sst}m-p). The MH shows moderate anomalies in zonal wind, generally negative south of about 50°S and positive to the north. The LIG shows a similar pattern but with a strong intensification in the Pacific sector (Fig. \ref{sst}n). Both LGM and MIS3 exhibit strong positive anomalies in zonal wind speed and wind stress ($>0.04$~N/m$^2$) across the circumpolar belt, although the LGM also shows some negative anomalies in the Atlantic-Indian sector (Fig. \ref{sst}o,p; Fig. \ref{wind}g,h).""")
rep(r"""In the glacial states the sea ice response to this wind intensification is in fact fairly uniform around the continent. Averaged between 60 and 75°S, winter ice concentration increases by 0.62 in the Ross sector, 0.70 in the Weddell sector and 0.53 in the Ad\'elie sector at the Last Glacial Maximum, and all three sectors cool to within about 0.1~K of the surface freezing point, so ice growth is no longer limited by the surface heat content anywhere. The Ross sector instead stands out in the last interglacial, where the surface ocean between 60 and 75°S is 0.97~K warmer than in PI and sits 2.72~K above the freezing point, against changes below 0.3~K in the other two sectors. Winter ice concentration there falls by 0.19, an order of magnitude more than the Weddell or Ad\'elie response. In that state""",
    r"""In the glacial states the sea ice increases all around the continent. Averaged between 60 and 75°S, winter ice concentration increases in the Ross, Weddell and Ad\'elie sectors at the Last Glacial Maximum, and all three sectors cool to within about 0.1~K of the surface freezing point, so ice growth is no longer limited by the surface heat content anywhere. The Ross sector instead stands out in the last interglacial. There the surface ocean between 60 and 75°S is warmer than in PI and sits well above the freezing point, and winter ice concentration falls. In that state""")

# Reviewer 3 on Gray et al. (2023): near-surface westerlies, not "jet"; add the wind stress they reconstruct
rep(r"""the jet sits at 45.7°S with 5.83~m~s$^{-1}$ in PI""", r"""the wind maximum sits at 45.7°S with 5.83~m~s$^{-1}$ in PI""")
rep(r"""We note that the simulated glacial jet is displaced poleward, whereas proxy reconstructions infer an equatorward shift of about 4.8° and a weakening of roughly 25\% at the Last Glacial Maximum relative to the mid-Holocene \cite{Gray2023}. This disagreement""",
    r"""We note that the simulated glacial westerlies are displaced poleward, whereas proxy reconstructions infer an equatorward shift of about 4.8° and a weakening of roughly 25\% at the Last Glacial Maximum relative to the mid-Holocene \cite{Gray2023}. The surface wind stress, the quantity these reconstructions target, shows the same poleward shift in our simulations. Its annual zonal-mean maximum over the ocean lies at 47.6°S with 0.172~N~m$^{-2}$ in MH and at 49.4°S with 0.195~N~m$^{-2}$ in LGM, one grid row (about 1.9°) further south and about 13\% stronger. This disagreement""")

# ---------------------------------------------------------------- 2.3: unit as a surface density flux (Figs. 3, 4)
rep(r"""We express it as the density change of a surface layer 1 m thick, in kg~m$^{-3}$~s$^{-1}$ per metre of water column (see Methods for more details).""",
    r"""We express it as a surface density flux in kg~m$^{-2}$~month$^{-1}$, which equals the density change of a surface layer 1 m thick over one month (see Methods).""")
rep(r"""with maximal values reaching $5 \times 10^{-6}$~kg~m$^{-3}$~s$^{-1}$ in the circumpolar band""",
    r"""with maximal values reaching about 13~kg~m$^{-2}$~month$^{-1}$ in the circumpolar band""")
rep(r"""Anomalies reach $-2 \times 10^{-6}$~kg~m$^{-3}$~s$^{-1}$ in the Pacific sector.""",
    r"""Anomalies reach about $-5$~kg~m$^{-2}$~month$^{-1}$ in the Pacific sector.""")
rep(r"""reach about $+2.5 \times 10^{-6}$~kg~m$^{-3}$~s$^{-1}$ per metre of water column in a circumpolar band""",
    r"""reach about $+6.5$~kg~m$^{-2}$~month$^{-1}$ in a circumpolar band""")
rep(r"""where they exceed $4 \times 10^{-6}$~kg~m$^{-3}$~s$^{-1}$ over a tenth of the area south of 65°S and locally reach $1.8 \times 10^{-5}$~kg~m$^{-3}$~s$^{-1}$.""",
    r"""where they exceed 10~kg~m$^{-2}$~month$^{-1}$ over a tenth of the area south of 65°S and locally reach about 47~kg~m$^{-2}$~month$^{-1}$.""")
# comment 18, wording of the letter
rep(r"""show modest changes relative to PI. MH exhibits no obvious thermal density tendency anomalies, while
LIG exhibits""",
    r"""show small to modest changes relative to PI north of the ice-covered region all around Antarctica. MH exhibits no obvious thermal density tendency anomalies in the ice-covered region, while
LIG exhibits""")

# ---------------------------------------------------------------- 2.4: SAM section
# the definition moved to the Introduction (Reviewer 3, L206-209); thresholds only in the captions (user, 09:10)
rep(r"""The Southern Annular Mode (SAM) is the dominant mode of atmospheric variability in the Southern Hemisphere extratropics, characterized by a meridional pressure gradient between mid-latitudes and Antarctica that modulates the strength and position of the circumpolar westerlies. This atmospheric variability superimposes on mean climate states   and exerts primary control on surface buoyancy forcing, thereby governing WMT rates. Through composite analysis comparing high-SAM versus low-SAM years (defined as JJA-mean SAM index exceeding $\pm$1.2$\sigma$ from the climatological mean) using 100-year monthly simulations, we examine how this atmospheric forcing modulates dense water formation across different climate states and regional sectors. The supporting composite fields are collected in Supplementary Figs. \ref{sam_pushpull}--\ref{sam_wmt}.""",
    r"""We now examine how the SAM, introduced above, modulates dense water formation across different climate states and regional sectors. We compare high-SAM and low-SAM winters in the 100-year monthly simulations of each state. High-SAM (low-SAM) years are those in which the JJA-mean SAM index lies more than a fixed multiple of its standard deviation above (below) its climatological mean, and the threshold used for each composite is given in the figure caption. The supporting composite fields are collected in Supplementary Figs. \ref{sam_wmt}--\ref{sam_windice}.""")
rep(r"""(Fig.~\ref{sam_wmt}, black lines)""", r"""(Fig.~\ref{sam_wmt}a-e, black lines)""")
rep(r"""while the sea ice freshwater forcing plays a second role, especially in the Ross sea and Weddell sea sectors,  This thermal dominance""",
    r"""while the sea ice freshwater forcing plays a secondary role, especially in the Ross Sea and Weddell Sea sectors (Fig.~\ref{sam_wmt}a-c,f-h,k-m). This thermal dominance""")
rep(r"""sea ice freshwater flux (green lines) emerges as the primary contributor.""",
    r"""sea ice freshwater flux (green lines) emerges as the primary contributor (Fig.~\ref{sam_wmt}d,e,i,j,n,o,s,t).""")
rep(r"""contribute consistently to the circumpolar SAM-WMT signal across interglacial climate states, while the Ross Sea exhibits more variable and sometimes opposing responses. Moreover, the glacial WMT response to SAM is overall dominated by the Weddell Sea region, whereas the Ross Sea and Ad\'elie sectors play a minor role.""",
    r"""contribute consistently to the circumpolar SAM-WMT signal across interglacial climate states (Fig.~\ref{sam_wmt}k-m,p-r), while the Ross Sea exhibits more variable and sometimes opposing responses (Fig.~\ref{sam_wmt}f-h). Moreover, the glacial WMT response to SAM is overall dominated by the Weddell Sea region (Fig.~\ref{sam_wmt}n,o), whereas the Ross Sea and Ad\'elie sectors play a minor role (Fig.~\ref{sam_wmt}i,j,s,t).""")
# comments 21 and 22: regions measured from the 1.0-sigma composite (fig_sam_buoyancy/numbers_heatloss15.txt)
rep(r"""Heat flux composites reveals  distinct responses between interglacial and glacial periods (Fig. \ref{sam_heat}). During interglacials, positive SAM phases systematically enhance ocean heat loss by more than 15~W/m$^2$ over the ice-free parts of the Southern Ocean south of about 50°S, where open water permits direct atmosphere-ocean coupling  (Fig.~\ref{sam_heat}), particularly in the Weddell Sea, the Bellingshausen Sea, and the Indian Ocean sector. Decomposition into components shows that turbulent fluxes (sensible and latent heat fluxes) account for most of this anomaly, while radiative contributions (longwave and shortwave fluxes) remain negligible.  In contrast, glacial periods (LGM and MIS3) show only weak SAM-driven heat flux changes south of the ice edge, because the extensive sea ice cover limits direct atmosphere-ocean exchange.""",
    r"""Heat flux composites reveal distinct responses between interglacial and glacial periods (Fig.~\ref{sam_heat}). The figure shows them as the surface density flux they produce, on the same scale as the freshwater composites in Fig.~\ref{sam_fwf}. During interglacials, positive SAM phases increase the ocean heat loss by more than 15~W~m$^{-2}$ over parts of the open ocean between 50 and 70°S, north of the winter ice edge (dashed lines in Fig.~\ref{sam_heat}a-c). These areas lie mainly in the Indian Ocean sector between 60 and 140°E, north of the Cooperation, Davis and Mawson Seas, and in the Pacific sector between 180° and 120°W, north of the Ross and Amundsen Seas, with smaller patches north of the Weddell Sea. Decomposition into components shows that turbulent fluxes (sensible and latent heat) account for most of this anomaly (Fig.~\ref{sam_heat}k-m), while radiative contributions (longwave and shortwave) remain small (Fig.~\ref{sam_heat}f-h). In contrast, glacial periods (LGM and MIS3) show only weak SAM-driven heat flux changes south of the winter ice edge, which lies near 55°S on average, because the extensive sea ice cover limits direct atmosphere-ocean exchange (Fig.~\ref{sam_heat}d,e).""")
rep(r"""primarily through sea ice processes rather than net precipitation or river runoff (Fig.~\ref{sam_fwf}). Total freshwater flux anomalies exhibit general dipole patterns with positive SAM driving freshwater loss (ocean salinification) in coastal regions accompanied by freshwater gain offshore.""",
    r"""primarily through sea ice processes rather than net precipitation or river runoff (Fig.~\ref{sam_fwf}f-j against k-o). Total freshwater flux anomalies exhibit general dipole patterns with positive SAM driving freshwater loss (ocean salinification) in coastal regions accompanied by freshwater gain offshore (Fig.~\ref{sam_fwf}a-e).""")
rep(r"""and the coastal brine input integrated south of 65°S (Supplementary Fig. \ref{sam_pushpull}).""",
    r"""and the coastal brine input integrated south of 65°S (Supplementary Fig. \ref{sam_pushpull}a,b).""")
rep(r"""Across the five states the two responses scale together ($r = 0.77$), and the transformation response follows both.""",
    r"""Across the five states the two responses scale together ($r = 0.77$; Supplementary Fig. \ref{sam_pushpull}c), and the transformation response follows both (Supplementary Fig. \ref{sam_pushpull}d).""")
rep(r"""The spatial patterns show why (Supplementary Fig. \ref{sam_windice}). Strengthened westerlies under the positive phase drive a sea ice dipole,""",
    r"""The spatial patterns show why. Strengthened westerlies under the positive phase (Supplementary Fig. \ref{sam_windice}a,c,e,g,i) drive a sea ice dipole (Supplementary Fig. \ref{sam_windice}b,d,f,h,j),""")
rep(r"""and weakest at the Last Glacial Maximum, at 8.0~Sv, even though""",
    r"""and weakest at the Last Glacial Maximum, at 8.0~Sv (Supplementary Fig. \ref{sam_pushpull}a), even though""")
# Reviewer 2, Figs. 5 and 6: both composites now in one unit (numbers: fig_sam_buoyancy/numbers.txt, 1.0 sigma)
rep(r"""The heat and freshwater composites are shown in their native units in Fig. \ref{sam_heat} and Fig. \ref{sam_fwf}, which makes the two forcings difficult to compare directly. Expressed as equivalent surface density tendencies, the freshwater contribution is the larger of the two wherever sea ice is present, and the heat contribution is larger only in the ice-free sector, which is the same partition that governs the mean state.""",
    r"""Because Figs.~\ref{sam_heat} and \ref{sam_fwf} share one unit and one colour scale, the two forcings can be compared directly. Where sea ice is present, the freshwater term is the larger of the two over 90--94\% of the area, and on average it is 6 to 10 times larger than the heat term. North of the ice edge the heat term is the larger one over most of the area. This is the same split between ice-covered and open water that governs the mean state. These are local magnitudes and not area totals, because the freshwater term makes the surface denser near the coast and lighter further offshore.""")

# ---------------------------------------------------------------- Discussion
# the 69% -> 61% figure was overturned (log/PITFALLS.md); use the density-class result given in the letter
rep(r"""The thermal share of the dense-class transformation falls from 69\% to 61\% and the sea ice contribution at the transformation maximum roughly doubles, from 0.86 to 1.84~Sv (Fig. \ref{domain}).""",
    r"""The sea ice contribution at the transformation maximum roughly doubles, from 0.86 to 1.84~Sv (Fig. \ref{domain}). Resolved by density class, only one class of the PI transformation in this sector remains thermally dominated, $\sigma_2 = 36.75$--$37.00$~kg~m$^{-3}$, and every other class is haline dominated.""")
# model evaluation passage as rewritten in the letter (fixes the 1.43 K misreading)
rep(r"""\citet{sidorenko2019evaluation} evaluate the coupled configuration against observations and report that the abyssal overturning cell associated with Antarctic Bottom Water is reproduced with a maximum of about 10~Sv, comparable to the pre-industrial value obtained here. The same evaluation identifies the Southern Ocean as the region of the most pronounced warm bias of 1.43~K against the PHC climatology at the resolution used here, and notes that Antarctic sea ice is too thin, hardly reaching 0.25~m in the central Weddell Sea in September. Their overall assessment is that although biases in the representation of water mass properties and ventilation mechanisms are present, the model still produces a reasonable density distribution that maintains realistic transports. A warm and thin-ice Southern Ocean is precisely the state in which open-ocean convection is favoured over shelf processes, so these documented biases are consistent with the convection behaviour we diagnose above.""",
    r"""\citet{sidorenko2019evaluation} evaluate the same coupled configuration and report an abyssal overturning cell of about 10~Sv, comparable to the pre-industrial value obtained here. They also find that the Antarctic sea ice of this configuration is thin, hardly reaching 0.25~m in the central Weddell Sea in September, and that the Southern Ocean surface is too warm at this resolution, which they associate with the reduced sea ice area. Their overall assessment is that although biases in the representation of water mass properties and ventilation mechanisms are present, the model still produces a reasonable density distribution that maintains realistic transports. Our own comparisons with observation-based products point to the same bias. The winter mixed layer of the pre-industrial run has the same large-scale pattern as the ORAS5 reanalysis \cite{Zuo2019} but is too deep in the interior of the Weddell gyre (Supplementary Fig. \ref{mld_obs}). Compared with the OSI SAF satellite sea ice record for 1979--2008 \cite{OSISAF2022}, the simulated September sea ice extent is about half of the observed value. Most of the deficit is in the Weddell sector, where the simulated ice edge lies about 10° of latitude too far south, and almost all of the ice melts in summer. A warm Southern Ocean with too little winter sea ice is the state in which open-ocean convection is favoured over shelf processes, so these biases are consistent with the convection behaviour we diagnose above.""")
rep(r"""while the interglacials remain 53--88\% thermally driven (Fig. \ref{domain})""",
    r"""while the interglacials remain 53--70\% thermally driven (Fig. \ref{domain})""")
# Reviewer 2 on Lhardy et al. (2021): reconstructed extents (their Sect. 3.3), short attribution, duplicate sentence removed
rep(r"""In our simulations the glacial ice cover expands sufficiently to insulate the interior gyres, and the area of deep winter mixed layers contracts to roughly one fifth of its PI value. The Southern Hemisphere sea ice distribution""",
    r"""In our simulations the glacial ice cover expands sufficiently to insulate the interior gyres, and the area of deep winter mixed layers contracts to roughly one fifth of its PI value. The simulated LGM sea ice extent is 29.8 million km$^2$ in winter (July to September) and 11.1 million km$^2$ in summer (January to March). Both lie within the proxy-based estimates of about $33\pm5$ and $10\pm3$ million km$^2$ \cite{Lhardy2021}. We have not tested why the model does well here. The relatively fine ocean mesh near Antarctica may help, and the simulated glacial climate is cold enough to keep the surface water around Antarctica at its freezing point. The Southern Hemisphere sea ice distribution""")
rep(r"""so the haline-dominated glacial regime reported here is unlikely to be an artefact of excessive simulated ice. This behaviour is what produces the haline-dominated glacial regime reported here, and it means our glacial result depends on the simulated sea ice expansion being realistic. Proxy-based reconstructions indicate extensive glacial winter ice, which supports the direction of the simulated change, but the quantitative agreement remains uncertain.""",
    r"""so the haline-dominated glacial regime reported here is unlikely to be an artefact of excessive simulated ice. Our glacial result still depends on the simulated sea ice expansion being realistic. Proxy-based reconstructions support the direction and the rough size of the simulated change, but their uncertainty is large.""")
rep(r"""not by the position of the jet, and Gray et al.""", r"""not by the position of the westerly wind belt, and Gray et al.""")
rep(r"""A bias in jet position therefore does not""", r"""A bias in the position of the westerlies therefore does not""")

# ---------------------------------------------------------------- Methods
rep(r"""its intensity, however, depends on the simulated wind-driven ice divergence rather than on a resolved polynya circulation.""",
    r"""its intensity, however, depends on the simulated wind-driven ice divergence rather than on small-scale processes inside the polynyas, which the mesh does not resolve. Resolving them would need a considerably finer mesh near the continent, of the order of a few kilometres.""")
# boundary conditions, wording of the letter
rep(r"""and it is clearest to list them by the component that uses them. Specifically, the atmosphere component receives""",
    r"""and it is clearest to list them by the model component that uses them. Specifically, the atmospheric model component receives""")
rep(r"""The land surface receives the same topography""", r"""The land surface model component receives the same topography""")
rep(r"""a dedicated mesh must be generated for each glacial period; we use meshes""",
    r"""a dedicated mesh must be generated for each glacial period. We use meshes""")
rep(r"""in both the latitude and the strength of the Southern Hemisphere westerly jet,""",
    r"""in both the latitude and the strength of the Southern Hemisphere surface westerlies,""")
rep(r"""The total surface density tendency, $\mathcal{B}$ (kg m$^{-2}$ s$^{-1}$, which divided by 1 m gives the density change of the top metre of the water column), is decomposed""",
    r"""The total surface density tendency, $\mathcal{B}$ (kg m$^{-2}$ s$^{-1}$; divided by 1 m it gives the density change of the top metre of the water column, and in the figures it is shown per month of 30 days), is decomposed""")

rep(r"""%% TODO before submission: verify the four Sidorenko et al. (2019) numbers in the
%%   Discussion (about 10 Sv abyssal cell, 1.43 K SST RMSE against PHC, 0.25 m
%%   September Weddell ice thickness, and the "reasonable density distribution"
%%   assessment) against the published PDF.""",
    r"""%% TODO before submission: verify the Sidorenko et al. (2019) statements in the
%%   Discussion (about 10 Sv abyssal cell, 0.25 m September Weddell ice thickness,
%%   too-warm Southern Ocean surface in the historical run, and the "reasonable
%%   density distribution" assessment) against the published PDF. The 1.43 K value
%%   (global SST RMSE, not a Southern Ocean bias) was removed on 2026-11-25.""")

# ---------------------------------------------------------------- figure captions
rep(r"""\caption{Simulated JJA-mean anomalies in MH, LIG, LGM, and MIS3 in relative to PI for (a-d) sea surface temperature""",
    r"""\caption{Simulated June-September mean anomalies in MH, LIG, LGM, and MIS3 in relative to PI for (a-d) sea surface temperature""")
rep(r"""for (a-e)) Southern Ocean, (f-j) Ross Sea,  (k-o) Weddell Sea, Adélie  Land, and (p-t). Five columns represent five climate experiments (PI, MH, LIG, LGM, and MIS3).  Lines show the mean values; shading denotes ±1  standard deviation across years.    \label{wmt_jja}}""",
    r"""for (a-e) Southern Ocean, (f-j) Ross Sea, (k-o) Weddell Sea and (p-t) Adélie Land. Five columns represent five climate experiments (PI, MH, LIG, LGM, and MIS3).  Lines show the mean values; shading denotes ±1  standard deviation across years.    \label{wmt_jja}}""")
rep(r"""for (a-e)) Southern Ocean, (f-j) Ross Sea,  (k-o) Weddell Sea, Adélie  Land, and (p-t). Five columns represent five climate experiments (PI, MH, LIG, LGM, and MIS3).  Lines show the mean values; shading denotes ±1  standard deviation across years.  \label{s2}}""",
    r"""for (a-e) Southern Ocean, (f-j) Ross Sea, (k-o) Weddell Sea and (p-t) Adélie Land. Five columns represent five climate experiments (PI, MH, LIG, LGM, and MIS3).  Lines show the mean values; shading denotes ±1  standard deviation across years.  \label{s2}}""")
unit_old = r"""Units: kg m$^{-3}$ month$^{-1}$ per metre of water column, that is, the density change of a surface layer 1 m thick."""
unit_new = r"""Units: kg m$^{-2}$ month$^{-1}$, a surface density flux, which equals the density change of a surface layer 1 m thick over one month."""
rep(unit_old, unit_new, count=2)
rep(r"""shown as austral winter high-SAM minus low-SAM composites for the five climate states.""",
    r"""shown as austral winter high-SAM minus low-SAM composites ($\pm$1.2$\sigma$ threshold) for the five climate states.""")
rep(r"""for high-SAM minus low-SAM years , shown for (a-e) Southern Ocean, (f-j) Ross Sea,  (k-o) Weddell Sea, Adélie  Land, and (p-t).""",
    r"""for high-SAM minus low-SAM years ($\pm$1.2$\sigma$ threshold), shown for (a-e) Southern Ocean, (f-j) Ross Sea, (k-o) Weddell Sea and (p-t) Adélie Land.""")
rep(r"""\caption{Simulated JJA mean (a-e)  mixed layer depth [m] and (f-j) sea ice concentration""",
    r"""\caption{Simulated June-September mean (a-e)  mixed layer depth [m] and (f-j) sea ice concentration""")
rep(r"""\caption{Simulated JJA-mean anomalies in MH, LIG, LGM, and MIS3 in relative to PI for (a-d) sea surface density""",
    r"""\caption{Simulated June-September mean anomalies in MH, LIG, LGM, and MIS3 in relative to PI for (a-d) sea surface density""")

# A3/A4 (Reviewer 2): new files in density units; threshold as actually used
rep(r"""\includegraphics[width=\textwidth]{figures/sam_heatflux_composite_t63_3rows_5cols.pdf}
\caption{Austral winter (JJA) anomalies of (a-e) total heat flux (all components combined), (f-j) radiative component (longwave + shortwave), and (k-o) turbulent component (sensible + latent heat) for high-SAM minus low-SAM composites (1.0$\sigma$ threshold) across five paleoclimate states.  Five columns represent PI, MH, LIG, LGM, and MIS3, respectively. Units: W/m$^2$. Positive values indicates reduced ocean heat loss during high-SAM years.\label{sam_heat}}""",
    r"""\includegraphics[width=\textwidth]{figures/figS_sam_heat_density.pdf}
\caption{Austral winter (JJA) high-SAM minus low-SAM composites ($\pm$1.0$\sigma$ threshold) of the surface density flux produced by (a-e) the total heat flux, (f-j) its radiative component (shortwave plus longwave) and (k-o) its turbulent component (sensible plus latent heat). Five columns represent PI, MH, LIG, LGM, and MIS3, respectively. The flux is converted with the thermal expansion coefficient at the winter surface temperature and salinity of each state, as in Fig. \ref{density_heat}, and shares one colour scale with Fig. \ref{sam_fwf}. Positive values mean that the surface water becomes denser. The black line is the 15\% winter sea ice edge of each state. The dashed purple line in (a-e) marks where high-SAM winters lose 15~W~m$^{-2}$ more heat than low-SAM winters. Units: kg m$^{-2}$ month$^{-1}$.\label{sam_heat}}""")
rep(r"""\includegraphics[width=\textwidth]{figures/sam_fwflux_composite_3rows_5cols.pdf}
\caption{Austral winter (JJA) anomalies of (a-e) total freshwater flux, (f-j) sea ice component (brine rejection and melt), and (k-o) other sources (net precipitation + runoff) for high-SAM minus low-SAM composites (1.0$\sigma$ threshold) across five paleoclimate states. Five columns represent PI, MH, LIG, LGM, and MIS3, respectively.  Units: mm/day. Positive indicates net freshwater gain to ocean, negative indicates salinification. \label{sam_fwf}}""",
    r"""\includegraphics[width=\textwidth]{figures/figS_sam_fw_density.pdf}
\caption{Austral winter (JJA) high-SAM minus low-SAM composites ($\pm$1.0$\sigma$ threshold) of the surface density flux produced by (a-e) the total freshwater flux, (f-j) its sea ice component (brine rejection and melt) and (k-o) other sources (net precipitation plus runoff). Five columns represent PI, MH, LIG, LGM, and MIS3, respectively. The flux is converted with the haline contraction coefficient and surface salinity of each state, as in Fig. \ref{density_fwf}, and shares one colour scale with Fig. \ref{sam_heat}. Positive values mean that the surface water becomes denser, that is, the ocean loses freshwater. The black line is the 15\% winter sea ice edge of each state. Units: kg m$^{-2}$ month$^{-1}$.\label{sam_fwf}}""")

# s1 is the glacial MLD panel the letter shows (figR7, JJA, 0-300 m)
rep(r"""\centering\includegraphics[width=0.8\textwidth]{figures/mld_lgm_mis_75S_jja.pdf}
        \caption{Simulated JJA-mean mixed layer depth across Southern Ocean (south of 65°S) for (a) LGM and (b) MIS3. Units: m. \label{s1}}""",
    r"""\centering\includegraphics[width=0.8\textwidth]{figures/figR7_mld_glacial_zoom.pdf}
        \caption{Simulated JJA-mean mixed layer depth south of 60°S for (a) LGM and (b) MIS3, on a 0--300 m colour scale with the 200 m contour in orange. Units: m. \label{s1}}""")

# ---------------------------------------------------------------- figure order = order of first citation
def grab_fig(label):
    """Return the full figure environment (from \\begin{figure} to \\end{figure}) holding \\label{label}."""
    i = T.index(r'\label{%s}' % label)
    s = T.rindex(r'\begin{figure}', 0, i)
    e = T.index(r'\end{figure}', i) + len(r'\end{figure}')
    return s, e

def take(label):
    global T
    s, e = grab_fig(label)
    block = T[s:e]
    T = T[:s] + T[e:]
    return block

# main text: MOC is cited before the 4000 m age map
age_blk = take('age')
s, e = grab_fig('moc')
T = T[:e] + '\n\n' + age_blk + T[e:]
# the common-axis figure is supplementary (Reviewer 1, comment 15)
common_blk = take('s_common_axis')

mld_obs_blk = r"""\begin{figure}
\centering
\includegraphics[width=\textwidth]{figures/figR17_mld_pi_vs_obs.pdf}
\caption{Winter (JJA) mixed layer depth south of 60°S in (a) the pre-industrial simulation and (b) the ORAS5 ocean reanalysis \cite{Zuo2019}, averaged over 1990--2009. ORAS5 defines the mixed layer by a density threshold of 0.01~kg~m$^{-3}$ and the model by a bulk Richardson number criterion, so the point-by-point difference is not purely model error. Units: m. \label{mld_obs}}
\end{figure}"""

app_order = ['wind', 'mld', 'reso', 's1', 's_common_axis', 'sam_wmt', 'sam_heat', 'sam_fwf',
             'sam_pushpull', 'sam_windice', 's2', 'mld_obs']
blocks = {'s_common_axis': common_blk, 'mld_obs': mld_obs_blk}
for lab in app_order:
    if lab not in blocks:
        blocks[lab] = take(lab)
a0 = T.index(r'\section{Supplementary figures}\label{secA1}') + len(r'\section{Supplementary figures}\label{secA1}')
a1 = T.index(r'\end{appendices}')
leftover = re.sub(r'\\clearpage|\s', '', T[a0:a1])
assert leftover == '', f'unexpected material left in the appendix: {leftover[:200]}'
T = T[:a0] + '\n\n' + '\n\n\\clearpage\n\n'.join(blocks[l] for l in app_order) + '\n\n' + T[a1:]


# comment 24 and boundary conditions: wording of the letter (quote check)
rep(r"""Strengthened and poleward-shifted westerlies increase the offshore component of the wind stress at the coast""",
    r"""Strengthened and poleward-shifted westerlies associated with positive SAM increase the offshore component of the wind stress at the coast""")
rep(r"""The ice exported offshore carries the compensating freshwater north, where it melts and freshens the surface. The result is the observed dipole, with salinification at the coast and freshening offshore, and it is the same mechanism identified""",
    r"""The ice exported offshore carries the compensating freshwater north, where it tends to melt and freshen the surface. This results in the observed dipole pattern, with salinification at the coast and freshening offshore. This is the same mechanism identified""")
rep(r"""The ocean component receives bathymetry and a modified coastline.""",
    r"""The ocean model component receives bathymetry and a modified coastline.""")


# Methods sign convention made consistent with the figures (B > 0 = densification, fluxes + into the ocean)
rep(r"""= -\frac{\alpha}{C_p} Q_{net} + \beta S F_{fw},""",
    r"""= -\frac{\alpha}{C_p} Q_{net} - \beta S F_{fw},""")
rep(r"""taken as mass fluxes in kg m$^{-2}$ s$^{-1}$,""",
    r"""taken as mass fluxes in kg m$^{-2}$ s$^{-1}$ and positive into the ocean,""")
rep(r"""Note that under this convention, a heat loss ($Q_{\text{net}} < 0$) or freshwater loss ($F_{fw} < 0$) results in a negative buoyancy flux ($\mathcal{B} < 0$) and thus densification.""",
    r"""Under this convention, a heat loss ($Q_{\text{net}} < 0$) or a freshwater loss ($F_{fw} < 0$) gives a positive $\mathcal{B}$, that is, densification.""")
rep(r"""the part of the haline term $\beta S F_{fw}$ that is carried""",
    r"""the part of the haline term $-\beta S F_{fw}$ that is carried""")

open(tex_path, 'w', encoding='utf8').write(T)

# ---------------------------------------------------------------- references (DOIs checked on Crossref / DataCite, abstracts read 2026-11-25)
NEW_BIB = r"""

@article{Gong1999,
  author  = {Gong, Daoyi and Wang, Shaowu},
  title   = {Definition of {Antarctic Oscillation} index},
  journal = {Geophysical Research Letters},
  year    = {1999},
  volume  = {26},
  number  = {4},
  pages   = {459--462},
  doi     = {10.1029/1999GL900003}
}

@article{HallVisbeck2002,
  author  = {Hall, Alex and Visbeck, Martin},
  title   = {Synchronous Variability in the {Southern Hemisphere} Atmosphere, Sea Ice, and Ocean Resulting from the Annular Mode},
  journal = {Journal of Climate},
  year    = {2002},
  volume  = {15},
  number  = {21},
  pages   = {3043--3057},
  doi     = {10.1175/1520-0442(2002)015<3043:SVITSH>2.0.CO;2}
}

@article{SenGupta2006,
  author  = {Sen Gupta, Alexander and England, Matthew H.},
  title   = {Coupled Ocean--Atmosphere--Ice Response to Variations in the {Southern Annular Mode}},
  journal = {Journal of Climate},
  year    = {2006},
  volume  = {19},
  number  = {18},
  pages   = {4457--4486},
  doi     = {10.1175/JCLI3843.1}
}

@article{Zuo2019,
  author  = {Zuo, Hao and Balmaseda, Magdalena Alonso and Tietsche, Steffen and Mogensen, Kristian and Mayer, Michael},
  title   = {The {ECMWF} operational ensemble reanalysis--analysis system for ocean and sea ice: a description of the system and assessment},
  journal = {Ocean Science},
  year    = {2019},
  volume  = {15},
  number  = {3},
  pages   = {779--808},
  doi     = {10.5194/os-15-779-2019}
}

@misc{OSISAF2022,
  author       = {{EUMETSAT Ocean and Sea Ice Satellite Application Facility}},
  title        = {Global Sea Ice Concentration Climate Data Record v3.0 -- Multimission ({OSI-450-a})},
  howpublished = {EUMETSAT OSI SAF},
  year         = {2022},
  doi          = {10.15770/EUM_SAF_OSI_0013}
}
"""
for key in ['Gong1999', 'HallVisbeck2002', 'SenGupta2006', 'Zuo2019', 'OSISAF2022']:
    assert ('{' + key + ',') not in B, f'{key} already in ref.bib'
open(bib_path, 'w', encoding='utf8').write(B.rstrip('\n') + NEW_BIB)
print('revised.tex and ref.bib updated')
