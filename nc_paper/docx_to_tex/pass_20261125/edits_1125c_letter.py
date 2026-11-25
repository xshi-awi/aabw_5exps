#!/usr/bin/env python3
"""2026-11-25 afternoon pass (FV): response_new.tex edits that go with edits_1125c_revised.py.
Quotes follow nc_paper/build/revised.tex after that script; figure numbers follow its compile
(Figs. 1-9, Supplementary Figs. A1-A12). Anchors must match exactly once.
Usage: python3 edits_1125c_letter.py <response_new.tex>
"""
import sys

path = sys.argv[1]
T = open(path, encoding='utf8').read()


def rep(old, new, count=1):
    global T
    n = T.count(old)
    assert n == count, f'anchor found {n} times, expected {count}: {old[:90]!r}'
    T = T.replace(old, new)


def rep_block(start, end, new):
    """Replace the text from `start` up to and including `end` (both unique)."""
    global T
    assert T.count(start) == 1, f'start anchor: {T.count(start)} {start[:80]!r}'
    i = T.index(start)
    j = T.index(end, i)
    assert T.count(end) == 1 or T.index(end, i) == j
    T = T[:i] + new + T[j + len(end):]


def change(text):
    return '\\begin{change}\n``' + text + "''\n\\end{change}"


# ------------------------------------------------------------------ quotes of the revised manuscript
ABSTRACT_SAM = ("The balance between these two pathways is set by sea ice extent, which is modulated by internal atmospheric variability. "
                "Further analysis suggests that the Southern Annular Mode modulates transformation by 3--8 Sv between its positive and negative phases in every climate state, "
                "working through heat loss when the ocean is open and through brine rejection when it is ice covered.")
ABSTRACT_MOC = ("Weaker glacial transformation coincides with a collapse of the simulated abyssal overturning cell from about 11 to 2 Sv and with older abyssal water. "
                "Ideal ages at 4000 m remain below 500 years during interglacials but exceed 750 years basin-wide during glacials, reaching 1500 years in the Pacific.")
INTRO_SAM = ("Beyond the mean state, dense water formation also responds to internal atmospheric variability. "
             "The Southern Annular Mode (SAM) is the leading mode of extratropical Southern Hemisphere circulation variability, describing a meridional redistribution of atmospheric mass between mid-latitudes and Antarctica (Gong and Wang, 1999; Thompson and Wallace, 2000; Marshall, 2003). "
             "Its positive phase strengthens the circumpolar westerlies and displaces them poleward, which changes Ekman divergence, air-sea heat exchange, and the northward export of sea ice (Hall and Visbeck, 2002; Sen Gupta and England, 2006). "
             "In the present-day ocean these adjustments modulate mixed-layer depth and surface buoyancy forcing (Sallée et al., 2010; Sen Gupta and England, 2006). "
             "Surface buoyancy forcing in turn sets the rate at which surface waters are transformed into denser classes (Cerovečki et al., 2013; Abernathey et al., 2016). "
             "The SAM has also been linked to observed changes in bottom water properties (Zhou et al., 2023; Silvano et al., 2020). "
             "Whether the SAM remains an effective modulator of dense water formation once the mean state changes, in particular once an expanded ice cover separates the ocean from the atmosphere, is not known.")
SEAICE_FIG1 = ("The sea ice anomalies (Fig. 1i-l) follow the temperature changes. In MH the changes are small. "
               "In LIG the winter sea ice decreases strongly in the Ross sector, where the winter ice edge retreats south by about 5° of latitude beneath the warm and salty surface anomaly. "
               "In LGM and MIS3 the winter sea ice area expands substantially and the ice edge moves north by about 10--15° of latitude, most of all in the Weddell sector.")
WINDS = ("The MH shows moderate anomalies in zonal wind, generally negative south of about 50°S and positive to the north. "
         "The LIG shows a similar pattern but with a strong intensification in the Pacific sector (Fig. 1n). "
         "Both LGM and MIS3 exhibit strong positive anomalies in zonal wind speed and wind stress (> 0.04 N/m2) across the circumpolar belt, although the LGM also shows some negative anomalies in the Atlantic-Indian sector (Fig. 1o,p; Fig. A1g,h). "
         "This intensified wind stress enhances Ekman transport and facilitates the northward expansion of sea ice, fundamentally altering the air-sea interaction interface.")
ROSS = ("The Ross sector instead stands out in the last interglacial. There the surface ocean between 60 and 75°S is warmer than in PI and sits well above the freezing point, and winter ice concentration falls. "
        "In that state the wind can export ice efficiently but the water is too far from freezing for the exported ice to be replaced, so the sector loses ice cover and gains exposure to the atmosphere.")
HEAT_REGIONS = ("Heat flux composites reveal distinct responses between interglacial and glacial periods (Fig. A7). "
                "The figure shows them as the surface density flux they produce, on the same scale as the freshwater composites in Fig. A8. "
                "During interglacials, positive SAM phases increase the ocean heat loss by more than 15 W m\\textsuperscript{-2} over parts of the open ocean between 50 and 70°S, north of the winter ice edge (dashed lines in Fig. A7a-c). "
                "These areas lie mainly in the Indian Ocean sector between 60 and 140°E, north of the Cooperation, Davis and Mawson Seas, and in the Pacific sector between 180° and 120°W, north of the Ross and Amundsen Seas, with smaller patches north of the Weddell Sea. "
                "Decomposition into components shows that turbulent fluxes (sensible and latent heat) account for most of this anomaly (Fig. A7k-m), while radiative contributions (longwave and shortwave) remain small (Fig. A7f-h). "
                "In contrast, glacial periods (LGM and MIS3) show only weak SAM-driven heat flux changes south of the winter ice edge, which lies near 55°S on average, because the extensive sea ice cover limits direct atmosphere-ocean exchange (Fig. A7d,e).")
SAM_COMMON = ("Because Figs. A7 and A8 share one unit and one colour scale, the two forcings can be compared directly. "
              "Where sea ice is present, the freshwater term is the larger of the two over 90--94\\% of the area, and on average it is 6 to 10 times larger than the heat term. "
              "North of the ice edge the heat term is the larger one over most of the area. This is the same split between ice-covered and open water that governs the mean state. "
              "These are local magnitudes and not area totals, because the freshwater term makes the surface denser near the coast and lighter further offshore.")
MODEL_EVAL = ("Independent evaluations of this model provide context for these limitations, and we summarise the relevant points here rather than leaving them to be sought in the model description literature. "
              "Sidorenko et al. (2019) evaluate the same coupled configuration and report an abyssal overturning cell of about 10 Sv, comparable to the pre-industrial value obtained here. "
              "They also find that the Antarctic sea ice of this configuration is thin, hardly reaching 0.25 m in the central Weddell Sea in September, and that the Southern Ocean surface is too warm at this resolution, which they associate with the reduced sea ice area. "
              "Their overall assessment is that although biases in the representation of water mass properties and ventilation mechanisms are present, the model still produces a reasonable density distribution that maintains realistic transports. "
              "Our own comparisons with observation-based products point to the same bias. "
              "The winter mixed layer of the pre-industrial run has the same large-scale pattern as the ORAS5 reanalysis (Zuo et al., 2019) but is too deep in the interior of the Weddell gyre (Supplementary Fig. A12). "
              "Compared with the OSI SAF satellite sea ice record for 1979--2008 (EUMETSAT OSI SAF, 2022), the simulated September sea ice extent is about half of the observed value. "
              "Most of the deficit is in the Weddell sector, where the simulated ice edge lies about 10° of latitude too far south, and almost all of the ice melts in summer. "
              "A warm Southern Ocean with too little winter sea ice is the state in which open-ocean convection is favoured over shelf processes, so these biases are consistent with the convection behaviour we diagnose above.")
LHARDY = ("The ability of the model to maintain an extensive glacial winter ice cover, and thereby to suppress open-ocean convection, is not a general feature of paleoclimate simulations. "
          "Many PMIP models underestimate glacial Southern Ocean sea ice and its seasonality, and colder glacial forcing on its own tends to intensify open-ocean convection rather than suppress it, which degrades the properties of the simulated bottom water; in one systematic study the only configuration that avoided an unrealistically deep northern-sourced cell was the one in which brine sinking along the Antarctic margin was parameterised explicitly (Lhardy et al., 2021). "
          "In our simulations the glacial ice cover expands sufficiently to insulate the interior gyres, and the area of deep winter mixed layers contracts to roughly one fifth of its PI value. "
          "The simulated LGM sea ice extent is 29.8 million km\\textsuperscript{2} in winter (July to September) and 11.1 million km\\textsuperscript{2} in summer (January to March). "
          "Both lie within the proxy-based estimates of about 33 ± 5 and 10 ± 3 million km\\textsuperscript{2} (Lhardy et al., 2021). "
          "We have not tested why the model does well here. The relatively fine ocean mesh near Antarctica may help, and the simulated glacial climate is cold enough to keep the surface water around Antarctica at its freezing point. "
          "The Southern Hemisphere sea ice distribution of these same five simulations has been published previously (Shi et al., 2025), and the pronounced glacial expansion described here is evident there as well. "
          "The simulated glacial sea ice of this model has also been assessed against diatom-based reconstructions in a multi-model comparison, which found that models including AWI-ESM tend to underestimate the winter sea ice extent inferred from the proxies while capturing the summer distribution more closely (Green et al., 2022). "
          "An underestimate of winter ice would if anything weaken the insulation effect we describe, so the haline-dominated glacial regime reported here is unlikely to be an artefact of excessive simulated ice. "
          "Our glacial result still depends on the simulated sea ice expansion being realistic. "
          "Proxy-based reconstructions support the direction and the rough size of the simulated change, but their uncertainty is large.")
CHEN = ("Our conclusions about the glacial state are consistent with recent work using the same ocean model in a forced configuration. "
        "Chen et al. (2025) attribute the large glacial volume of Atlantic bottom water to a substantial increase in sea ice export toward lower latitudes, which supplies dense shelf water year round, and to weaker mixing between northern- and southern-sourced water, which allows that water to retain the properties of its origin. "
        "The agreement is worth stating explicitly, and the two studies are complementary rather than overlapping. "
        "Chen et al. infer the role of sea ice from water mass volumes, the overturning streamfunction, and mixed-layer and sea ice fields, in ocean-only simulations of two states with a prescribed atmosphere. "
        "The transformation framework applied here measures the surface buoyancy forcing directly and attributes it to individual flux components in sverdrups, so the contribution of sea ice is quantified rather than inferred from covarying fields. "
        "Running the analysis across five coupled climate states, in which sea ice and the atmospheric fluxes evolve together, is also what makes the change of regime visible, since a shift between two forcing regimes cannot be identified from a single glacial state. "
        "The present analysis therefore tests and quantifies the mechanism that Chen et al. proposed.")
ARTEFACT = ("It is reasonable to ask whether the regime shift is then simply an artefact of a biased interglacial end member, with the glacial state being the more realistic of the two. "
            "Two results argue against that reading. First, the shift survives the domain test. "
            "Recomputed over the sea ice sector alone, the glacial states remain 15\\% and 17\\% thermally driven while the interglacials remain 53--70\\% thermally driven (Fig. 8), so the contrast is not created by including open water in the interglacial integrals. "
            "Second, the mechanism is the ice cover itself rather than the convection style. "
            "Insulation of the surface from the atmosphere and concentration of brine rejection follow from the areal expansion of sea ice, and both would operate in a model that formed its dense water on the shelf. "
            "What such a model would change is the density and the geographic origin of the resulting water, not the direction of the shift in the surface buoyancy budget.")
DRIFT = ("Residual drift over the analysis period is small relative to the paleoclimate signals of interest. "
         "The trend in global mean 2 m temperature across the final century is +0.02 K century\\textsuperscript{-1} for PI, -0.03 for MH, -0.06 for LIG, +0.04 for LGM and +0.10 for MIS3. "
         "The LIG and MIS3 values exceed the ±0.05 K century\\textsuperscript{-1} criterion applied in earlier work with this model (Shi et al., 2022), and we note this explicitly rather than leaving it implicit. "
         "The glacial-interglacial differences examined here are large by comparison: the same five simulations give a JJAS global mean surface air temperature of 15.5, 15.6 and 16.9 °C for PI, MH and LIG against 11.1 and 13.0 °C for LGM and MIS3 (Shi et al., 2025), a contrast of several kelvin. "
         "A century-scale drift of 0.1 K is therefore two orders of magnitude smaller than the signal and cannot account for it. "
         "The mid-Holocene simulation is also indistinguishable from the pre-industrial control in both the latitude and the strength of the Southern Hemisphere surface westerlies, which argues against residual drift as the origin of the wind changes reported in the other paleoclimate states.")

# ------------------------------------------------------------------ Reviewer 1, general comments
rep(r"""We diagnosed winter mixed layer depth (Figure~\ref{fR:mld}), reflecting""",
    r"""We diagnosed winter mixed layer depth (Figure~\ref{fR:mld}, now Supplementary Fig. A12), reflecting""")
rep_block(r"""\begin{change}
``Independent evaluations of this model provide context""", r"""a place in the Supplement.}""",
          change(MODEL_EVAL))
rep(r"""We have added the quantitative comparison to the text and to the new Fig. 8, reproduced as Figure~\ref{fR:wmt} above.""",
    r"""We have added the quantitative comparison to the Discussion and the transformation over the sea ice zone to the new Fig. 8. The side-by-side comparison with the published figure of Pellichero et al. (2018) is Figure~\ref{fR:wmt} above. We show it only in this letter, because one of its panels reproduces their published figure.""")
rep(r"""Supplementary Fig. A6. Model and observations""", r"""Supplementary Fig. A2. Model and observations""")

# ------------------------------------------------------------------ Reviewer 1, specific comments
rep(change("The shift between the thermal and haline pathways is set by sea ice cover , which is strongly influenced by the Southern Annular Mode. It modulates transformation by 3-8 Sv between its positive and negative phases, working through heat loss when the ocean is open and through brine rejection when it is ice covered."),
    change(ABSTRACT_SAM))
rep(change("The shift between the thermal and haline pathways is set by sea ice cover, which is strongly influenced by the Southern Annular Mode. It modulates transformation by 3-8 Sv between its positive and negative phases, working through heat loss when the ocean is open and through brine rejection when it is ice covered."),
    change(ABSTRACT_SAM))

rep(r"""% NOTE: abstract now says "from about 11 to 2 Sv" (D6, 2026-11-08)

\response The abstract now bridges from surface forcing to ventilation via the overturning circulation, stating that the weaker glacial transformation of surface water into dense water coincides with a weakening of the simulated abyssal cell from about 10 to 2 Sv, before the ages are given. This also incorporates the new overturning analysis requested by Reviewer 3.

% NOTE: abstract now says "from about 11 to 2 Sv" (D6, 2026-11-08)

In the revised manuscript this now reads:""",
    r"""\response The abstract now bridges from surface forcing to ventilation via the overturning circulation, stating that the weaker glacial transformation of surface water into dense water coincides with a weakening of the simulated abyssal cell from about 11 to 2 Sv, before the ages are given. This also incorporates the new overturning analysis requested by Reviewer 3.

The abstract now reads:""")
rep_block(r"""\begin{change}
``The weaker WMT in the glacial simulations""", r"""or quote the passage that is there.}""", change(ABSTRACT_MOC))

rep(r"""please consider providing a slightly more detailed introduction.}%那你就把它加进revision""",
    r"""please consider providing a slightly more detailed introduction.}%那你就把它加进revision
%claude回复：这段已经在修订稿引言里（原来接在古气候那段的末尾，不好找），现在单独成段，并补上 Gong & Wang 1999、Hall & Visbeck 2002、Sen Gupta & England 2006 等引用（都在 Crossref 核过 DOI、读过摘要）。下面的引文与修订稿一致。""")
rep_block(r"""\begin{change}
``The Southern Annular Mode is the leading mode""", r"""is not known.''
\end{change}""", change(INTRO_SAM))

rep_block(r"""\begin{change}
``As a result, the MLD is strongly reduced""", r"""%我在回复信里面做的修改，你需要在revision里面做相应修改。""",
          change("As a result, the mixed layer depth is strongly reduced and becomes confined to narrow coastal bands and polynyas (Fig. A3).")
          + "\n\n%我在回复信里面做的修改，你需要在revision里面做相应修改。\n%claude回复：已把修订稿原句改成这句（MLD 在稿件里没定义，写全称）。后面的面积数字留在稿件里，第 10 条的回复也用它们。")

rep(r"""\response Thank you. Fig. 1i-l shows the winter (JJA) sea ice concentration anomalies relative to PI, and the revised Results now describe them.""",
    r"""\response Thank you. Fig. 1i-l shows the winter (June to September) sea ice concentration anomalies relative to PI, and the revised Results now describe them. The numbers below are June to August means.""")
rep_block(r"""The text proposed for the Results reads:""", r"""Please write 10--15° with an en dash.}""",
          "The text added to the Results reads:\n\n" + change(SEAICE_FIG1)
          + "\n\n%claude回复（11-25）：MH、LIG 西威德尔海冰减少那句删了：它只在被覆盖的 PI 海冰文件下成立，用正确 PI（2300--2399）是增加。留下的三句两份文件下都成立，已写进修订稿 2.1 节。")

rep(change("The MH shows moderate anomalies in zonal winds, generally negative over sea ice zone and positive to the north. The LIG exhibits similar behaviour, but displays strong intensification in the Pacific sector (Fig. 1n). Both LGM and MIS3 exhibit strong positive anomalies in zonal wind and wind stress (> 0.04 N/m2) across the circumpolar belt, although some negative anomalies in the coastal Atlantic–Indian sector (Fig. 1o,p; Fig. A3g,h). This intensified wind stress enhances Ekman transport and facilitates the northward expansion of sea ice, fundamentally altering the air-sea interaction interface."),
    change(WINDS) + "\n\n%claude回复（11-25）：已写进修订稿。按 6--9 月 10 m 风核过：MH 从 50°S 往南为负、以北为正；大西洋--印度洋扇区的负异常只在 LGM 有（40--50°S 和沿岸），MIS3 没有，所以写成 the LGM also shows。")

rep(r"""\response Line 146-148 is for the glacial experiments which present increased wind stress and sea ice in the Ross sector. So we think here the reviewer is referring to the LIG experiment with increased wind stress but less sea ice in that region. We checked this and the mechanism the reviewer proposes is real. In LIG the surface of the Ross sector is about 1 K warmer than PI, and the increased salinity there reduces the freezing point.  There the wind can export ice efficiently but the water is too far from freezing for the exported ice to be replaced, which is precisely the reviewer's proposed mechanism. We have added this mechanism in the revised manuscript.""",
    r"""\response Lines 146--148 describe the glacial experiments, which show stronger wind stress and more sea ice in the Ross sector. We therefore think the reviewer refers to the LIG experiment, which has stronger winds but less sea ice there. We checked this, and the mechanism the reviewer proposes is at work. In LIG the surface of the Ross sector is about 1 K warmer than in PI and sits well above its freezing point. The wind can export ice efficiently, but the water is too far from freezing for the exported ice to be replaced. We have added this mechanism to the revised manuscript.""")
rep_block(r"""\begin{change}
``The Ross sector instead stands out""", r"""请更新）} """,
          change(ROSS) + "\n\n%claude回复（11-25）：按你的意见改了。回复按建议改写；引文和修订稿第 114 行只谈 Ross、只谈方向，删了 0.97 K、2.72 K、0.19 和 order of magnitude 那句。同段冰期那句也去掉了 0.62/0.70/0.53 这几个数，只说三个扇区都增加。")

rep(r"""through net precipitation and river runoff. .''""", r"""through net precipitation and river runoff.''""")

rep(r"""All rows of the figure share one colour scale for PI and one for the anomalies, in the same unit, so we now draw each colour bar once, under the bottom row, and label the rows on the left.""",
    r"""All rows of the figure share one colour scale for PI and one for the anomalies, in the same unit, so we now draw each colour bar once, under the bottom row, and label the rows on the left. The colour bars give the unit as a surface density flux, kg m\textsuperscript{-2} month\textsuperscript{-1}, the same quantity and unit as the Southern Annular Mode composites in Supplementary Figs. A7 and A8, which Reviewer 2 asked to see on one scale.""")
rep(r"""%你好claude，我记得有一个审稿人说要统一这类图的单位（好像是原始稿件的fig 5，6），按照他的来。""",
    r"""%你好claude，我记得有一个审稿人说要统一这类图的单位（好像是原始稿件的fig 5，6），按照他的来。
%claude回复（11-25）：是审稿人 2。Fig. 3、4 和新的 A7、A8（原 Fig. 5、6）现在都用 kg m-2 month-1（表面密度通量，数值与原来的 per metre 单位相同），色标下只写这个；正文第 2.3 节的数也换成这个单位。""")

rep(change("Interglacial anomalies (MH, LIG; Fig. 3b,c,g,h,l,m) show small to modest changes circumpolar north of the ice covered region relative to PI. MH exhibits no obvious thermal density tendency anomalies in the ice covered region, while LIG exhibits pronounced increases in thermal-driven surface density tendency localized to the Ross Sea sector."),
    change("Interglacial anomalies (MH, LIG; Fig. 3b,c,g,h,l,m) show small to modest changes relative to PI north of the ice-covered region all around Antarctica. MH exhibits no obvious thermal density tendency anomalies in the ice-covered region, while LIG exhibits pronounced increases in thermal-driven surface density tendency localized to the Ross Sea sector."))
rep(r"""coefficient (that is the $-\left(\alpha/C_p\right)Q_{net}$ in equation 1 in Method), and to the sea-ice-driven density tendency, meaning the part of the haline term (that is $\beta S F_{fw}$ in equation 1 in Method) that is carried""",
    r"""coefficient, that is the $-\left(\alpha/C_p\right)Q_{net}$ term, and to the sea-ice-driven density tendency, meaning the part of the haline term $\beta S F_{fw}$ that is carried""")

rep(r"""\response We now cite Hall and Visbeck (2002), Cerovečki et al. (2013), and Abernathey et al. (2016) in support of the statement.""",
    r"""\response We now support this statement with references. Following Reviewer 3, the statement has also moved to the Introduction, where it is part of the paragraph on the Southern Annular Mode quoted under comment 5 above. It now cites Hall and Visbeck (2002), Sen Gupta and England (2006) and Sallée et al. (2010) for the effect of the mode on the westerlies, sea ice export and surface buoyancy forcing, and Cerovečki et al. (2013) and Abernathey et al. (2016) for the link between surface buoyancy forcing and transformation. The definition of the mode now cites Gong and Wang (1999), Thompson and Wallace (2000) and Marshall (2003).

Gong, D., Wang, S., Definition of Antarctic Oscillation index, \textit{Geophys. Res. Lett.} 26, 459–462 (1999).

Thompson, D. W. J., Wallace, J. M., Annular modes in the extratropical circulation. Part I: Month-to-month variability, \textit{J. Clim.} 13, 1000–1016 (2000).

Marshall, G. J., Trends in the Southern Annular Mode from observations and reanalyses, \textit{J. Clim.} 16, 4134–4143 (2003).

Sen Gupta, A., England, M. H., Coupled ocean–atmosphere–ice response to variations in the Southern Annular Mode, \textit{J. Clim.} 19, 4457–4486 (2006).

Sallée, J. B., Speer, K. G., Rintoul, S. R., Zonally asymmetric response of the Southern Ocean mixed-layer depth to the Southern Annular Mode, \textit{Nat. Geosci.} 3, 273–279 (2010).""")
rep(r"""\claudenote{Hall and Visbeck (2002) is not yet cited in revised.tex and not in ref.bib. Cerovečki et al. (2013) and Abernathey et al. (2016) are cited there (line 142).} %按照我前面说的去引用""",
    r"""%按照我前面说的去引用""")
rep(r"""%确认

\comment{R1}{21.""", r"""%确认
%claude回复（11-25）：已做。SAM 的定义与对浮力强迫的作用并进引言那段（审稿人 3 也要求挪到引言），引用见上面回复；第 2.4 节开头那两句删了，避免重复。只引了 Crossref 核过 DOI 并读过摘要的文章。Lovenduski & Gruber 2005 讲的是 SST 和叶绿素，没引。

\comment{R1}{21.""")

rep(r"""\response \docxcomment{6}{Microsoft Office User, 2026-11-08}{Done, and the same treatment has been applied throughout the Result...}{Rewrite! See revision on the printed manuscript}""",
    r"""%claude回复（11-25）：已做，原 docx 批注 6（Rewrite! See revision on the printed manuscript）已按你的说明处理，批注删去。
\response Done. Section 2.4 now cites the subpanels of each figure it describes, for example Supplementary Fig. A6a-e for the circumpolar transformation response, Fig. A6f-h for the Ross Sea, Fig. A7a-c and A7k-m for the interglacial heat flux response, and Fig. A8f-j for the sea ice term.""")
rep(r"""\response \docxcomment{7}{Microsoft Office User, 2026-11-08}{Both instances have been rewritten. They now read "over the ice-fre...}{Rewrite!} %按照我前面说的改
%claude回复：同上，和第 21 条一起改。""",
    r"""\response Both instances have been rewritten, and the regions are now named. Where the interglacial heat loss increases by more than 15 W m\textsuperscript{-2} was measured from the composite itself, and the latitude of the glacial ice edge from the sea ice field. The passage now reads:

""" + change(HEAT_REGIONS) + r"""

%按照我前面说的改
%claude回复（11-25）：已改，原 docx 批注 7（Rewrite!）删去。15 W m-2 这次从 1.0σ 合成数据量出：间冰期 97%（PI）、90%（MH）、96%（LIG）在冰缘以北的开阔洋，主要在 60--140°E 和 180--120°W，Weddell 只占约一成；冰期冰缘纬圈平均 55--56°S，冰下超过 15 W m-2 的面积只有 1--3%（fig_sam_buoyancy/numbers_heatloss15.txt）。所以没写 Davis Sea 等沿岸海本身，而写成这些海以北。""")

# ------------------------------------------------------------------ Reviewer 2
rep(r"""not on a resolved circulation inside each polynya, which the mesh is too coarse to represent.""",
    r"""not on the small-scale processes inside each polynya, which the mesh is too coarse to resolve.""")
rep(r"""We also mentioned that the model with higher resolution close to the continent (i.e., \textasciitilde{}5 km) would be helpful for resolve these fine scale processes.""",
    r"""We also state that resolving these processes would need a considerably finer mesh near the continent, of the order of a few kilometres.""")
rep(r"""%好，请改

\comment{R2}{The authors highlight""", r"""%好，请改
%claude回复（11-25）：信里和修订稿 Methods（第 271 行）都改了。原来信里说的 ~5 km 修订稿里没有，现在 Methods 加了一句需要更细的网格（几公里量级），信里同步。

\comment{R2}{The authors highlight""")

rep(r"""%Lhardy文章中说重建是sie jas：33加减5， jfm：10加减3，正好我们模拟的不错，可以把数据写上，然后可以简单归因于resolution，再说一下模拟的气候够冷，所以海冰够多。不要写的太多太冗长，毕竟咱们也不知道为什么模拟的好。""",
    r"""%Lhardy文章中说重建是sie jas：33加减5， jfm：10加减3，正好我们模拟的不错，可以把数据写上，然后可以简单归因于resolution，再说一下模拟的气候够冷，所以海冰够多。不要写的太多太冗长，毕竟咱们也不知道为什么模拟的好。
%claude回复（11-25）：核过 Lhardy et al. 2021 第 3.3 节：冬季 32.9、夏季 10.2（10^6 km2，示意误差 15% 和 30%），定义同为密集度 >15% 的面积。已写进回复和讨论（L255），归因只写 may help，并说没有检验过。""")
rep(r"""and the area of deep winter mixed layers contracts to roughly one fifth of its pre-industrial value. \hl{Three""",
    r"""and the area of deep winter mixed layers contracts to roughly one fifth of its pre-industrial value. The simulated LGM sea ice extent is 29.8 million km\textsuperscript{2} in winter (July to September) and 11.1 million km\textsuperscript{2} in summer (January to March). Both lie within the proxy-based estimates of Lhardy et al. (2021), about 33 ± 5 and 10 ± 3 million km\textsuperscript{2}. We have not tested why the model does well here. The relatively fine ocean mesh near Antarctica may help, and the simulated glacial climate is cold, as the following numbers show. \hl{Three""")
rep(r"""We now state that our glacial result depends on that simulated expansion being realistic, and that while proxy reconstructions support the direction of the change, the quantitative agreement remains uncertain.""",
    r"""We now state that our glacial result still depends on that simulated expansion being realistic, and that the reconstructions support the direction and the rough size of the change but carry a large uncertainty.""")
rep_block(r"""\begin{change}
``The ability of the model to maintain""", r"""remains uncertain.''
\end{change}""", change(LHARDY))

rep(r"""%可以统一，确认""", r"""%可以统一，确认
%claude回复（11-25）：已重画。A3、A4（现在按引用顺序编为 A7、A8）都换成密度通量 kg m-2 month-1、各 3 行 5 列、两张图同一色标（±2），替换了补充材料里的旧图；下面只留两张图和一句话，原来的 R22 和那段长分析删了。""")
rep_block(r"""\response We agree, and we have now put the two forcings on one scale.""", r"""and manuscript Fig. 4 and its text now include the density of water in the freshwater term.}""",
          r"""\response We agree. The two composites, Figs. 5 and 6 of the submitted version and now Supplementary Figs. A7 and A8, are redrawn in one unit, the surface density flux that each forcing produces, in kg m\textsuperscript{-2} month\textsuperscript{-1}. Both figures use one colour scale, and the unit is the same as in manuscript Figs. 3 and 4. Positive values mean that the surface water becomes denser.

\rfig{0.9}{../letter_figs/figS_sam_heat_density.png}{Revised Supplementary Fig. A7. Winter (JJA) high-SAM minus low-SAM composites of the surface density flux produced by (a-e) the total heat flux, (f-j) its radiative part (shortwave plus longwave) and (k-o) its turbulent part (sensible plus latent heat), for the five climate states. The composites use the winters in which the SAM index lies more than 1.0 standard deviation above or below its mean. Positive values mean that the surface water becomes denser. The black line is the 15\% winter sea ice edge of each state. The dashed purple line in (a-e) marks where high-SAM winters lose 15 W m\textsuperscript{-2} more heat than low-SAM winters.\label{fR:samheat}}

\rfig{0.9}{../letter_figs/figS_sam_fw_density.png}{Revised Supplementary Fig. A8. As Figure~\ref{fR:samheat}, for (a-e) the total freshwater flux, (f-j) its sea ice part and (k-o) net precipitation plus runoff, on the same colour scale.\label{fR:samfw}}

On this common scale the freshwater term is the larger of the two under sea ice, and the heat term is the larger one over most of the open water. The corresponding statement in the Results reads:

""" + change(SAM_COMMON))

rep(r"""Their simulations are ocean model-only, with a prescribed atmosphere, and only cover the Last Glacial Maximum.""",
    r"""Their simulations are ocean model-only, with a prescribed atmosphere, and cover two climate states.""")
rep_block(r"""\begin{change}
``Our conclusions about the glacial state are generally consistent""", r"""tests and quantifies the mechanism.''
\end{change}""", change(CHEN))

# ------------------------------------------------------------------ Reviewer 3
rep(r"""We diagnosed winter mixed layer depth (Figure~\ref{fR:mldR3}), reflecting""",
    r"""We diagnosed winter mixed layer depth (Figure~\ref{fR:mldR3}, now Supplementary Fig. A12), reflecting""")
rep(r"""{(New analysis) Annual mean WMT from our pre-industrial run and from ORAS5 data (1999-2018) for (a,b) the seasonal sea ice zone and (c,d) south of 60°S,  with the sea ice contribution removed. \label{fR:wmtR3cf}}""",
    r"""{New analysis. WMT from our pre-industrial run and from ORAS5 (1999--2018), both with the sea ice contribution removed, for (a,b) the seasonal sea ice zone in the annual mean and (c,d) the region south of 60°S in JJA.\label{fR:wmtR3cf}}""")
rep(r"""\claudenote{Panels (c,d) are winter (JJA) means south of 60°S, not annual means. Suggested caption: ``New analysis. WMT from our pre-industrial run and from ORAS5 (1999--2018), both with the sea ice contribution removed, for (a,b) the seasonal sea ice zone in the annual mean and (c,d) the region south of 60°S in JJA.''}""",
    r"""%claude回复（11-25）：图注已按子图实际内容改（c、d 是 60°S 以南的 JJA，不是年平均）。""")

rep_block(r"""\response Thanks for the reviewer’s comment. We have tried to answer it honestly rather than deflect it.""", r"""more relevant to the formation of intermediate and mode waters than to bottom water proper.''
\end{change}""",
          r"""\response Thank you for this question. We agree that open-ocean convection in our pre-industrial state is a model bias, and we now say so in the Discussion. The regime shift, however, does not rest on that bias. Proxy reconstructions agree that glacial winter sea ice was far more extensive than today (Lhardy et al., 2021; Green et al., 2022), and our simulated LGM winter extent lies within the reconstructed range (see our reply to Reviewer 2). Where sea ice covers the ocean it blocks the heat loss to the atmosphere, so the thermal route to dense water weakens, while brine rejection continues. The shift in our simulations follows from this. We also tested that it does not come from open water in the interglacial integrals. Over the sea ice zone alone, the glacial states are 15\% (LGM) and 17\% (MIS3) thermally driven at the transformation maximum, against 53--70\% in the interglacial states. A model that formed its dense water on the shelf would change the density and the origin of the water produced, but not the direction of this shift. We have narrowed our claim accordingly. The paper now presents how the split of surface buoyancy forcing between heat and freshwater responds to changing sea ice cover, not a reconstruction of past bottom water formation rates.

The relevant passage in the Discussion reads:

""" + change(ARTEFACT))
rep(r"""%Thank you for this question. We agree that open-ocean convection in our pre-industrial state is a model bias, and we now say so in the Discussion. The regime shift, however, does not rest on that bias. Proxy reconstructions agree that glacial winter sea ice was far more extensive than today. Where sea ice covers the ocean it blocks the heat loss to the atmosphere, so the thermal route to dense water weakens, while brine rejection continues. The shift in our simulations follows from this. We also tested that it does not come from open water in the interglacial integrals. Over the sea ice zone alone, the glacial states are 15\% (LGM) and 17\% (MIS3) thermally driven at the transformation maximum, against 53--70\% in the interglacial states. A model that formed its dense water on the shelf would change the density and the origin of the water produced, but not the direction of this shift. We have narrowed our claim accordingly. The paper now presents how the split of surface buoyancy forcing between heat and freshwater responds to changing sea ice cover, not a reconstruction of past bottom water formation rates.
%可以，按照你说的来。""",
    r"""%可以，按照你说的来。
%claude回复（11-25）：已换成压短的版本，重建文献用 Lhardy 2021 和 Green 2022（都核过，已在 ref.bib）。原来末尾那段 Pellichero 密度范围的引文换成讨论里正面回答这个问题的那段；那段里的 53--88% 改成 53--70%（海冰区内的间冰期范围，68--88% 是 60°S 以南的值）。""")

rep(r"""We located the simulated westerly jet as the latitude where the winter (JJA) 10 m westerly wind, averaged along each latitude circle, is strongest.""",
    r"""We located the simulated westerly wind belt by the latitude where the winter (JJA) 10 m westerly wind, averaged along each latitude circle, is strongest.""")
rep(r"""The atmosphere model has a grid spacing of about 1.9° in latitude, so the shift spans two grid rows. Gray et al. (2023) inferred""",
    r"""The atmosphere model has a grid spacing of about 1.9° in latitude, so the shift spans two grid rows. Gray et al. (2023) reconstruct the surface wind stress, so we also checked that quantity. The annual zonal-mean zonal wind stress over the ocean peaks at 47.6°S with 0.172 N m\textsuperscript{-2} in MH and at 49.4°S with 0.195 N m\textsuperscript{-2} in LGM, one grid row (about 1.9°) further south and about 13\% stronger. Gray et al. (2023) inferred""")
rep(r"""rather than on the jet position; Gray et al. (2023) found""", r"""rather than on the position of the westerlies; Gray et al. (2023) found""")
rep(r"""so a jet-position bias does not automatically propagate""", r"""so a bias in the position of the westerlies does not automatically propagate""")
rep(r"""%可以

\response The reviewer is correct and we now discuss this.""",
    r"""%可以
%claude回复（11-25）：已改。信和修订稿都不再用 jet 指 10 m 风，改成 westerly wind belt / wind maximum；风应力的数只算海洋格点重算过（年平均 MH 47.6°S 0.172，LGM 49.4°S 0.195 N m-2，强约 13%），也加进了修订稿第 2.1 节。

\response The reviewer is correct and we now discuss this.""")
rep(r"""the jet moves to 47.6°S and 6.92 m s\textsuperscript{-1}.""", r"""the wind maximum moves to 47.6°S and 6.92 m s\textsuperscript{-1}.""")
rep(r"""On drift: MH is indistinguishable from PI in both jet latitude (45.7°S in both) and strength (5.83 against 5.84 m s\textsuperscript{-1}).""",
    r"""On drift: MH is indistinguishable from PI in both the latitude (45.7°S in both) and the strength (5.83 against 5.84 m s\textsuperscript{-1}) of the wind maximum.""")
rep_block(r"""\begin{change}
``Residual drift over the analysis period""", r"""reported in the other paleoclimate states.''
\end{change}""", change(DRIFT))

rep(r"""% NOTE: revised.tex now: computed from the simulated vertical velocity; -11.7/-11.3/-11.4 Sv to -2.3/-1.7 Sv, roughly 80-86%; upper cell 18.5 Sv in PI against 20.1 and 21.5 Sv (D6/D8, 2026-11-08)

The overturning streamfunction, diagnosed online by the model, is now shown in the new Fig. 6. The abyssal cell weakens from -10.0 Sv in PI, -10.1 Sv in MH and -9.2 Sv in LIG, to -2.4 Sv in LGM and -1.6 Sv in MIS3, a reduction of roughly 75–85\%, while the upper cell changes comparatively small (19.1 Sv in PI against 20.0 and 21.7 Sv in the glacials).""",
    r"""The overturning streamfunction, computed from the simulated vertical velocity, is now shown in the new Fig. 5. The abyssal cell weakens from -11.7 Sv in PI, -11.3 Sv in MH and -11.4 Sv in LIG, to -2.3 Sv in LGM and -1.7 Sv in MIS3, a reduction of roughly 80 to 86\%, while the upper cell does not weaken and strengthens moderately (18.5 Sv in PI against 20.1 and 21.5 Sv in the glacials).""")
rep(r"""% NOTE: the figure file is the D6 version (geographic binning); the docx showed the earlier one

""", "")
rep(r"""{Manuscript Fig. 6. Global overturning streamfunction""", r"""{Manuscript Fig. 5. Global overturning streamfunction""")

rep(r"""\response Thanks for the reviewer’s suggestion. We have removed the phrase. The abstract now refers specifically to the abyssal overturning cell and to abyssal water.

\claudenote{The reviewer also asked for global overturning and ideal age plots. They are Figures~\ref{fR:moc} and \ref{fR:ageglobal} in the reply to the earlier comment on ideal age, and one sentence pointing there would answer that part.}""",
    r"""\response Thanks for the reviewer’s suggestion. We have removed the phrase. The abstract now refers specifically to the abyssal overturning cell and to abyssal water. The global overturning streamfunction and the global ideal age are now shown in manuscript Figs. 5 and 7 (Figures~\ref{fR:moc} and \ref{fR:ageglobal} in our reply to the comment on ideal age above).""")
rep(r"""%具体怎么写的，把文字贴过来

\response We expanded the SAM in the introduction.""",
    r"""%具体怎么写的，把文字贴过来
%claude回复（11-25）：已贴在下面，与修订稿引言那段一致。

\response We expanded the SAM in the Introduction. The new paragraph, also quoted under comment 5 of Reviewer 1, reads:

""" + change(INTRO_SAM))
rep(r"""\comment{R3}{L206-209: should be moved to the introduction maybe?}

\response We moved it.""",
    r"""\comment{R3}{L206-209: should be moved to the introduction maybe?}

\response We moved it. The definition of the SAM and its effect on surface buoyancy forcing are now in the Introduction, with the references Reviewer 1 asked for (see our reply to Reviewer 1, comments 5 and 20).""")
rep(r"""and we no longer suggest it in our discussion now.

\claudenote{Language: ``no longer'' and ``now'' say the same thing. Suggested: ``and the Discussion no longer suggests it.''}""",
    r"""and the Discussion no longer suggests it.""")

# ------------------------------------------------------------------ figure numbers after the reorder
rep(r"""Figure~\ref{fR:commonaxis} (Supplementary Fig. S5) now shows""", r"""Figure~\ref{fR:commonaxis} (Supplementary Fig. A5) now shows""")
rep(r"""{New Supplementary Fig. S5. Winter transformation""", r"""{New Supplementary Fig. A5. Winter transformation""")
rep(r"""which is where the glacial dense water is produced. This is the figure we now use in the supplement.\label{fR:mldzoom}}""",
    r"""which is where the glacial dense water is produced. This is now Supplementary Fig. A4.\label{fR:mldzoom}}""")
rep(r"""{New Supplementary Figure. (a) The positive phase""", r"""{New Supplementary Fig. A9. (a) The positive phase""")
rep(r"""{New Supplementary Figure. Zonal wind stress anomaly""", r"""{New Supplementary Fig. A10. Zonal wind stress anomaly""")
rep(r"""\response We have added this diagnosis to the Results, and Figure~\ref{fR:mld} above shows the winter mixed layer depth for all five climate states.""",
    r"""\response We have added this diagnosis to the Results. Figure~\ref{fR:mld} above shows the pre-industrial winter mixed layer depth against ORAS5, and Supplementary Fig. A2 shows it for all five climate states.""")

rep(r"""while the Ross Sea and Adelie sectors have small-to-modest contribution.''""",
    r"""while the Ross and Adélie sectors contribute little.''""")

rep(r"""the part of the haline term $\beta S F_{fw}$ that is carried""", r"""the part of the haline term $-\beta S F_{fw}$ that is carried""")

for left in ['fR:samrho', 'Fig. A3g', 'Figs. A3 and A4', 'Supplementary Fig. S5', 'new Fig. 6']:
    assert left not in T, f'left over: {left}'
open(path, 'w', encoding='utf8').write(T)
print('letter updated')
