"""Apply the 2026-11-22 highlight pass to revision/response_new.tex.

Usage: python3 edits_1122.py <response_new.tex> <seaice_block.tex> <comment11_block.tex> <discussion_seaice.tex> <r2_sam_block.tex>
Every anchor must match exactly once, so the script refuses to run on a
version that the user has changed in the same place.
"""
import re
import sys

path = sys.argv[1]
SEAICE = open(sys.argv[2]).read().strip()
C11 = open(sys.argv[3]).read().strip()
DISC_SEAICE = open(sys.argv[4]).read().strip()
R2SAM = open(sys.argv[5]).read().strip()
s = open(path).read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:120])
    s = s.replace(old, new)


def rep_re(pat, new, count=1):
    global s
    m = list(re.finditer(pat, s, flags=re.S))
    assert len(m) == count, (len(m), pat[:120])
    s = re.sub(pat, lambda _: new, s, flags=re.S)


# ---------------------------------------------------------------- preamble
rep(r"""\DeclareRobustCommand{\docxcomment}[4]{{\upshape\sffamily\footnotesize\color{red}[Comment #1 (#2) on ``#3'': #4]}}
""", r"""\DeclareRobustCommand{\docxcomment}[4]{{\upshape\sffamily\footnotesize\color{red}[Comment #1 (#2) on ``#3'': #4]}}
% Note added in the 2026-11-22 pass: a point still for the authors to decide. Delete the note once settled.
\DeclareRobustCommand{\claudenote}[1]{{\upshape\sffamily\footnotesize\color{magenta}[Note: #1]}}
""")

# ---------------------------------------------------------------- R1, Fig. R1 caption
rep(r"""(b) the observational estimate of Pellichero et al. (2018), their Fig. 3a, \hl{reproduced} for comparison.""",
    r"""(b) the observation-based estimate of Pellichero et al. (2018), shown as their published Fig. 3a, copied unchanged from their paper.""")
rep(r"""Where the same band is quoted from the binned diagnostic of \hl{Fig. R2.3} it appears as γ = 27.33–27.59 kg m\textsuperscript{-3}, the single bin that falls inside it.""",
    r"""When the transformation is instead summed in density classes 0.25 kg m\textsuperscript{-3} wide in σ\textsubscript{2}, as in our reply to Reviewer 2, the only class that falls inside this band is γ = 27.33–27.59 kg m\textsuperscript{-3}.""")
# the same word in the Reviewer 2 and Reviewer 3 captions
rep(r"""(b) the estimate of Pellichero et al. (2018), their Fig. 3a, reproduced for comparison.""",
    r"""(b) the estimate of Pellichero et al. (2018), shown as their published Fig. 3a, copied unchanged from their paper.""", count=2)
rep(r"""The broad pattern shown in the ORAS5 reanalysis is reproduced by our pre-industrial simulation.""",
    r"""Our pre-industrial simulation shows the same broad pattern as the ORAS5 reanalysis.""", count=2)

# ---------------------------------------------------------------- R1, a data problem found on 2026-11-22
rep(r"""Pellichero et al. (2018) reported a minimum near -22 Sv and a densification peak of 5±5 Sv. The sign structure and the order of magnitude of the densification branch therefore agree.""",
    r"""Pellichero et al. (2018) reported a minimum near -22 Sv and a densification peak of 5±5 Sv. The sign structure and the order of magnitude of the densification branch therefore agree.

\claudenote{Found on 2026-11-22 and not yet fixed. The sea ice zone behind panels a and c (and behind the 42.7\%, 70\%, -9.2 and +6.6 Sv values here and in the replies to Reviewers 2 and 3) was built from pi/a\_ice\_clim.nc. That file was overwritten on 2026-03-15 with a 10-year mean (model years 2090-2099) of a run we could not identify. The PI period of the paper (years 2300-2399) is kept in a\_ice\_clim\_beta.nc and gives a 13\% smaller September ice zone. The same overwritten PI file (a\_ice\_reg.nc) is the PI reference of manuscript Figs. 1 and 2, which were drawn in March and April, and it enters Fig. 4. These numbers and figures should be recomputed with the 2300-2399 file before resubmission.}""")

# ---------------------------------------------------------------- R1, comment 1 (limits of the observational estimate)
rep_re(r"""\\docxcomment\{1\}\{[^{}]*\}\{alone\.\}\{[^{}]*\} Some discrepancy is therefore expected""",
       r"""Their estimate also has limits of its own. It combines Argo floats, ship-based profiles and instrumented seals, but the data are still sparse under winter sea ice, and their densification estimate of 5±5 Sv carries an uncertainty as large as the value itself. Some discrepancy is therefore expected""")

# ---------------------------------------------------------------- R1, ORAS5 caveat
rep(r"""ORAS5 is a model constrained by observations rather than a measurement, \hl{and the two fields the calculation needs are the two that are least constrained}, since no observing system measures an air-sea flux directly; its heat flux is computed""",
    r"""ORAS5 is a model constrained by observations rather than a measurement. The calculation needs its surface heat and freshwater fluxes, and these are the parts of the reanalysis that observations constrain least, because no observing system measures air-sea fluxes directly over the Southern Ocean. Its heat flux is computed""")

# ---------------------------------------------------------------- R1, model limitations: sea ice against satellite data
rep(r"""\hl{Add a winter sea ice cover plot, model vs. satellite, and discuss the similarity and difference.}""", SEAICE)

rep_re(r"""In addition, the published evaluation reports \\docxcomment\{2\}\{[^{}]*\}\{the published evaluation reports\}\{Provide reference!\} that Antarctic sea ice in this configuration is thiner \\hl\{compared to \?\}, especially in the central Weddell Sea in September, and that the Southern Ocean has a warm bias of 1\.43 K\. The abyssal overturning cell is reported at about 10 Sv, comparable to our pre-industrial value\.""",
       r"""The model evaluation of Sidorenko et al. (2019), which uses the same coupled model on the same horizontal ocean mesh, points the same way. In their 1000-year pre-industrial run the Antarctic sea ice volume stays low, the ice hardly reaches 0.25 m thickness in the central Weddell Sea in September, and the September sea ice area of the Southern Hemisphere keeps decreasing through the run. They relate the thin ice to a surface warming of about 1 °C over the Southern Ocean during the integration, but they do not compare the ice thickness with observations. In their historical simulation at the same resolution the Southern Ocean surface is too warm, which they associate with the reduced sea ice area. Their abyssal overturning cell reaches about 10 Sv, comparable to our pre-industrial value. Observation-based estimates of bottom water formation are of the same order. From chlorofluorocarbon inventories, Orsi et al. (1999) estimated a circumpolar Antarctic Bottom Water production of about 8 Sv, or 9.5 Sv if the bottom layer is assumed to be well mixed.""")

# Discussion passage quoted under the model-limitations reply (comment 3)
rep_re(r"""``Independent evaluations of this model provide context for these limitations, and we summarise the relevant points here rather than leaving them to be sought in the model description literature\.\\docxcomment\{3\}\{[^{}]*\}\{literature\.\}\{[^{}]*\} Sidorenko et al\. \(2019\) evaluate the coupled configuration.*?consistent with the convection behaviour we diagnose above\.''""",
       r"""``Independent evaluations of this model provide context for these limitations, and we summarise the relevant points here rather than leaving them to be sought in the model description literature. Sidorenko et al. (2019) evaluate the same coupled configuration and report an abyssal overturning cell of about 10 Sv, comparable to the pre-industrial value obtained here. They also find that the Antarctic sea ice of this configuration is thin, hardly reaching 0.25 m in the central Weddell Sea in September, and that the Southern Ocean surface is too warm at this resolution, which they associate with the reduced sea ice area. Their overall assessment is that although biases in the representation of water mass properties and ventilation mechanisms are present, the model still produces a reasonable density distribution that maintains realistic transports. Our own comparisons with observation-based products point to the same bias. """ + DISC_SEAICE + r""" A warm Southern Ocean with too little winter sea ice is the state in which open-ocean convection is favoured over shelf processes, so these biases are consistent with the convection behaviour we diagnose above.''
\end{change}

\claudenote{This Discussion passage is new text. revised.tex (line 245) still has the old version, which also misreads Sidorenko et al. (2019): their 1.43 K is the global sea surface temperature RMSE of the historical run against PHC, not a Southern Ocean bias. If you keep this version, the manuscript passage and the TODO comment at line 366 should be updated, and the sea ice and mixed layer comparison figures need a place in the Supplement.}
\begin{change}
\ignorespaces""")
# the extra change environment opened above is closed by the original \end{change}; make it empty-safe
rep("\\begin{change}\n\\ignorespaces\n\\end{change}", "")

# ---------------------------------------------------------------- R1, comment 4 (abstract bridge)
rep(r"""stating that weaker glacial transformation coincides with a weakening""",
    r"""stating that the weaker glacial transformation of surface water into dense water coincides with a weakening""")
rep(r"""``\hl{Weaker glacial transformation} coincides with a weakening of the simulated abyssal overturning cell""",
    r"""``The weaker transformation of surface water into dense water in the glacial simulations coincides with a weakening of the simulated abyssal overturning cell""")

# ---------------------------------------------------------------- R1, comment 5 (SAM introduction references)
rep(r"""The modern observation showed that these adjustments modulate mixed-layer depth and surface buoyancy forcing, and have been linked to changes in bottom water properties \hl{(Add references).}""",
    r"""In the present-day ocean these adjustments modulate mixed-layer depth and surface buoyancy forcing (Sallée et al., 2010), and they have been linked to observed changes in bottom water properties (Silvano et al., 2020; Zhou et al., 2023).""")

# ---------------------------------------------------------------- R1, comments 9 and 10 ("residual")
rep(r"""The \hl{residual} glacial signals in coastal regions are shallower than their interglacial counterparts""",
    r"""The deep mixed layers that remain along the coast in the glacial states are shallower than their interglacial counterparts""")
rep(r"""The text now says explicitly that the residual glacial signals are shallower than their interglacial counterparts.""",
    r"""The text now says explicitly that the deep mixed layers that remain along the glacial coast are shallower than their interglacial counterparts.""")

# ---------------------------------------------------------------- R1, comment 11 (Fig. 1i-l, sea ice anomalies)
rep(r"""\response \hl{You need to discuss sea ice anomalies!}""", C11)

# ---------------------------------------------------------------- R1, comment 13 (Ross Sea)
rep(r"""the glacial sea ice response is in fact fairly uniform around the continent: concentration rises by 0.62 in the Ross Sea sector and 0.70 in the Weddell Sea sector at the Last Glacial Maximum, and the two sectors cool to within about 0.1 K of the surface freezing point, so \hl{the ice expansion is \textit{ facilitated by intensified wind-induced Ekman transport}}\textit{.}""",
    r"""the glacial sea ice response is in fact similar in the two sectors. At the Last Glacial Maximum the ice concentration rises by 0.61 in the Ross sector and 0.80 in the Weddell sector, both sectors become almost fully ice covered (0.95), and both cool to within 0.05 K of the surface freezing point. In the glacial states the Ross Sea therefore does gain ice, and there is no Ross Sea exception to explain. The Ross sector departs from the Weddell sector in the last interglacial instead.""")
rep_re(r"""\\docxcomment\{4\}\{[^{}]*\}\{Figure R5 \(new analysis\}\{Only analyze Weddell and Ross two sectors \} New analysis\. \(a\) Winter surface temperature above the local freezing point and \(b\) winter sea ice concentration, by sector and climate state\. \(c\) The same quantities as anomalies relative to PI\. The Ross sector separates from the others only in the last interglacial, where it warms to 2\.7 K above freezing and loses 0\.19 of ice concentration, an order of magnitude more than the Weddell or Adélie response\. In the glacial states all three sectors collapse onto the freezing point and gain ice comparably\.""",
       r"""New analysis. (a) Winter (JJA) surface temperature above the local freezing point and (b) winter sea ice concentration in the Ross sector (180°–60°W) and the Weddell sector (60°W–79°E), averaged between 60 and 75°S, for each climate state. (c) The same quantities as anomalies relative to PI. The Ross sector separates from the Weddell sector only in the last interglacial, where it sits 2.7 K above freezing and loses 0.20 of ice concentration while the Weddell sector gains 0.08. In the glacial states both sectors reach the freezing point and gain ice by similar amounts.""")
rep(r"""So in the last interglacial the Ross sector is 0.97 K warmer than in PI and sits 2.72 K above the freezing point, against changes below \hl{0.3 K} elsewhere, and its winter ice cover falls by 0.19.""",
    r"""In the last interglacial the surface of the Ross sector sits 2.72 K above the freezing point, 0.98 K further from freezing than in PI, while the Weddell sector changes by only 0.12 K. The winter ice concentration of the Ross sector falls by 0.20, while that of the Weddell sector rises by 0.08.""")
rep(r""" This asymmetry is also why the Ross sector departs from the circumpolar pattern in the \hl{variability} analysis.""", "")

# ---------------------------------------------------------------- R1, comment 17 (Fig. 3 colour bars, comment 5)
rep_re(r"""Reviewer 3 made the same request\\docxcomment\{5\}\{[^{}]*\}\{request\}\{You may just keep the bottom color bar\} \.""",
       r"""Reviewer 3 made the same request. All rows of the figure share one colour scale for PI and one for the anomalies, in the same unit, so we now draw each colour bar once, under the bottom row, and label the rows on the left.""")
rep(r"""so that the columns are identified in the same way as in the other figures. The freshwater counterpart, Fig. 4, has been given the same headings.""",
    r"""so that the columns are identified in the same way as in the other figures. The rows share one colour bar for PI and one for the anomalies, drawn under the bottom row. The freshwater counterpart, Fig. 4, has been treated the same way.""")

# ---------------------------------------------------------------- R1, comment 20 and 24 (reference lists)
rep(r"""\hl{List these references here}

Hall, A., Visbeck, M., Synchronous Variability in the Southern Hemisphere Atmosphere, Sea Ice, and Ocean Resulting from the Annular Mode, \textit{J. Clim.} 15, 3043–3057 (2002).""",
    r"""Hall, A., Visbeck, M., Synchronous Variability in the Southern Hemisphere Atmosphere, Sea Ice, and Ocean Resulting from the Annular Mode, \textit{J. Clim.} 15, 3043–3057 (2002).

Cerovečki, I., Talley, L. D., Mazloff, M. R., Maze, G., Water Mass Transformations in the Southern Ocean Diagnosed from Observations: Contrasting Effects of Air–Sea Fluxes and Diapycnal Mixing, \textit{J. Phys. Oceanogr.} 43, 1485–1511 (2013).

Abernathey, R. P., Cerovecki, I., Holland, P. R., Newsom, E., Mazloff, M., Talley, L. D., Water-mass transformation by sea ice in the upper branch of the Southern Ocean overturning, \textit{Nat. Geosci.} 9, 596–601 (2016).

\claudenote{Hall and Visbeck (2002) is not yet cited in revised.tex and not in ref.bib. Cerovečki et al. (2013) and Abernathey et al. (2016) are cited there (line 142).}""")
rep(r"""more generally \hl{(Add the two references?)} .''""",
    r"""more generally (Ohshima et al., 2013; Abernathey et al., 2016).''""")
rep(r"""\end{change}

\hl{List these references here}""",
    r"""\end{change}

Ohshima, K. I., Fukamachi, Y., Williams, G. D., Nihashi, S., Roquet, F., Kitade, Y., Tamura, T., Hirano, D., Herraiz-Borreguero, L., Field, I., Hindell, M., Aoki, S., Wakatsuchi, M., Antarctic Bottom Water production by intense sea-ice formation in the Cape Darnley polynya, \textit{Nat. Geosci.} 6, 235–240 (2013).

Abernathey, R. P., Cerovecki, I., Holland, P. R., Newsom, E., Mazloff, M., Talley, L. D., Water-mass transformation by sea ice in the upper branch of the Southern Ocean overturning, \textit{Nat. Geosci.} 9, 596–601 (2016).""")

# ---------------------------------------------------------------- R1, comment 26
rep(r"""by which the freshwater term exceeds the heat term in \hl{their sector}, and we compare""",
    r"""by which the freshwater term exceeds the heat term inside the region they analyse, the area enclosed by the September sea ice edge, and we compare""")

# ---------------------------------------------------------------- R2 and R3, domain paragraph (twice)
rep(r"""region enclosed bythe contour""", r"""region enclosed by the contour""", count=2)
rep(r"""and is therefore \hl{water} their analysis does not include.""",
    r"""and therefore lies outside the region their analysis covers.""", count=2)
rep(r"""Recomputing our transformation over the ice covered sector defined exactly as theirs, which reproduces their domain to within 2.5\% in area, sharpens the picture considerably.""",
    r"""Recomputing our transformation over the ice covered sector, defined in the same way as theirs, sharpens the picture considerably. Because the simulated winter ice zone is smaller than the observed one (Figure~\ref{fR:seaice}), this region is also smaller in our model than in their analysis.""", count=2)
rep(r"""so what they recover is the surface flux together with the diapycnal mixing term, \hl{which that budget cannot separate}, whereas""",
    r"""so what they recover is the surface flux together with the diapycnal mixing term, because their budget method cannot tell the two apart, whereas""", count=2)

# ---------------------------------------------------------------- R1 and R2, ORAS5 winter peak (not highlighted, wrong)
rep(r"""and both peak in the same density class at γ 27.28 and 27.08 respectively.""",
    r"""and they peak one density class apart, at γ 27.28 and 27.08 respectively.""")
rep(r"""both are positive almost everywhere and peak in the same density class, but ours is the stronger""",
    r"""both are positive almost everywhere and peak one density class apart, but ours is the stronger""", count=2)

# ---------------------------------------------------------------- R2, polynya resolution
rep(r"""so the brine signal is represented in the model when new ice forms, \hl{but its intensity depends on the simulated wind-driven divergence rather than on resolved polynya circulation}.""",
    r"""so the brine signal is represented in the model whenever new ice forms. How much brine is released therefore depends on how strongly the simulated winds and currents push the ice apart and open the water, not on a resolved circulation inside each polynya, which the mesh is too coarse to represent.""")

# ---------------------------------------------------------------- R2, how the model achieves its glacial ice cover (comment 8)
C8 = re.search(r"""\\docxcomment\{8\}\{[^{}]*\}\{expands\}\{[^{}]*\}""", s).group(0)
rep_re(r"""\\hl\{In our simulations the glacial ice expands\}(\\docxcomment\{8\}\{[^{}]*\}\{expands\}\{[^{}]*\}) sufficiently to insulate the interior gyres, and the area of deep winter mixed layers contracts to roughly one fifth of its pre-industrial value\.""",
       r"""In our simulations the glacial winter ice expands far enough to cover the interior gyres, and the area of deep winter mixed layers contracts to roughly one fifth of its pre-industrial value. Three simulated changes work together to produce this ice cover. The winter air over the ocean between 55 and 70°S is 14 K colder in LGM and 12 K colder in MIS3 than in PI. The surface water between 60 and 75°S stays within 0.2 K of its freezing point in every sector, against 1.5–2.0 K above freezing in PI, so the upper ocean holds almost no heat that could melt the ice or stop new ice from forming. The glacial westerlies are also stronger (Fig. 1o,p), which favours the northward export of ice. These changes reinforce each other, because ice that covers the ocean also stops the ocean from warming the air above it. @@C8@@ \claudenote{Comment 8 is not built into the text yet. Our output shows brine release along the coast, but we have no diagnostic showing that brine rejection causes the ice expansion. Please say what you have in mind.}""")
rep('@@C8@@', C8)
rep(r"""\hl{The haline-dominated glacial regime reported here means our glacial result depends on the simulated sea ice expansion being realistic.}""",
    r"""Because the glacial regime reported here is driven by sea ice, our glacial result holds only if the simulated sea ice expansion is realistic.""")

# ---------------------------------------------------------------- R2, Figs 5 and 6 on one buoyancy scale
rep_re(r"""\\response \\hl\{We have added a statement of the comparison in equivalent surface density tendency terms to the Results:.*?\n\n\\hl\{The corresponding statement in the Results reads:``figures themselves, which are now supplementary,.*?The text now makes the relative magnitudes explicit\.\}""",
       R2SAM)

# ---------------------------------------------------------------- R3, first comment: rebuilt in the order the reviewer asks
R3QUOTE1 = open(sys.argv[6]).read().strip()
R3QUOTE2 = open(sys.argv[7]).read().strip()
R3SEAICE = open(sys.argv[8]).read().strip()
a = s.index(r"""\rfig{0.9}{../letter_figs/figR20_wmt_merged.png}{New analysis. Transformation rate against neutral density γ, in three comparisons. The seasonal sea ice zone in the annual mean""")
m9 = re.search(r"""\\docxcomment\{9\}\{[^{}]*\}\{[^{}]*\}\{Rewrite\}\n""", s[a:])
assert m9
b = a + m9.end()
seg = s[a:b]
o1 = seg.index("Compared Fig.~\\ref{fR:wmtR3ab} a and b")
o2 = seg.index("\\rfig{0.9}{../letter_figs/figR20_wmt_merged.png}{\\hl{New analysis.")
obs_text = seg[o1:o2].strip()
r1 = seg.index("An independent reanalysis supports the diagnostic.")
r2 = seg.index("\\hl{The passage added to the Discussion reads:}")
rean_text = seg[r1:r2].strip()
RE_HEAD = "An independent reanalysis supports the diagnostic. We also computed the transformation from ORAS5 over both domains, removing the sea ice term from our own flux to match what the reanalysis can supply (Fig.~\\ref{fR:wmtR3cf}c-f). "
assert rean_text.count(RE_HEAD) == 1, rean_text[:200]
newseg = (R3SEAICE + "\n\n"
 + "The passage added to the Discussion reads:\n\n\\begin{change}\n``" + R3QUOTE1 + "''\n\\end{change}\n\n"
 + "Further on, the Discussion adds:\n\n\\begin{change}\n``" + R3QUOTE2 + "''\n\\end{change}\n\n"
 + "On the reference the reviewer gives, we cite Toggweiler and Samuels (1995), ``Effect of sea ice on the salinity of Antarctic bottom waters''. That paper is about brine released by sea ice on the continental shelf rather than about melting beneath ice shelves. The same authors published a second paper in 1995, on Drake Passage, which is the one more often cited, so we give the title to make clear which one we mean. Its conclusion that ocean models are too fresh near Antarctica because of errors in their circulation, not because of their surface boundary conditions, applies directly to our model.\n\n"
 + "The reviewer also suggests testing the difference by comparing our model only in the regions that have observations. We did this, and the result is shown in Figure~\\ref{fR:wmtR3ab}.\n\n"
 + "\\rfig{0.9}{../letter_figs/figR20a_wmt_obs.png}{New analysis, the observational part of the comparison. Transformation rate against neutral density γ in the seasonal sea ice zone, annual mean, with all terms present. (a) Our pre-industrial state, split into the heat, freshwater and sea ice contributions. (b) The observation-based estimate of Pellichero et al. (2018), shown as their published Fig. 3a, copied unchanged from their paper. Pink shading in (a) marks where the heat term carries the total and blue shading where the freshwater term does.\\label{fR:wmtR3ab}}\n\n"
 + obs_text + "\n\n"
 + "The reviewer's second suggestion was to apply the transformation framework to a reanalysis such as GLORYS. A transformation budget needs the net surface heat flux and the net surface freshwater flux. The GLORYS12 monthly product (GLOBAL\\_MULTIYEAR\\_PHY\\_001\\_030) provides temperature, salinity, currents, sea surface height, mixed layer depth and sea ice, but no surface flux fields, so the calculation cannot be made from it. ORAS5 does provide the net surface heat flux and the net freshwater flux, although its freshwater flux leaves out the salt exchanged when sea ice forms and melts. We therefore computed the transformation from ORAS5 and removed the sea ice term from our model in the same way, so that both sides contain the same two terms. The result is shown in Figure~\\ref{fR:wmtR3cf}.\n\n"
 + "\\rfig{0.9}{../letter_figs/figR20b_wmt_oras5.png}{New analysis, the reanalysis part of the comparison. Transformation rate against neutral density γ from our pre-industrial run and from ORAS5 for 1999-2018. Top row, the seasonal sea ice zone in the annual mean: (c) our model with the sea ice contribution removed and (d) ORAS5, which carries the heat flux and evaporation minus precipitation minus runoff only. Bottom row, the domain and season the paper uses, south of 60°S in JJA: (e) our model with the sea ice contribution removed and (f) ORAS5. The panels are not shaded by the dominant term, because with the sea ice term missing from both sides such shading would describe an incomplete freshwater flux.\\label{fR:wmtR3cf}}\n\n"
 + rean_text.replace(RE_HEAD, "") + " ORAS5 is itself a model constrained by observations, and its surface fluxes are the parts that observations constrain least, so we use this comparison as a check on our calculation rather than as an observational truth.\n")
s = s[:a] + newseg + s[b:]

# ---------------------------------------------------------------- R3, generalisability
rep(r"""Recomputed over the sea ice sector alone, the glacial states remain \hl{15\% and 17\%} thermally driven while the interglacials remain \hl{53–88\%} thermally driven. The contrast is therefore not an artefact of including open water in the interglacial integrals, which is the most obvious way \hl{the biased end member could have manufactured it}.""",
    r"""Recomputed over the sea ice zone alone, the glacial states are 15\% (LGM) and 17\% (MIS3) thermally driven at the transformation maximum, while the interglacials are 53–70\% thermally driven (70\% in PI, 53\% in MH and 68\% in LIG). Over the original domain south of 60°S the interglacial values are 68–88\%. The contrast therefore does not come from including open water in the interglacial integrals. That would have been the most direct way for a biased pre-industrial state to create a regime shift that is not real.""")
rep(r"""whereas the haline-dominated lower cell \hl{they describe} occupies""",
    r"""whereas the lower overturning cell, which Pellichero et al. (2018) find to be driven by freshwater fluxes, occupies""")

# ---------------------------------------------------------------- R3, SAM spatial patterns
rep(r"""The spatial patterns show the mechanism directly (\hl{Figure~\ref{fR:windice}}).\textbf{ }These results""",
    r"""The spatial patterns show the mechanism directly (Figure~\ref{fR:windice}). These results""")

# ---------------------------------------------------------------- R3, westerly jet
rep(r"""We measured our simulated jet: the winter \hl{zonal-mean maximum} sits at 45.7°S with 5.83 m s\textsuperscript{-1} in PI and moves to 49.4°S with 6.30 and 6.85 m s\textsuperscript{-1} in LGM and MIS3, a poleward shift of 3.7° with strengthening.""",
    r"""We located the simulated westerly jet as the latitude where the winter (JJA) 10 m westerly wind, averaged along each latitude circle, is strongest. In PI this maximum sits at 45.7°S with 5.83 m s\textsuperscript{-1}, and in LGM and MIS3 it sits at 49.4°S with 6.30 and 6.85 m s\textsuperscript{-1}. That is a poleward shift of 3.7° together with stronger winds. The atmosphere model has a grid spacing of about 1.9° in latitude, so the shift spans two grid rows.""")

# ---------------------------------------------------------------- R3 minor comments
rep(r"""We have retained them thereafter because the figures are organised in five columns labelled by these codes. We spell the names \hl{(i.e., MH, LIG) out} .""",
    r"""We have kept them after that, because the figures are organised in five columns labelled with these codes and spelling the names out each time would make the text much longer. The abstract spells the names out in full.""")
rep(r"""\response Thanks for the reviewer’s suggestion. \hl{Abyssal overturning cell and abyssal water specifically,} and the new global age figure covers the whole water column rather than a single level.""",
    r"""\response Thanks for the reviewer’s suggestion. We have removed the phrase. The abstract now refers specifically to the abyssal overturning cell and to abyssal water, and the new global ideal age figure covers the whole water column rather than a single level.""")
rep(r"""Reviewer 1 asked for a supporting reference for the same passage; \hl{we concluded the proposal was not well founded and have withdrawn it.}""",
    r"""Reviewer 1 asked for a supporting reference for the same passage. We agree that the slow sedimentation around the Antarctic margin makes such a reconstruction impractical, and we no longer suggest it. The passage now presents the persistence of the coupling only as a property of the simulated system (see our reply to Reviewer 1, comment 29).""")

open(path, 'w').write(s)
print('ok, remaining \\hl:', s.count('\\hl{'), ' docxcomment:', s.count('\\docxcomment{') - 1)
