#!/usr/bin/env python
"""User's 11-26 08:22 notes, manuscript side. Usage: python3 edits_1126b_revised.py <revised.tex>
Every anchor must match exactly once."""
import sys
p = sys.argv[1]
s = open(p).read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:80])
    s = s.replace(old, new)


# Discussion: the user's cut of the quoted passage (letter, 08:22), grammar of the last sentence fixed
rep("Because the interglacial dense water forms by open-ocean convection in the model, one may ask whether the regime shift simply reflects a biased interglacial state, with the glacial state being the more realistic of the two. Two results make this unlikely. The shift remains when the analysis is restricted to the sea ice zone. There the glacial states are 15\\% and 17\\% thermally driven and the interglacial states 53--70\\% (Fig. \\ref{domain}), so the contrast does not come from the open water that the interglacial integrals include. The shift also follows directly from the sea ice cover. Where ice covers the ocean, it shields the surface from heat loss to the atmosphere, while brine is still released where new ice grows. Both effects would also act in a model that forms its dense water on the continental shelf. Such a model would probably produce denser water in other places, and its surface buoyancy forcing would still move from heat towards brine as the ice cover expands. The glacial simulations are not free of bias either, and proxy data constrain them less well than the modern ocean. As is common in paleoclimate modelling, we therefore focus on the differences between each past state and PI, in which biases shared by both states partly cancel.",
    "The shift remains when the analysis is restricted to the sea ice zone. There the glacial states are 15\\% and 17\\% thermally driven and the interglacial states 53--70\\% (Fig. \\ref{domain}), so the contrast does not come from the open water convection that the interglacial integrals include. The shift also follows the sea ice cover. Where ice covers the ocean, it shields the surface from heat loss to the atmosphere, while brine is still released where new ice grows. The surface buoyancy forcing would become more brine-driven as the ice concentration increases.")

# ORAS5 comparison: the 'same large-scale pattern' claim does not hold (explore/mld_coastal_oras5)
rep("The winter mixed layer of the pre-industrial run has the same large-scale pattern as the ORAS5 reanalysis \\cite{Zuo2019} but is too deep in the interior of the Weddell gyre (Supplementary Fig. \\ref{mld_obs}).",
    "Compared with the ORAS5 reanalysis \\cite{Zuo2019}, the winter mixed layer of the pre-industrial run is too deep in the interior of the Weddell gyre and too shallow along the Antarctic coast (Supplementary Fig. \\ref{mld_obs}).")

# A12 caption: new colour scale
rep("so the point-by-point difference is not purely model error. Units: m. \\label{mld_obs}}",
    "so the point-by-point difference is not purely model error. The colour scale is finer between 0 and 200 m. Units: m. \\label{mld_obs}}")

# A13 caption: ice edge contour removed from the figure (user, GC)
rep("Lines in (a,b) mark the 1000 m isobath, the 15\\% winter sea ice edge, the 400 m winter mixed layer depth and 60°S.",
    "Lines in (a,b) mark the 1000 m isobath, the 400 m winter mixed layer depth and 60°S.")

open(p, 'w').write(s)
print('revised.tex: 4 edits applied')
