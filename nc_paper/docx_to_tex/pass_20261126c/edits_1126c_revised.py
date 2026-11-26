#!/usr/bin/env python
"""User's 11-26 09:48-10:46 notes, manuscript side. Usage: python3 edits_1126c_revised.py <revised.tex>
Every anchor must match exactly once."""
import sys
p = sys.argv[1]
s = open(p).read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:90])
    s = s.replace(old, new)


# Results 2.1: the Gray/PMIP passage duplicates the Discussion (user: delete)
rep(" We note that the simulated glacial westerlies are displaced poleward, whereas proxy reconstructions infer an equatorward shift of about 4.8° and a weakening of roughly 25\\% at the Last Glacial Maximum relative to the mid-Holocene \\cite{Gray2023}. The surface wind stress, the quantity these reconstructions target, shows the same poleward shift in our simulations. Its annual zonal-mean maximum over the ocean lies at 47.6°S with 0.172~N~m$^{-2}$ in the mid-Holocene and at 49.4°S with 0.195~N~m$^{-2}$ in LGM, one grid row (about 1.9°) further south and about 13\\% stronger. The four PMIP4 models with Last Glacial Maximum output available to us \\cite{Kageyama2021} move the winter 10 m wind maximum by less than 1.2° between LGM and PI and change its strength by 6\\% or less. They do not reproduce the reconstructed shift either, and our simulated poleward shift is larger than in any of them. We return to this disagreement, and to its consequences for our conclusions, in the Discussion.", "")

# Discussion: the user's edited version of the Gray paragraph (letter quote, 10:xx)
rep("A related limitation concerns the simulated glacial winds. The westerlies in our glacial experiments shift poleward and strengthen, whereas proxy reconstructions infer an equatorward shift of about 4.8° and a weakening of about 25\\% at the Last Glacial Maximum relative to the mid-Holocene \\cite{Gray2023}. Gray et al. \\cite{Gray2023} found that the inferred shift exceeds that of any PMIP3/4 model. In the four PMIP4 models with Last Glacial Maximum output available to us, the winter wind maximum moves by less than 1.2° between LGM and PI, so current models in general do not reproduce the reconstructed change. Our model departs further in the opposite direction, with a poleward shift of 3.7°, which is two grid rows of the atmosphere model. This affects the two parts of our results differently. The mean-state regime shift depends on the area of sea ice and on brine rejection. Gray et al. \\cite{Gray2023} found no correlation across the model ensemble between Antarctic sea ice extent and either the latitude of the westerlies or the position of the surface temperature front. A bias in the position of the westerlies therefore need not carry over into the sea ice field on which our main result rests. The Southern Annular Mode results describe how the model responds to a wind anomaly added to a background state whose mean winds are biased. We therefore read them as a description of the mechanism by which wind variability changes transformation, and we do not use them to reconstruct glacial wind-driven variability.",
    "A related limitation concerns the simulated glacial winds. The westerlies in our glacial experiments shift poleward and strengthen, whereas proxy reconstructions infer an equatorward shift of about 4.8° and a weakening of about 25\\% at the LGM relative to the mid-Holocene \\cite{Gray2023}. Gray et al. \\cite{Gray2023} found that the inferred shift exceeds that of any PMIP3/4 model. In the four PMIP4 LGM simulations \\cite{Kageyama2021}, the winter wind maximum moves by less than 1.2° between LGM and PI, so current models in general do not reproduce the reconstructed change. Our model departs further in the opposite direction, with a poleward shift of 3.7°. Moreover, Gray et al. \\cite{Gray2023} found no correlation across the model ensemble between Antarctic sea ice extent and either the latitude of the westerlies or the position of the surface temperature front. A bias in the position of the westerlies therefore need not carry over into the sea ice field on which our main result rests. Thus the SAM results describe how the model responds to a wind anomaly added to a background state whose mean winds are biased.")

# push/pull: panel c (scatter, r = 0.77) removed from Fig. A9 (user)
rep("Across the five states the two responses scale together ($r = 0.77$; Supplementary Fig. \\ref{sam_pushpull}c), and the transformation response follows both (Supplementary Fig. \\ref{sam_pushpull}d).",
    "Both responses are positive in all five states, and so is the response of the peak transformation rate (Supplementary Fig. \\ref{sam_pushpull}c).")
rep("(c) The two quantities plotted against each other across the five climate states, showing that they respond in concert ($r = 0.77$). (d) The resulting anomaly of the peak Southern Ocean transformation rate, with the density class of the maximum indicated. \\label{sam_pushpull}}",
    "(c) The resulting anomaly of the peak Southern Ocean transformation rate, with the density class of the maximum indicated. \\label{sam_pushpull}}")

# Methods: back to the submitted wording plus the SST and AMOC trends (user)
rep("We analyze the final 100 years of each run, over which the simulated climate is in a quasi-equilibrium state, and the climatology of each variable is represented by its average over those 100 years. Residual drift over the analysis period is small relative to the paleoclimate signals of interest. The trend in global mean 2~m temperature across the final century is $+0.02$~K~century$^{-1}$ for PI, $-0.03$ for the mid-Holocene, $-0.06$ for the Last Interglacial, $+0.04$ for LGM and $+0.10$ for MIS3. The Last Interglacial and MIS3 values exceed the $\\pm0.05$~K~century$^{-1}$ criterion applied in earlier work with this model \\cite{Shi2022JCLI}, and we note this explicitly.",
    "We analyze the final 100 years of each run, representing a quasi-equilibrium state. Over these 100 years the trend of global mean sea surface temperature is within $\\pm0.03$~K~century$^{-1}$ in PI, the mid-Holocene, the Last Interglacial and LGM and $+0.06$~K~century$^{-1}$ in MIS3. The trend of the Atlantic meridional overturning circulation is not significant, except for $+1.3$~Sv~century$^{-1}$ in PI ($p = 0.04$). These are the two quantities by which the PMIP4 protocol judges equilibrium \\cite{Kageyama2017}.")
i = s.index("The glacial-interglacial differences examined here are large by comparison:")
j = s.index("against a glacial weakening of about 9~Sv.", i) + len("against a glacial weakening of about 9~Sv.")
para = s[i:j]
assert para.count("\n\n") == 0 and "\\section" not in para, 'drift paragraph boundary'
# drop the paragraph together with the blank line before it
k = s.rindex("\n\n", 0, i)
assert s[k:i].strip() == '', 'paragraph is not preceded by a blank line'
s = s[:k] + s[j:]

open(p, 'w').write(s)
print('revised.tex: edits applied')
