#!/usr/bin/env python
"""
Move the SAM composite figures from the main text into the supplementary section
(Reviewer 2's request) and add the two new SAM mechanism figures there
(Reviewer 3's request for more SAM analysis).
"""
import re
import sys
from pathlib import Path

P = Path('build/revised.tex')
s = P.read_text()


def extract(label):
    """Cut the \\begin{figure}...\\end{figure} block containing \\label{label}."""
    i = s.find(f'\\label{{{label}}}')
    if i < 0:
        raise SystemExit(f'label {label} not found')
    b = s.rfind('\\begin{figure}', 0, i)
    e = s.find('\\end{figure}', i) + len('\\end{figure}')
    return b, e, s[b:e]


# pull the two SAM composite figures out of the main text
blocks = {}
for lab in ['sam_heat', 'sam_fwf']:
    b, e, blk = extract(lab)
    blocks[lab] = blk
    s = s[:b] + s[e:]

# tidy any leftover blank runs where they were
s = re.sub(r'\n{4,}', '\n\n\n', s)

# fix the stale "1 sigma" in those captions while we have them
for lab in blocks:
    blocks[lab] = blocks[lab].replace('(1$\\sigma$ threshold)', '(1.2$\\sigma$ threshold)')

NEW_FIGS = r"""

\begin{figure}
\centering
\includegraphics[width=\textwidth]{figures/figR4_sam_push_pull.pdf}
\caption{The Southern Annular Mode acts on both the wind-driven upwelling and the sea ice salt pump, in every simulated climate state. All quantities are austral winter composites, high-SAM minus low-SAM years at the $\pm$1.2$\sigma$ threshold. (a) Anomaly of the zonally integrated northward Ekman transport at 60°S, computed from the simulated zonal wind stress. (b) Anomaly of the coastal brine-equivalent freshwater loss integrated south of 65°S, where a positive value denotes a stronger salt input to the ocean. (c) The two quantities plotted against each other across the five climate states, showing that they respond in concert ($r = 0.77$). (d) The resulting anomaly of the peak Southern Ocean transformation rate, with the density class of the maximum indicated. \label{sam_pushpull}}
\end{figure}

\clearpage

\begin{figure}
\centering
\includegraphics[width=\textwidth]{figures/figR5_sam_wind_ice.pdf}
\caption{Spatial expression of the Southern Annular Mode mechanism, shown as austral winter high-SAM minus low-SAM composites for the five climate states. (a, c, e, g, i) Zonal wind stress anomaly. (b, d, f, h, j) Sea ice concentration anomaly. Under the positive phase the westerlies strengthen over the circumpolar belt in every state, and the sea ice response is a dipole with a loss near the coast, where polynyas open and brine is rejected, and a gain further north where the exported ice melts. The dipole is displaced northward and is substantially stronger in the glacial states, consistent with the shift of the transformation response from the thermal to the haline pathway. \label{sam_windice}}
\end{figure}

\clearpage
"""

# insert everything at the start of the supplementary section
anchor = r'\section{Supplementary figures}\label{secA1}'
if s.count(anchor) != 1:
    raise SystemExit('supplementary anchor not found uniquely')

ins = (anchor + '\n' + NEW_FIGS + '\n' + blocks['sam_heat'] +
       '\n\n\\clearpage\n\n' + blocks['sam_fwf'] + '\n\n\\clearpage\n')
s = s.replace(anchor, ins, 1)

P.write_text(s)
print('moved sam_heat and sam_fwf to supplementary; added sam_pushpull and sam_windice')
