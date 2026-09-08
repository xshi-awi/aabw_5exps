#!/usr/bin/env python
"""Insert the new figure environments into build/revised.tex."""
from pathlib import Path
import sys

P = Path('build/revised.tex')
s = P.read_text()

ANCHOR = r"""\caption{(a-e) Simulated spatial distribution of Southern Ocean ventilation age at 4000 m depth for (a) PI, (b) MH, (c) LIG, (d) LGM,  (e) MIS3.  (f-i) Simulated anomalies of ventilation age at 4000 m for (f) MH minus PI, (g) LIG minus PI, (h) LGM minus PI, and (i) MIS3 minus PI. Units: year. \label{age}}
\end{figure}"""

NEW = ANCHOR + r"""

\begin{figure}
\centering
\includegraphics[width=\textwidth]{figures/figR2_moc_5exps.pdf}
\caption{Global meridional overturning streamfunction, averaged over the final 20 years of the online model diagnostic. (a-e) Absolute streamfunction for PI, MH, LIG, LGM and MIS3. Positive values (red) denote clockwise circulation in the latitude-depth plane; the negative (blue) cell below about 2500 m represents the northward spreading of southern-sourced bottom water. The strength of that abyssal cell, defined as the streamfunction minimum below 2000 m and south of 30°S, is given in each panel. (f-i) Anomalies relative to PI for MH, LIG, LGM and MIS3. Units: Sv. \label{moc}}
\end{figure}

\begin{figure}
\centering
\includegraphics[width=\textwidth]{figures/figR3_age_global.pdf}
\caption{Global ideal age. (a-d) Basin-mean vertical profiles for the Atlantic, Indian, Pacific and Southern Ocean, for the five climate states. (e) Basin-mean ideal age below 2000 m. (f-j) Zonal-mean ideal age sections for PI, MH, LIG, LGM and MIS3. Basins are defined as Atlantic 70°W-20°E and 35°S-65°N, Indian 20°E-115°E and 35°S-25°N, Pacific 120°E-70°W and 35°S-65°N, and Southern Ocean south of 35°S. Units: years. \label{age_global}}
\end{figure}

\begin{figure}
\centering
\includegraphics[width=\textwidth]{figures/figR1_wmt_domain_sensitivity.pdf}
\caption{Sensitivity of the thermal and haline partition to the integration domain, shown as annual-mean transformation rates for the five climate states. (a-e) Integrated south of 60°S, the domain used throughout this study. (f-j) Integrated over the seasonal sea ice zone only, defined as the region where the climatological September sea ice concentration exceeds 15\%, which reproduces the domain of Pellichero et al. \cite{Pellichero2018} to within 2.5\% in area. The share of the total transformation carried by the thermal term at the transformation maximum, and the integrated area of each domain, are given in each panel. Restricting the integration to the ice-covered sector increases the haline share, as expected, but leaves the contrast between thermally dominated interglacials and haline dominated glacials intact. Units: Sv. \label{domain}}
\end{figure}"""

if s.count(ANCHOR) != 1:
    print(f'anchor found {s.count(ANCHOR)} times', file=sys.stderr)
    sys.exit(1)

s = s.replace(ANCHOR, NEW, 1)
P.write_text(s)
print('inserted 3 figure environments')
