#!/usr/bin/env python
"""Second pass: remaining minor reviewer fixes applied to build/revised.tex."""
import sys
from pathlib import Path

P = Path('build/revised.tex')
s = P.read_text()
edits = []


def E(old, new, label):
    edits.append((old, new, label))


# R1: typo "spaning"
E("we perform analysis on five simulated climate states spaning PI, MH, LIG, LGM, and MIS3",
  "we analyse five simulated climate states spanning PI, MH, LIG, LGM, and MIS3",
  'typo-spaning')

# R1#28: lines 323 and 326 repeat each other, and the general claim does not hold
# equally in every sector for the peak-density classes.
E(r"""SAM emerges as a persistent modulator of ventilation across all climate states. Our simulations indicate that the positive SAM phase systematically enhances dense water formation through enhancing turbulent heat loss over open waters while simultaneously driving sea ice divergence that triggers brine rejection at coastal regions. Our composite analysis reveals that positive SAM phases systematically enhance transformation, with maximum anomalies reaching 3-8 Sv between high and low SAM years. The dominant forcing pathway shifts from thermal during interglacials to haline during glacials, mirroring the mean-state regime shift described above.""",
  r"""The Southern Annular Mode emerges as a persistent modulator of transformation across all climate states. In its positive phase it enhances turbulent heat loss over open water and simultaneously drives sea ice divergence that promotes brine rejection near the coast, with composite anomalies of 3--8 Sv between high and low phases. The dominant pathway shifts from thermal during interglacials to haline during glacials, mirroring the mean-state regime shift described above. This circumpolar picture does not hold uniformly in every sector. The Ross Sea in particular departs from it, showing weaker and at some densities opposing responses, and in the glacial states the circumpolar signal is carried largely by the Weddell sector while the Ross and Ad\'elie sectors contribute little.""",
  'sam-repetition')

# R1#29 and R3: the sediment-record proposal was stated too confidently.
E(r"""The persistent SAM-AABW relationship across glacial and interglacial periods found in our study implies potential for paleoclimate proxy development. Given that SAM variability influences formation rates by 15-20\% on multi-year timescales, and formation changes propagate to deep ocean tracer signatures, high-resolution sediment core records from AABW formation regions might preserve SAM signals. Advances in chronological methods and high-accumulation site analysis may eventually enable marine-based SAM reconstructions to complement existing ice-core records.""",
  r"""Whether this persistent coupling could be exploited for proxy reconstruction is less clear than we previously suggested. Recovering an interannual-to-decadal signal from marine sediments would require accumulation rates and chronological precision that are rarely available around the Antarctic margin, where sedimentation is slow and bioturbation mixes the record over intervals far longer than the variability of interest. We therefore present the persistence of the coupling as a property of the simulated system rather than as a practical route to a marine Southern Annular Mode reconstruction.""",
  'sediment-caveat')

# R3 L347: it is the carbon in the water that is dated, not the water
E(r"""Benthic-planktonic $^{14}$C age offsets reveal that glacial deep ocean radiocarbon ages increased globally, with strongest signals in the Pacific and Southern Ocean \cite{Skinner2010,Skinner2017}.""",
  r"""Benthic-planktonic $^{14}$C age offsets indicate that the radiocarbon age of dissolved inorganic carbon in the glacial deep ocean increased globally, with the strongest signals in the Pacific and Southern Ocean \cite{Skinner2010,Skinner2017}.""",
  'radiocarbon-carbon')

# R1#22 second instance of "across at high latitudes"
E(r"""positive SAM phases systematically enhance ocean heat loss by more than 15~W/m$^2$ across the high-latitude ice-free  Southern Ocean where open water permits direct atmosphere-ocean coupling""",
  r"""positive SAM phases systematically enhance ocean heat loss by more than 15~W/m$^2$ over the ice-free parts of the Southern Ocean south of about 50°S, where open water permits direct atmosphere-ocean coupling""",
  'across-at-2')

# R3: "in the model" framing for the convection statement in Results 2.1
E(r"""deep mixed layers ($>$400 m) are observed in the Weddell and Ross Sea gyres (Fig. \ref{mld}a-c), indicating active open-ocean convection.""",
  r"""the model produces deep mixed layers ($>$400 m) in the Weddell and Ross Sea gyres (Fig. \ref{mld}a-c), indicating active open-ocean convection.""",
  'in-the-model')

# R1#30: spell out abbreviated components in Methods
E(r"""Here we further decompose the heat fluxes into shortwave radiation, longwave radiation, latent heat, and sensible heat fluxes.""",
  r"""Here we further decompose the net heat flux into its shortwave radiative, longwave radiative, latent heat, and sensible heat components. In the figures the radiative component denotes the sum of the shortwave and longwave fluxes, and the turbulent component denotes the sum of the latent and sensible heat fluxes.""",
  'methods-components')

# R1#19: explain thermal and sea-ice-driven density tendency in Methods
E(r"""After xbudget decomposes surface density tendency into individual heat and salt components, these decomposed budgets are passed directly to xwmt (see the next subsection) for WMT calculation.""",
  r"""We refer throughout to the thermal density tendency, meaning the surface density change produced by the net heat flux acting through the thermal expansion coefficient, that is the $-\left(\alpha/C_p\right)Q_{net}$ term, and to the sea-ice-driven density tendency, meaning the part of the haline term $\beta S F_{fw}$ that is carried by the thermodynamic growth and melt of sea ice. A positive value denotes densification of the surface water in both cases. Brine released during ice growth increases surface salinity and therefore density, whereas melting releases freshwater and reduces it.

After xbudget decomposes surface density tendency into individual heat and salt components, these decomposed budgets are passed directly to xwmt (see the next subsection) for WMT calculation.""",
  'methods-tendency-defn')

# R1#20: reference for the statement about turbulent flux dominance
E(r"""This thermal densification is overwhelmingly controlled by turbulent fluxes (sensible and latent heat), with radiative fluxes playing a minor role in the buoyancy budget (Fig. \ref{density_heat}f-j vs. Fig. \ref{density_heat}k-o).""",
  r"""This thermal densification is overwhelmingly controlled by turbulent fluxes (sensible and latent heat), with radiative fluxes playing a minor role in the buoyancy budget (Fig. \ref{density_heat}f-j vs. Fig. \ref{density_heat}k-o), consistent with observationally based surface buoyancy budgets for the Southern Ocean \cite{Cerovecki2011,Abernathey2016}.""",
  'turbulent-ref')

missing = []
for old, new, label in edits:
    n = s.count(old)
    if n != 1:
        missing.append((label, n))
        continue
    s = s.replace(old, new, 1)

if missing:
    print('FAILED:')
    for label, n in missing:
        print(f'  {label}: found {n}')
    sys.exit(1)

P.write_text(s)
print(f'applied {len(edits)} minor edits')
