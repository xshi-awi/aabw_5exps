"""EE2 and EE3 (2026-11-22) on revised.tex. Every anchor must match exactly once.

EE2: the freshwater density term was 1000 times too small (rho_w dropped in the plotting
script); units are stated as kg m-3 per metre of water column. EE3: Figs. A3/A4 (SAM heat
and freshwater composites) were made at +/-1.0 sigma, so their captions and the threshold
sentence say so; the figures are not redrawn.
"""
import sys

p = sys.argv[1]
s = open(p).read()


def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, (n, old[:100])
    s = s.replace(old, new)


# ---- EE2: unit stated once where the quantity is introduced in the Results
rep(r"""we convert heat and freshwater fluxes into surface density tendency during austral winter (see Methods for more details).""",
    r"""we convert heat and freshwater fluxes into surface density tendency during austral winter. We express it as the density change of a surface layer 1 m thick, in kg~m$^{-3}$~s$^{-1}$ per metre of water column (see Methods for more details).""")

# ---- EE2: the freshwater numbers (line 149), recomputed with rho_w restored
rep(r"""Total freshwater density tendency anomalies reach $+5 \times 10^{-9}$~kg~m$^{-3}$~s$^{-1}$ in a circumpolar band around Antarctica. The sea ice component (Fig.~5i,j) dominates this signal, with strongest anomalies in the coastal regions ($> 5 \times 10^{-9}$~kg~m$^{-3}$~s$^{-1}$).""",
    r"""Total freshwater density tendency anomalies reach about $+2.5 \times 10^{-6}$~kg~m$^{-3}$~s$^{-1}$ per metre of water column in a circumpolar band between 55 and 70°S. The sea ice component (Fig. \ref{density_fwf}i,j) dominates this signal, with the strongest anomalies along the coast, where they exceed $4 \times 10^{-6}$~kg~m$^{-3}$~s$^{-1}$ over a tenth of the area south of 65°S and locally reach $1.8 \times 10^{-5}$~kg~m$^{-3}$~s$^{-1}$.""")

# ---- EE2: figure captions match the colour bars (per month, per metre of water column)
rep(r"""columns 2-5 show anomalies in MH, LIG, LGM and MIS3 relative to PI, respectively.  Units: kg m$^{-3}$ s$^{-1}$. \label{density_heat}}""",
    r"""columns 2-5 show anomalies in MH, LIG, LGM and MIS3 relative to PI, respectively.  Units: kg m$^{-3}$ month$^{-1}$ per metre of water column, that is, the density change of a surface layer 1 m thick. \label{density_heat}}""")
rep(r"""columns 2-5 show anomalies in MH, LIG, LGM and MIS3 relative to PI, respectively. Units: kg m$^{-3}$ s$^{-1}$. \label{density_fwf}}""",
    r"""columns 2-5 show anomalies in MH, LIG, LGM and MIS3 relative to PI, respectively. Units: kg m$^{-3}$ month$^{-1}$ per metre of water column, that is, the density change of a surface layer 1 m thick. \label{density_fwf}}""")

# ---- EE2: the unit in the Methods definition
rep(r"""The total surface density tendency, $\mathcal{B}$ (m$^2$ s$^{-3}$), is decomposed""",
    r"""The total surface density tendency, $\mathcal{B}$ (kg m$^{-2}$ s$^{-1}$, which divided by 1 m gives the density change of the top metre of the water column), is decomposed""")
rep(r"""Similarly, the freshwater fluxes $F_{fw}$ are partitioned""",
    r"""Similarly, the freshwater fluxes $F_{fw}$, taken as mass fluxes in kg m$^{-2}$ s$^{-1}$, are partitioned""")

# ---- EE3: Figs. A3 and A4 are 1.0 sigma composites
rep(r"""(defined as JJA-mean SAM index exceeding $\pm$1.2$\sigma$ from the climatological mean)""",
    r"""(defined as JJA-mean SAM index exceeding $\pm$1.2$\sigma$ from the climatological mean; the heat and freshwater flux composites in Fig. \ref{sam_heat} and Fig. \ref{sam_fwf} use $\pm$1.0$\sigma$)""")
rep(r"""(k-o) turbulent component (sensible + latent heat) for high-SAM minus low-SAM composites (1.2$\sigma$ threshold)""",
    r"""(k-o) turbulent component (sensible + latent heat) for high-SAM minus low-SAM composites (1.0$\sigma$ threshold)""")
rep(r"""(k-o) other sources (net precipitation + runoff) for high-SAM minus low-SAM composites (1.2$\sigma$ threshold)""",
    r"""(k-o) other sources (net precipitation + runoff) for high-SAM minus low-SAM composites (1.0$\sigma$ threshold)""")

open(p, 'w').write(s)
print('ok')
