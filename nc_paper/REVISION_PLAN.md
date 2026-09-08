# NC Revision Plan — Shi et al., "Glacial–interglacial regime shift in Southern Ocean dense water formation"

Status: **analysis phase COMPLETE. Writing phase next.** Started 2026-09-08.

RESOLVED since the first pass:
- The submitted manuscript is `nc_paper/nc-main.tex` (Nature format, 5 authors incl. Lohmann),
  NOT `nc_paper/main.tex` (that is an older AGU-format version). Confirmed against submitted.pdf.
- Build pipeline works: `nc_paper/build/`, texlive module `texlive/live2025-gcc-13.3.0`
  (pdflatex, bibtex AND latexdiff all present). Baseline recompiles to 32 pages, 0 undefined citations.
- New results in `NEW_ANALYSIS_RESULTS.md`; verified literature in
  `LITERATURE_VERIFIED_BY_MAIN.md` and `reviewer_refs_verified.bib` (all DOIs confirmed).
- Three new figures built: figR1 (domain sensitivity), figR2 (MOC), figR3 (global age).
Inputs: `nc_paper/submitted.pdf` (31 pp), `nc_paper/comments.txt` (4 reviewers), live manuscript `nc_paper/main.tex` (420 lines).

Deliverables requested by user: (1) revised manuscript, final + diff/track-changed version; (2) point-by-point response letter;
(3) new figures as needed; (4) literature search where needed. Tone: earnest, every comment taken seriously.

---

## 1. The two make-or-break issues

The editor's own letter singles these out: "concerns regarding a mismatch between the model and observations
and the limitations of the model that must be addressed." Everything else is secondary.

### Issue A — PI is thermally dominated, Pellichero et al. (2018) say the modern ocean is haline dominated
Raised by R2 (major #1), R3 (main critique + L291-297, L302-305), R1 (model evaluation).
The submitted text disposes of this in one sentence (line ~305: "model biases or regional differences in
analysis domains"). All three reviewers independently call that inadequate. R2 goes further: comparing our
Fig 3a/4a with Pellichero Fig 2a/c the domains look similar, so it is *probably* model bias — and if the
PI/interglacial end-member is biased, the headline regime shift is in question.

Both R2 and R3 hand us the test to run, and R3 names two concrete options:
  - (i) recompute our WMT restricted to Pellichero's actual observational domain//density range and compare like with like;
  - (ii) apply the identical WMT machinery to a reanalysis (R3 suggests GLORYS,
        `GLOBAL_MULTIYEAR_PHY_001_030`) to produce an "observational" counterpart to the PI analysis.
Doing (i) is mandatory. (ii) is the stronger answer and is worth attempting if GLORYS surface flux fields can be obtained.

R3's deeper framing point must also be answered in the text, not just by a figure: today AABW forms on the
shelves (ice-shelf cavity melt → near-freezing shelf water → coastal polynya brine rejection → downslope
gravity plumes; Orsi 1999, Toggweiler 1995), not by open-ocean convection. The model does open-ocean
convection under PI. So R3 asks, pointedly, whether we are merely watching a shift *from a biased PI state
to a more realistic glacial state*. That question deserves a direct, honest answer in the Discussion, plus a
reframing of what we claim is generalisable. Note the model has no ice-shelf cavities in this configuration —
state that explicitly as a limitation.

### Issue B — Chen et al. 2025 (GRL 52, e2025GL114809), not cited
R2 major #2. Same ocean model, same glacial dense-water result reached with different tools, and Lohmann is
a co-author on both. This looks bad and must be fixed head-on: cite it, discuss it, and state plainly what the
WMT decomposition adds that Chen et al. did not already show. Needs a genuine answer, not a hand-wave.
Also uncited and requested: **Lhardy et al., Clim. Past 17, 1139–1159 (2021)** (most PMIP models fail to
get glacial SO sea ice / over-convect) and **Gray et al. 2023** (proxies suggest equatorward+weakened glacial
westerlies, which our model does *not* reproduce — R3 wants this discussed against our "SAM" mechanism).

---

## 2. Data feasibility — what was verified today

Working dir `/work/ba1066/a270064/cc_projects/aabw_5exps`. Experiments `{pi,mh,lig,lgm,mis}`.

**MOC streamfunction — AVAILABLE.** This unblocks R3's most substantive new-analysis request.
`/work/ba1066/a270064/production/{pi,mh,lig,lgm,mis}_age/diag_moc.nc` and `diag_gmoc.nc`,
all five experiments. Dims `MOC(time=50, deps=48, lats=85)`, plus `lats`, `deps`, `time`.
Caveat: `xr.open_dataset` fails with "dimension 'time' already exists as a scalar variable";
open with `decode_times=False` plus `drop_variables='time'`, or go through netCDF4 directly.
`diag_gmoc` = global, `diag_moc` = Atlantic (confirm which is which before plotting).

**3-D ideal age — AVAILABLE, global.** `{exp}/age.nc` on the native mesh (`nod2=126858, nz1=47`)
and `{exp}/age_reg.nc` already regridded to a global 1°×1° × 47-level grid (`lat=180, lon=360`).
R3 wants ideal age shown *globally* and at more than the single 4000 m level, so `age_reg.nc` is
exactly what is needed for basin-mean vertical profiles. Note the submitted manuscript already had
`age_vertical.pdf` / `age_vertical_anm.pdf` figures that were commented out of main.tex —
reinstating them addresses R3's L25 and L214 comments almost directly.

**No 3-D velocity fields locally**, so MOC cannot be recomputed from scratch; use the model's own
`diag_*moc.nc` diagnostics above. That is sufficient.

**T/S climatologies** `{exp}/temp.clim.nc`, `salt.clim.nc` — available, for stratification/water-mass diagnostics.

**WMT**: climatological `{exp}/wmt*.nc`; 100-yr monthly `{exp}/wmt_results/wmt_{region}_100years_{exp}.nc`
for 4 regions (southern_ocean, ross_sea, weddell_sea, adelie).

**SAM composites**: `composite_sam/` holds SAM indices, heat/FW/tendency/WMT composites for all 5 exps,
JJA and DJF. Threshold is 1.2σ (already reconciled across script/paper/CLAUDE.md in a previous session).
Beware there are TWO fwflux plotting scripts; the *active, correct* one is
`composite_sam/fw_data/plot_sam_fwflux_composite.py` (regular grid, square filled panels).
Run it with cwd inside `fw_data/`, then copy PDF+PNG to `figures/` using absolute paths.

**Meshes**: core2 (PI/MH/LIG), glac1d (LGM), glac1d_38k (MIS3), under `/home/a/a270064/bb1029/inputs/`.

**Gotcha carried over from earlier sessions**: run matplotlib/cartopy plotting scripts in the *foreground*.
A `run_in_background` Bash attempt previously produced an empty output file and stale timestamps.

---

## 3. Planned new/modified figures

Numbering to be finalised once the text is restructured.

1. **PI WMT vs Pellichero (2018) observational estimate**, like-for-like domain — the central rebuttal figure
   for Issue A. Answers R1#26 (Fig A4 shows only model, add observations), R2 major#1, R3 L302-305.
2. **Global MOC streamfunction, 5 panels** from `diag_gmoc.nc` — R3.
3. **Global ideal-age vertical sections / basin profiles** (reinstate + extend the commented-out figures) — R3 L25, L214.
4. **Model evaluation panel for PI**: SO sea ice extent/concentration, MLD, stratification vs observations
   (NSIDC sea ice, de Boyer Montégut or Argo MLD, WOA T/S) — R1 general "model evaluation", R2, R3.
5. **Boundary-condition summary table** (GHG, orbital params, ice sheet, land-sea mask, bathymetry) — R1 general + R1#32.
6. **SAM figures rescaled to a common buoyancy-flux unit** so heat and freshwater are directly comparable — R2 minor.
   R2 also suggests moving SAM Figs 5/6 to Supplementary and shortening that section; R3 by contrast finds the
   SAM result the *most* generalisable part. These pull in opposite directions — flag for the user's decision,
   suggested compromise is to keep SAM in the main text but tighten it, and answer R2 explicitly on why.
7. Add lat/lon labels to all map figures; add climate-state labels on top of every panel column (R1#17, R1 general, R3).
8. Fig 2 (WMT) with common y-axis limits, at least as a supplementary version (R1#15).
9. Check colour maps for colour-vision deficiency (editorial requirement) — avoid red/green pairs.

## 4. Editorial compliance items (non-negotiable for resubmission)
Data Availability section; Code Availability section; Source Data file; CVD-safe colours; ORCID for
corresponding author; code checklist upload. The Zenodo deposit was already assembled in a previous session
at `zenodo_deposit/` (~1.3 GB, 141 .nc + README) — the DOI needs to be inserted into the Data Availability
statement once the user uploads it.

## 5. Text-level comments
R1 gives 32 numbered items, R3 gives ~15 minor ones. Most are quick (typo "moer"→"more" at line ~125 of the
tex; figure-numbering order A1/A2/A3; capitalise LIG/LGM/MIS; spell out MH/LIG or define them on first use in
Results, since section 4 defines them too late; "across at high latitudes" phrasing; citation-format checks at
lines 291/294/388; add references at lines 209/339; per-subpanel references throughout). These should be worked
through one at a time against the line numbers in `submitted.pdf`, not against `main.tex` line numbers —
**the two differ**, so build a line-number map from the PDF first.

R1#11 notes Fig 1 i–l is never discussed. R1#18 notes an internal contradiction about the LIG
("modest changes" vs "pronounced increase"). R1#28 notes lines 323/326 repeat each other and the general claim
does not hold for the Ross/Weddell peak-density classes. These need real text fixes, not cosmetic ones.

## 6. Suggested execution order for the next session
1. Build the PDF-line-number → main.tex mapping.
2. Literature search: Chen 2025, Lhardy 2021, Gray 2023, Orsi 1999, Toggweiler 1995, Millet 2025,
   plus AWI-ESM2/FESOM2 SO evaluation papers for the model-performance discussion.
3. Run the like-for-like Pellichero-domain WMT comparison (Issue A). This gates the framing of the whole revision.
4. MOC + global age figures.
5. Rewrite Discussion around Issues A and B; reframe what is claimed to be generalisable.
6. Sweep all 47 minor comments.
7. Produce final tex, latexdiff track-changed tex/PDF, and the response letter.

## 7. Open questions for the user
- SAM section: R2 wants it shortened and moved to SI, R3 finds it the most valuable part. Which way?
- Is a GLORYS reanalysis WMT calculation in scope, or is the restricted-domain comparison enough?
- Has the Zenodo upload happened, i.e. is there a DOI to insert?
- Author list on the submitted PDF vs main.tex should be confirmed (main.tex lists Shi, Liu, Yang, Yang;
  earlier project notes mention Shi, Lohmann, Stepanek — and R2's remark about Lohmann being a co-author
  on Chen et al. 2025 implies Lohmann is an author here. Verify against submitted.pdf before writing the letter.)
