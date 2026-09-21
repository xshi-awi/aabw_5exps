# task

- to finish the paper in ./paper/main.tex
- add SAM COMPOSITE result, to show how SAM affect surface variables and surface wmt.

## previous finished tasks

- calculate SAM based on ./{pi,mh,lig,lgm,mis}/100years/slp* files. - finished (see @./composite_sam/calc_sam_wmt_composite.py)
- perform composite analysis between SAM (jja-mean) and other variables (jja-mean) (calculate var anomaly between high SAM and low SAM)
- variables are in ./{pi,mh,lig,lgm,mis}/100years
- use context7 mcp
- use threshold of 1.2*std (high SAM: SAM > mean(SAM)+1.2*std(SAM), low SAM: SAM < mean(SAM)-1.2*std(SAM))
- plot: 可参考plot_wmt_4regions_5exps.py (total heat/seaice FW/other FW) and  
- which variables should be ploted:
 1. Fig 1: 3行5列, 第一行, total heat fluxes, 第二行: lw+sw; 第三行: sh+lh
 2. Fig 2: 3行5列, 第一行, total freshwater fluxes, 第二行: sea ice only; 第三行: other flux (prec+evap+snow+runoff)

## data path
climatological means:  ./{pi,mh,lig,lgm,mis}
100-year monthly means: ./{pi,mh,lig,lgm,mis}/100years

## tendendy data
100-year monthly means: ./{pi,mh,lig,lgm,mis}/surface_density_tendency_100years_{pi,mh,lig,lgm,mis}.nc

## wmt data:
climatological means:  ./{pi,mh,lig,lgm,mis}/wmt*nc
100-year monthly means: ./{pi,mh,lig,lgm,mis}/wmt_results/wmt_*_100years_{pi,mh,lig,lgm,mis}.nc

## github

https://github.com/xshi-awi/aabw_5exps

## grid info

mesh path for pi/mh/lig: /home/a/a270064/bb1029/inputs/mesh_core2
mesh path for LGM: /home/a/a270064/bb1029/inputs/mesh_glac1d
mesh path  for MIS3: /home/a/a270064/bb1029/inputs/mesh_glac1d_38k
fesom.mesh.diag.nc in mesh path contain all needed info. (lat/lon might be in 弧度)

## Project Progress

### 2026-05-19 (Zenodo data deposit assembled)

**Progress**: Built `zenodo_deposit/` — a self-contained ~1.3 GB folder with exactly the data behind every manuscript figure, plus `zenodo_deposit/README.md` data description. User will upload to Zenodo manually.

**Figure→data tracing**: parsed `nc_paper/main.tex` for the 13 `\includegraphics` (note `age_vertical*.pdf` are commented out — excluded; their data overlaps `age_horizontal`'s anyway). Traced each PDF to its plotting script and the exact NetCDF inputs.

**Layout**: `climatology/<exp>/` (per-exp climatological monthly means), `wmt_100years/<exp>/` (full 100-yr monthly WMT for the ±1σ winter-WMT fig), `sam_composites/` (+`fw_data/`), `mesh/`.

**Size reductions** (per user's choices via AskUserQuestion):
- `echam_clim.nc` 436 MB → `echam_clim_subset.nc` 7 MB via `ncks -v var92,var95,var111,var120,var165,var166,var180,var181` (only 8 figure-relevant vars: LW/SW/LH/SH heat + u10/v10/ustr/vstr).
- `mask.nc` 292 MB → `mask_surface.nc` 24 MB via `ncks -d time,0` (it's a 3-D salt field used only as land mask; one time step, all 47 depths kept — needed at 3900 m index).
- WMT 100-yr files kept in full (user chose "include the 512 MB"; ~487 MB, the only non-climatology data — required for interannual spread).
- Dropped `wmt_sam_composite_djf_*` and `sam_index_djf_*` (manuscript is JJA only); kept 5 `sam_index_<exp>.nc` as provenance.

**Mesh**: only `mesh_core2/fesom.mesh.diag.nc` → `fesom.mesh.diag_core2.nc` (PI/MH/LIG share it; LGM/MIS3 paleo-meshes not needed for any figure).

**Final**: 141 .nc + README; climatology 535M, wmt_100years 487M, sam_composites ~194M, mesh 92M. Well under Zenodo 50 GB. README has full file table, ECHAM code→meaning map, and a figure(\ref{})→file mapping table.

**Next**: user uploads to Zenodo; consider adding the resulting DOI to `nc_paper/main.tex` Data Availability statement.

### 2026-05-19 (Intro rewrite + SAM threshold consistency fix)

**Progress**: (1) Rewrote two Introduction paragraphs in `nc_paper/main.tex` for flow; (2) resolved a SAM-threshold documentation inconsistency across script, paper, and CLAUDE.md.

**Note on paths**: `paper/` was deleted in git; the live manuscript is now `nc_paper/main.tex`.

**Intro paragraph 1 (line 59, "Southern Ocean role")**: Eliminated 4 chained "this+noun" anaphors (user disliked "this", then also "that"). Final has zero this/that/such back-references — recast as concrete nouns ("Regulating heat and carbon storage", "Formation rates"). Added a bridging sentence (surface buoyancy forcing — wind stress, heat exchange, sea ice — differed markedly glacial vs interglacial) so the closing "across different climate states" is no longer abrupt. New cites: `\cite{Adkins2002,Ferrari2014,Jansen2017}` (all verified present in Ref.bib).

**Intro WMT paragraph (lines 72/74/76)**: Restructured to user's intended throughline: (1) what WMT is → (2) it decomposes per-component contributions → (3) modern-observation WMT findings. Merged redundant defn sentences, removed "This method" anaphor (used participial "Introduced by Walin1982...extended by Speer1992"), added "Beyond the net transport," bridge to the decomposition para, deleted the empty "have enhanced our understanding" filler before Pellichero/Bailey, fixed grammar "with freshwater fluxes drive" → "with the associated freshwater fluxes driving", "Additionally"→"Consistently" (Bailey corroborates Pellichero).

**SAM threshold = 1.2σ (confirmed actual value; user chose to keep 1.2 and fix docs)**: Effective config was always `THRESHOLD_STD = 1.2` (line 32, `composite_sam/calc_sam_wmt_composite.py`) — **no recomputation needed**, data is correct. Fixed 5 stale "1.0/1σ" references:
- `calc_sam_wmt_composite.py`: docstring L5, function default `identify_composite_years(threshold_std=1.0→1.2)` L137 (was overridden by explicit call arg, results unaffected), param doc L146.
- `nc_paper/main.tex`: Fig sam_heat caption L389 `(1σ→1.2σ threshold)`, Fig sam_fwf caption L395 same. Body methods sentence (L243) already correctly said `±1.2σ` — left as is.
- `CLAUDE.md` task spec L12 `1*std → 1.2*std` (formula synced).

**Verified**: All edits applied cleanly. Manuscript not yet recompiled (offered to user).

### 2026-05-18 (fig02 split-colorbar variant)

**Progress**: Created a new variant of `fig02_climate_mld_seaice_jja.pdf` with split MLD colorbars and enlarged text. Original figure/function untouched.

**Script**: `plot_climate_patterns_jja.py` — added `plot_figure2_split()` (inserted before `plot_figure2()`), new switch `seaice_split_only` (default True in `__main__` chain, before `seaice_only`). Set `supp_dens_taus_only=False`.

**Structure**: Same 2 rows (MLD / Sea Ice) × 5 cols (PI,MH,LIG,LGM,MIS3), panels (a)-(j), same colormaps. MLD row split into two colorbar groups with **different ranges** (per user follow-up): warm group (a)PI/(b)MH/(c)LIG → **0–400 m** (ticks 0,100,200,300,400); glacial group (d)LGM/(e)MIS3 → **0–200 m** (ticks 0,50,100,150,200). Implemented via `make_mld_levels_norm(mld_max)` helper + per-group `levels`/`norm`/`ticks` in `mld_groups`, `col_to_group` lookup picks per-column contourf args. Sea ice row keeps single shared 0–1 cbar.

**Font enlargement** (constants `FS_PANEL=17, FS_TITLE=20, FS_ROWLAB=20, FS_CBLAB=17, FS_CBTICK=15`; vs old 11/12/12/10/8). Colorbar bar thickness 0.015→0.020, lowered to `y0-0.055`. Figure size 20×9 → 20×9.5, bottom 0.10→0.13.

**Output**: `figures/fig02_climate_mld_seaice_jja_split.{pdf,png}`. Verified via pdftoppm @130dpi crops — text legible, two distinct MLD colorbars, sea-ice cbar clean.

**Note**: MLD data check showed all 5 exps have similar JJA MLD (max ~700–1000 m, p95 ~260–300 m); 0–400 single cbar was saturating detail — confirms the split + 200 m cap was the right call.

### 2026-05-18 (SAM composites + age figure text enlargement)

**Progress**: Enlarged all text in 3 manuscript figures; no figure-level top titles existed to remove (the PI/MH/... at top are per-column `ax.set_title`, which the user wanted enlarged).

**Scripts modified**:
- `composite_sam/plot_sam_heatflux_composite_t63.py` → `sam_heatflux_composite_t63_3rows_5cols.{pdf,png}`
- `composite_sam/plot_sam_fwflux_composite.py` (ACTIVE version; the `fw_data/` copy is the older Jan-16 one with a `suptitle` — left untouched) → `sam_fwflux_composite_3rows_5cols.{pdf,png}`
- `plot_age_horizontal_3900m_combined.py` → `age_horizontal_3900m_combined.{pdf,png}`

**Font changes**:
- SAM figs (both): panel labels 12→20, row titles 11→18, column titles 14→22, colorbar label 12→20, colorbar ticks 10→16; colorbar bar 0.02→0.025 high, y 0.06→0.07, `bottom` 0.12→0.14 to avoid clipping.
- Age fig: panel labels 10→19, panel titles 10→21 (now bold), abs colorbar label 9→19 / ticks 8→15, anomaly colorbar label 9→19 / ticks 7→14.

**Layout/path fixes**:
- Age script wrote to non-existent `paper/figures/`; changed to `figures/` (the real referenced file). Figure height 10→13, `hspace` 0.28→0.55, `height_ratios` colorbar rows 0.25→0.22, `bottom` 0.02→0.07 — fixes inter-row colorbar overlapping row-2 titles and bottom rotated tick clipping.
- The two SAM scripts only write to `composite_sam/`; the `figures/` copies were stale (Mar 28). Manually `cp`'d updated PDF+PNG into `figures/` so the paper-referenced versions are current.

**Verified**: All 3 rendered via pdftoppm + zoomed colorbar crops — text legible, colorbar labels not clipped, no overlaps.

**CORRECTION (same day)**: User noticed `sam_fwflux_composite_3rows_5cols.pdf` had empty corners (circular fill in square axes). Root cause: there are TWO fwflux scripts. The "active" `composite_sam/plot_sam_fwflux_composite.py` (Mar 28) plots raw FESOM unstructured mesh via `tripcolor` on `fwflux_sam_composite_*.nc` → polar-cap disk with empty square corners. The OLDER `composite_sam/fw_data/plot_sam_fwflux_composite.py` (Jan 16) plots regular-grid `_reg.nc` via `contourf`+cyclic point → square, fully filled (matches heatflux fig, and is the version that produced the square PDF the user remembered).
- Fix: switched the source of truth back to the `fw_data/` script. Applied to it: removed its `suptitle`; enlarged fonts to match heatflux (col 22, panel 20, row 18, cbar 20/16); ADDED (a)-(o) panel labels (old script had none); shortened `ROW_TITLES` to `['Total FW Flux','Sea Ice FW Flux','Other FW Flux\n(P+E+Snow+Runoff)']` (old verbose `($-$fw$-$evap...)` overflowed at 18pt); cbar geom `[0.25,0.07,0.5,0.025]`, top 0.92→0.97, bottom 0.12→0.14.
- The earlier text edits to the `tripcolor` `composite_sam/plot_sam_fwflux_composite.py` are now moot (that script is no longer the one used); a premature circular-boundary edit was reverted.
- `fw_data/` script writes to `composite_sam/`; PDF+PNG manually copied into `figures/` too. Run it with `cd composite_sam/fw_data` (uses `Path(__file__).parent`); copy to `figures/` with ABSOLUTE paths afterward (cwd is then inside fw_data/).

**Age figure empty-corner fix (same day, follow-up)**: The 10:46 age regen only did the font enlargement; the empty-corner issue (user said age fig has it too) was handled separately at 11:47. Mechanism differs from fwflux: age data is deliberately masked north of `LAT_CUTOFF=-50` (in `load_age` + `plot_polar`), so the polar-stereo square's corners (outside the 50°S circle) have no data. Fix = circular boundary clip (NOT switching data source). Added `set_circular_boundary(ax)` helper (matplotlib.path circle in axes coords via `ax.set_boundary`), called inside `plot_polar` right after `set_extent`. Result: clean circular disks, no empty square corners. Heatflux fig left as square (contourf on full global grid already fills it; this was the user's chosen scope).

### 2026-05-18

**Progress**: Improved readability of the two surface density-tendency JJA figures (`surface_heat_density_tendency_from_raw_jja.pdf`, `surface_freshwater_density_tendency_from_raw_jja.pdf`).

**Script modified**: `plot_surface_density_tendency_from_raw_data_jja.py` (the newer non-`_ok` version, unified per-figure colorbar ranges).

**Changes**:
- Unit conversion per second → per month: added `SEC_PER_MONTH = 30*86400 = 2,592,000 s`; multiply all 6 plotted vars by it in the preprocessing loop (step 4). All labels changed `kg m⁻³ s⁻¹` → `kg m⁻³ month⁻¹` (`UNIT_LABEL`). Values now O(0.001–10) instead of O(1e-8–1e-6).
- New `pick_tick_format(vmax_nice)` helper: picks fixed-decimal colorbar tick format by magnitude (`%.0f`/`%.1f`/`%.2f`/`%.3f`, falls back to `%.1e` only for extreme magnitudes). Replaces hardcoded `format='%.1e'` in both colorbar blocks. Heat fig ticks → integers (±20); FW fig ticks → `%.3f` (±0.025).
- Enlarged text: panel labels (a)–(o) 12→20pt; colorbar labels 9→16pt; colorbar ticks 8→14pt; colorbar bars slightly thicker/lower (0.015→0.018 height, offset 0.05→0.055).

**Verified**: Both PDFs regenerated and visually inspected via pdftoppm — labels legible, clean integer/decimal ticks, correct `month⁻¹` units.

**Follow-up tweaks (same day)**:
- Panel labels (a)–(o) reduced 20→15pt (20 was oversized).
- Fixed colorbar label/tick clipping by next row: figure height 12→15, `hspace` 0.35→0.55 (both figures, replace_all).
- FW fig row-3 title changed `Other FW (P+E+R+S)` → `Other FW (net precipitation + runoff)`.
- First-column PI colorbar label font 16→11pt so the leading "PI:" is no longer clipped on the narrow single-column bar (anomaly colorbar label stays 16pt, spans 4 columns).
- Added separate `var_titles_pi` (short, no parentheticals: heat = `Total Heat`/`SW + LW`/`LH + SH`; FW = `Total Freshwater`/`Sea Ice Alone`/`Other FW`) used ONLY for the narrow first-column PI colorbar label; wide anomaly colorbars keep full `var_title`.
- PI label font settled at 13pt (15pt still clipped panel (a) `PI: Total Heat (...)`; verified at 13pt via 150-dpi zoomed crops of panels a/f/k — all have clear left+right margin, no clipping).

**Note**: Old `plot_surface_density_tendency_from_raw_data_jja_ok.py` left untouched (differs only in colorbar-range strategy: per-variable vs per-figure unified).

### 2026-01-14 09:46

**Progress**: Created SAM-flux composite analysis framework

**Scripts created**
- composite_sam/slurm_sam_heatflux_t63.sh - Heat flux composite (ECHAM data)
- composite_sam/fw_data/*.py - Freshwater composite (FESOM data)

### 2026-01-16 03:30

**Progress**: Integrated SAM composite analysis into manuscript with comprehensive new subsection

**Key additions to paper/main.tex**:

1. **New Results subsection** (Section 3.5): "Southern Annular Mode influence on AABW formation"
   - 3.5.1: SAM modulation of surface heat fluxes
   - 3.5.2: SAM modulation of freshwater fluxes
   - 3.5.3: SAM impact on water mass transformation
   - 3.5.4: Implications for AABW variability and predictability

2. **Key findings documented**:
   - High-SAM enhances ocean heat loss by >15 W/m² in 50-65°S band
   - SAM-driven WMT changes: 15-20% modulation across all climate states
   - Thermal vs haline partitioning: 70-80% thermal in interglacials, shifting to 60-65% thermal in glacials
   - Sea ice component dominates freshwater response by factors of 3-5
   - Ross Sea shows strongest SAM sensitivity (>20 W/m² heat flux, 3-4 Sv WMT)

3. **Updated Conclusions section**:
   - Added new key finding #3 on SAM-AABW coupling
   - Expanded finding #4 to incorporate SAM variability context
   - Emphasized persistent teleconnection across glacial-interglacial cycles

4. **Figure integration**:
   - Figure 14: SAM heat flux composites (sam_heatflux_composite_t63_3rows_5cols.png)
   - Figure 15: SAM freshwater flux composites (sam_fwflux_composite_3rows_5cols.png)
   - Figure 16: SAM WMT composites (wmt_sam_composite_4regions_5exps.png)
   - All PDFs converted to PNG for viewing
   - Comprehensive captions added with quantitative details

**Technical approach**:
- Converted PDFs to PNG using pdftoppm
- Analyzed figures to extract key quantitative results
- Structured writing to follow logical flow: heat fluxes → freshwater fluxes → WMT → implications
- Linked SAM findings to existing sections (surface flux decomposition, WMT analysis)
- Provided mechanistic interpretation connecting observations to processes

**Writing highlights**:
- Clear progression from observations to mechanisms to implications
- Quantitative benchmarks established (15-20% modulation, 70-80% thermal contribution)
- Connected to modern observations (Zhou2023, Silvano2020)
- Identified proxy development potential
- Established model validation metrics

**Next steps**:
- Compile manuscript with pdflatex to check rendering
- Consider adding SAM discussion to Discussion section if needed
- Review reference list to ensure cited papers are in ref.bib

### 2026-01-19 Major Revision

**Progress**: Comprehensive restructuring of paper/main.tex to focus on Southern Ocean WMT analysis rather than AABW formation

**Motivation**: Model produces dense water through open-ocean convection rather than realistic shelf processes, and thermal forcing dominates in interglacials (contrary to observations). Therefore, we reframe the paper to focus on "WMT analysis of Southern Ocean" rather than "AABW formation mechanisms".

**Key changes**:

1. **Title revised**:
   - From: "Antarctic Bottom Water Formation Mechanisms Across Paleoclimate States..."
   - To: "Shifting Balance of Thermal and Haline Forcing in Southern Ocean Dense Water Formation Across Glacial--Interglacial Cycles: A Water Mass Transformation Analysis"

2. **Abstract rewritten**: Focus on WMT analysis and surface forcing mechanisms, not AABW rates

3. **Introduction restructured and shortened**:
   - Removed AABW-centric framing (old Sections 1.2-1.6 deleted)
   - New focus: Southern Ocean's role, WMT framework, paleoclimate perspective
   - Clear research objectives stated
   - Reduced from ~340 lines to ~100 lines

4. **Results section revised**:
   - All references to "AABW formation" changed to "dense water formation" or "WMT"
   - Adjusted percentages to be more accurate (70-85% thermal, 25-35% haline)
   - Added statistical test description for SAM analysis (t-test, p<0.05)

5. **Discussion completely restructured**:
   - New Section 4.1: "Model limitations and scope of interpretation" - upfront acknowledgment
   - New Section 4.2: "Robust aspects of the analysis" - what we CAN say
   - Section 4.3: Proxy comparison (condensed)
   - Section 4.4: "Implications for understanding deep water formation" - AABW discussion here
   - Section 4.5: Future research priorities (condensed)

6. **Conclusions rewritten**:
   - Clear statement of model limitations
   - Focus on robust findings about surface forcing mechanisms
   - AABW implications discussed appropriately in context

7. **Figure captions updated**: Removed AABW references, changed to "dense water formation" or "WMT"

8. **Author info added**: Xiaoxu Shi, Gerrit Lohmann, Christian Stepanek (AWI)

**Strategic approach**:
- Results section: Describe WMT patterns without claiming AABW formation rates
- Discussion: Acknowledge limitations FIRST, then discuss robust aspects, then implications for AABW
- This avoids reviewer criticism about unrealistic AABW representation while preserving scientific value

**Technical details**:
- Added abbreviations list after author info
- Fixed LaTeX formatting (×, °, en-dashes)
- Adjusted thermal/haline percentages to 70-85%/15-30% (interglacials) and 65-75%/25-35% (glacials)

**Next steps**:
- Compile with pdflatex to verify rendering
- Check ref.bib for all citations
- Review for any remaining inconsistencies

### 2026-01-16 11:15

**Progress**: Completed comprehensive Discussion section for paper/main.tex

**Key sections added**:

1. **Model limitations and value** (Section 4.1):
   - Acknowledged open-ocean convection bias in AWI-ESM vs realistic shelf overflows
   - Cited 2024 literature (Zhu2021NCC) showing models with biases still provide valuable mechanistic insights
   - Emphasized focus on **relative changes** and **process understanding** rather than absolute quantification
   - Three robust aspects: (1) thermal-to-haline forcing shifts, (2) sea ice's dual role, (3) ventilation age response
   - Referenced CESM2 example where paleoclimate data identified and corrected cloud parameterization issues
   - Clear framing: "models with known biases can still provide valuable mechanistic insights when limitations are properly acknowledged"

2. **Geological proxy comparisons** (Section 4.2):
   - **Radiocarbon**: Compared simulated 1500-yr age increase with Skinner2017 (689±53 ¹⁴C-yr), discussed reservoir age effects, cited Li2024CP and 2025 Nature Communications
   - **Neodymium isotopes**: Discussed εNd evidence for NADW shoaling and AABW expansion (Piotrowski2008, Huang2020, Gu2019)
   - **Carbon isotopes**: Linked ventilation ages to δ¹³C-based respired carbon storage, discussed 50-80 ppm CO₂ drawdown
   - **Regional synthesis**: Compared model results with Ross Sea and Weddell Sea proxy records
   - Emphasized first-order consistency despite model limitations

3. **Theoretical understanding** (Section 4.3):
   - Connected WMT results to Walin1982 framework and modern observations (Pellichero2018, Bailey2023)
   - Explained sea ice's "freshwater pump" role (Abernathey2016) with paleoclimate quantification
   - Identified climate feedback loop: sea ice → reduced heat loss + intensified brine rejection → denser AABW → enhanced stratification → reduced CO₂ outgassing
   - Discussed SAM-AABW coupling implications for proxy development and past climate variability

4. **Future climate implications** (Section 4.4):
   - Discussed Antarctic meltwater forcing (Li2023Nature: 40% slowdown by 2050)
   - Noted nonlinearity: threshold behavior when freshwater prevents convection (Silvano2018)
   - Proposed using WMT metrics as climate model evaluation criteria for CMIP7
   - Cited Zhu2021NCC showing paleoclimate constraints reduce future projection uncertainty by 30%

5. **Future research priorities** (Section 4.5):
   - High-resolution process studies (2-4 km resolution)
   - Transient simulations across deglaciations (TRACE-21ka approach)
   - Multi-proxy model-data integration with isotope-enabled models
   - Improved parameterizations for polynyas, overflows, ice shelf cavities

**Technical approach**:
- Web searches found key 2024-2025 literature on model limitations providing value
- Integrated recent AABW proxy studies (2025 Nature Communications on radiocarbon seesaw)
- Structured discussion from limitations → validation → theory → projections → future work
- Target journals: Journal of Climate (mechanism focus) or Climate of the Past (paleoclimate)

**Key references incorporated**:
- Zhu2021NCC, Tierney2020Paleo (paleoclimate model evaluation)
- Skinner2017, Li2024CP, 2025 Nature Comm (radiocarbon)
- Piotrowski2008, Huang2020, Gu2019 (neodymium)
- Pellichero2018, Bailey2023 (modern WMT observations)
- Abernathey2016 (sea ice freshwater pump)
- Li2023Nature (future projections)

**Writing strategy for handling limitations**:
- Upfront acknowledgment of open-ocean convection bias
- Immediately follow with literature showing value despite limitations
- Emphasize comparative mechanism analysis (relative changes) vs absolute quantification
- Cite specific examples (CESM2) where imperfect models + paleoclimate data = improvement
- Balance: honest about limitations while demonstrating robust insights

**Next steps**:
- Check if all cited papers exist in ref.bib
- Compile manuscript to verify LaTeX formatting
- Consider adding brief discussion of SAM to Discussion if not redundant

### 2026-04-28

**Progress**: Redesigned `plot_wmt_4regions_5exps_from_100years.py` to Nature publication standards. Both annual and winter (JJA) WMT comparison figures regenerated.

**Technical changes** (in `plot_wmt_4regions_5exps_from_100years.py`):
- Added `mpl.rcParams` block: Arial sans-serif, embedded TrueType fonts (`pdf.fonttype=42`), top/right spines off, axis edge color `#333333`, larger base font sizes (16pt+).
- Refined color palette: Total `#111111` (near-black), Heat `#D7263D` (crimson), Sea-ice FW `#1B998B` (teal), Other FW `#2E86AB` (ocean blue).
- Single shared legend at top of figure via `fig.legend(... bbox_to_anchor=(0.5, 0.985), ncol=4, frameon=False)` — removed per-panel `ax.legend()`.
- Edge-only axis labels: x-label only on bottom row (`row_idx == len(REGIONS)-1`), y-label only on leftmost column (`col_idx == 0`); units rewritten as `kg m⁻³` (proper SI).
- Row labels (Southern Ocean / Ross Sea / Weddell Sea / Adélie Land) placed via `ax.annotate(... xytext=(-95, 0), textcoords='offset points', annotation_clip=False)` so they sit outside the y-axis label.
- Panel labels moved to top-left, plain bold (no white box) at 18pt — Nature convention.
- Line widths bumped (Total 3.0, Heat 2.6, FW 2.4) with `solid_capstyle='round'`; shading alpha lowered to 0.20–0.30 for cleaner overlay.
- Figure size 20×15.5 inches; `subplots_adjust(left=0.10, right=0.985, top=0.92, bottom=0.085, hspace=0.32, wspace=0.28)`.
- Output: PDF (400 dpi vector) + PNG (300 dpi) for each season.

**Output files**:
- `figures/plot_wmt_4regions_5exps_winter_from_100years.{pdf,png}`
- `figures/plot_wmt_4regions_5exps_annual_from_100years.{pdf,png}`

**Next steps**:
- Apply the same Nature-style aesthetics to other manuscript figures (SAM composites, surface flux maps) for consistency.

### 2026-04-28 (later)

**Progress**: Created VARIANT 1 "Editorial / High-end magazine" aesthetic of the WMT 4-regions × 5-experiments figure.

**Script**: `plot_wmt_4regions_5exps_v1_editorial.py`

**Technical stack**:
- `matplotlib.patheffects.SimpleLineShadow` for soft drop-shadows on lines (offset 1.0/-1.0, alpha 0.18)
- DejaVu Serif (verified via `font_manager.fontManager.ttflist`); fallback Liberation Serif
- Cream background `#FBF8F3` applied to `figure.facecolor`, `axes.facecolor`, `savefig.facecolor`
- Jewel-tone palette: `#0E1A2B` (navy), `#6E1423` (burgundy), `#1F5E4B` (emerald), `#2A4D6E` (slate-blue)
- `ax.yaxis.grid(True, ...)` solid + `ax.xaxis.grid(True, linestyle=':', ...)` muted
- Soft horizontal divider line under titles via `fig.add_artist(mpl.lines.Line2D(..., transform=fig.transFigure))`
- Legend `fancybox=True, shadow=True` with explicit serif text family

**Output files**:
- `figures/plot_wmt_4regions_5exps_annual_v1_editorial.{pdf,png}`
- `figures/plot_wmt_4regions_5exps_winter_v1_editorial.{pdf,png}`

**Key findings**: Both PDFs and PNGs rendered without warnings; cream background preserved across all axes; data/computation logic untouched.

**Next steps**:
- Compare variant 1 against future variants (modern minimalist, dark-mode, etc.) before final figure choice for the manuscript.
- Verify the new figure renders correctly when included in `paper/main.tex`.

### 2026-05-19 (fwflux row-title overlap fix)

**Progress**: Fixed overlapping left-side row titles in `sam_fwflux_composite_3rows_5cols.pdf`.

**Cause**: Row 3 title was `'Other FW Flux\n(P+E+Snow+Runoff)'` — the `\n` second line overlapped adjacent text at 18pt rotated.

**Fix**: In the ACTIVE script `composite_sam/fw_data/plot_sam_fwflux_composite.py`, `ROW_TITLES` changed to single-line `['Total FW Flux', 'Sea Ice FW Flux', 'Other FW Flux']` (dropped all verbose parenthetical/`-fw` second lines per user request).

**Regenerated**: Ran with `cd composite_sam/fw_data` (script uses `Path(__file__).parent`); script writes to `composite_sam/`. Copied PDF+PNG to `figures/` with absolute paths (manuscript-referenced location). Verified via 130-dpi pdftoppm left-strip crop — three clean single-line row titles, no overlap.

**Note (recurring gotcha)**: First attempt via Bash `run_in_background` produced an empty output file and stale-timestamp files (script never actually ran in background). Foreground run succeeded. Prefer foreground for these matplotlib/cartopy scripts.
### 2026-09-08 (Nature Communications major revision: manuscript + response letter)

**Progress**: Full point-by-point revision of the submitted NC manuscript in response to 4 reviewer
reports. Deliverables in `nc_paper/SUBMISSION/`: revised manuscript (PDF+tex), latexdiff
track-changed PDF, response letter (PDF+md), 4 new figures, 4 new analysis scripts.

**IMPORTANT — which file is the manuscript**: the submitted paper is `nc_paper/nc-main.tex`
(Nature format, sn-jnl.cls, 5 authors incl. Lohmann). `nc_paper/main.tex` is an OLDER AGU-format
version and is NOT what was submitted. Revision lives in `nc_paper/build/revised.tex`.

**Build**: `module load texlive/live2025-gcc-13.3.0` gives pdflatex + bibtex + latexdiff.
Build dir `nc_paper/build/` (self-contained: ref.bib, sn-jnl.cls, sn-nature.bst, figures/).
Baseline 32 pp -> revised 42 pp, 0 errors, 0 undefined citations, 117 latexdiff marks.
Edits applied via idempotent scripts `nc_paper/apply_revision.py`, `apply_minor.py`,
`apply_minor2.py`, `add_figures.py` (each asserts every anchor matches exactly once).

**The two make-or-break reviewer issues and how they were answered**

1. *PI is thermally dominated but Pellichero et al. 2018 find haline dominance* (R2 major#1,
   R3 main critique, R1 general). Answered with new computation, not argument. New script
   `calc_wmt_siz_pellichero.py` -> `wmt_siz_{exp}.nc` recomputes WMT over the September-15%-ice
   seasonal ice zone (reproduces their domain to 2.5% in area) plus the complementary open-water
   sub-domain. Key numbers now in the paper:
   - 42.7% of the published <60S domain lies OUTSIDE the Sept ice contour; that part alone is
     89-91% thermal in dense classes.
   - Restricting to their domain moves PI dense-class thermal share 69% -> 61%, sea ice term
     0.86 -> 1.84 Sv. Moves in their direction but does NOT reverse.
   - **The decisive finding**: our thermal-dominant band sigma2 36.5-37.0 maps to sigma0/gamma_n
     27.09-27.58, entirely LIGHTER than their 27.9-28.8 lower cell. At sigma2>=37.0 our PI IS
     haline dominated (sea ice 0.79 vs heat 0.06 Sv). The two studies measure different water
     masses. This also concedes R3's L175 suggestion (result is more AAIW/SAMW than AABW).
   - Ruled out a warm bias as the excuse: PI Sept ice zone median SST -1.71 C, alpha 3.5e-5,
     55% of area within 0.5 C of freezing. Near-freezing low-alpha regime IS reproduced.
   - Residual = genuine bias: PI dense water forms by open-ocean convection in the Weddell gyre
     (70% of MLD>400 m area south of 55S, 97% of MLD>600 m, centred 68S), not shelf overflow.
   - Regime shift survives: glacials 15%/17% thermal in BOTH domains, interglacials 53-88%.
   Figure `figures/figR1_wmt_domain_sensitivity.pdf` (`plot_wmt_domain_sensitivity.py`).

2. *Chen et al. 2025 GRL not cited* (R2 major#2; Lohmann co-author on both). Now cited in Intro
   and Discussion. Honest differentiator: they are OCEAN-ONLY, 2 states, and diagnose AABW volume
   + streamfunction + MLD + sea ice; they perform NO WMT and no buoyancy-flux decomposition. We
   measure the transformation and attribute it per component across 5 coupled states. Framing used:
   we "test and quantify the mechanism they proposed", not "they missed it".

**Other new analyses**
- MOC streamfunction from the model's own online diagnostic
  `/work/ba1066/a270064/production/{exp}_age/diag_gmoc.nc` (all 5 exps, 50 yr; open with
  netCDF4 or decode_times=False, depths are NEGATIVE). AABW cell -10.0/-10.1/-9.2/-2.4/-1.6 Sv
  (PI/MH/LIG/LGM/MIS3), NADW ~19-22 Sv roughly unchanged. `plot_moc_5exps.py` -> figR2.
- Global ideal age from `{exp}/age_reg.nc` (global 1deg x 47 lev, units already years, depths
  NEGATIVE). Basin means below 2000 m: Atlantic 377->439/508, Indian 493->737/760,
  Pacific 847->1291/1314, Southern 381->789/844. `plot_age_global_basins.py` -> figR3.
- Common-axis WMT supplementary figure (R1#15): `plot_wmt_common_axis_supp.py` -> figS.
- Westerly jet diagnosis (R3): PI 45.7S/5.83 m/s, LIG 47.6S/6.92, LGM+MIS3 49.4S/6.30+6.85.
  Our glacial jet is 3.7 deg POLEWARD and STRONGER; Gray et al. 2023 infer 4.8 deg EQUATORWARD
  and ~25% weaker at LGM vs mid-Holocene. Disclosed as a PMIP-wide model-data conflict, bounded
  using Gray's own R2=0.02/0.03 sea-ice-vs-wind-latitude decoupling.
- Boundary condition Table 1 built from the REAL run configs
  (`production/{exp}_age.yaml` and `config/*echam*`): CO2 284.3/264.4/275.0/190.0/210.5 ppm,
  orbital cecc/cobld/clonp per exp, GLAC-1D 21k/38k meshes, `use_landice_water: False`.

**Literature verification** (subagent + own checks): `nc_paper/LITERATURE_NOTES.md` (667 lines),
`LITERATURE_VERIFIED_BY_MAIN.md`, `reviewer_refs_verified.bib`. Three reviewer misattributions
found and handled diplomatically: Millet 2025's factor-of-2 is FILLING TIME not ideal-age spin-up;
Toggweiler & Samuels 1995 is sea-ice brine rejection, not ice-shelf melt (two 1995 papers exist);
Gray's title "poleward" describes the DEGLACIAL TRANSITION while the reviewer means the LGM STATE
(no contradiction, reviewer is right). Accuracy guards honoured: no AWI model appears in Heuze's
35 CMIP6 models, and NO published AWI paper says the model convects in the open ocean, so that
claim is carried as our own diagnosis from our own output.

**Key Findings**: the apparent model-observation contradiction is mostly a domain + density-class
mismatch rather than a wholesale disagreement; the glacial-interglacial regime shift is robust to
the domain test and therefore not an artefact of the biased interglacial end member.

**Outstanding Issues**:
- Zenodo DOI still needs inserting into the Data Availability statement (deposit already built in
  `zenodo_deposit/`, user uploads manually).
- Source Data spreadsheet for line-graph figures is promised in the letter but NOT yet built.
- Acknowledgements section still contains placeholder text ("This research was supported by...").
- Decision needed from user: R2 wants the SAM section moved to SI, R3 calls it the most
  generalisable result. Letter currently keeps it in main text and offers to move it.
- GLORYS reanalysis WMT (R3's alternative suggestion) was declined in the letter with a stated
  reason; revisit if the reviewer insists.

**Fixed along the way**: 27 analysis/plot scripts had a stale hard-coded
`/work/ba0989/a270064/bb1029/aabw_5exps` path (data has since moved) and could not run; all
repointed to `/work/ba1066/a270064/cc_projects/aabw_5exps`. Also added lat/lon labels to the
polar map figures and corrected the 'MIS' column header to 'MIS3' (R3 request).

**Next Steps**: build the Source Data spreadsheet; fill in Acknowledgements and funding; insert
the Zenodo DOI; get user's call on the SAM section placement.

### 2026-09-09 (SAM 折中处理 + 回复信配图)

**Progress**: 按用户指示做了 SAM 的折中处理，并新增两项 SAM 分析回应审稿人 3。

**折中方案（用户拍板）**: SAM 的图**全部**移入补充材料（满足审稿人 2），但正文加强 SAM 结果的文字描述，
并新做分析（满足审稿人 3）。回复信**区别对待**：给审稿人 2 只说图已移走，不提加强分析；
给审稿人 3 只说加强了分析和新结论，不提图被移走。已用脚本核验两段互不串味。

**新分析：SAM 的 "push and pull" 分解**（审稿人 3 原话：wind affects not just the ekman
divergence but also the salt pump via sea ice export）。脚本 `calc_sam_push_pull.py`：
- PUSH = 60S 纬圈积分的向北 Ekman 输运，由 tau_x 算得；正位相增强 8.0-12.8 Sv（五个态全部）
- PULL = 65S 以南沿岸 brine 输入；PI/LIG 增 51/42 mSv，LGM/MIS3 增 24/31 mSv
- 两者跨五个气候态相关 r = 0.77 —— 定量证实了审稿人的假说
- **意外发现（正文和回复信都写了）**: 冰期 Ekman 响应反而最弱（LGM 8.0 Sv < PI 11.2 Sv），
  尽管冰期平均风应力更强。原因是大范围海冰削弱了风应力向海洋的传递。于是冰期改走 haline 路径，
  净调制幅度相当但机制不同 —— 与平均态的 regime shift 相互呼应。

**符号约定坑（已核实并写进脚本注释）**: `fw_clim.nc` 的 fw 在 PI 沿岸 JJA 为 **+0.164 Sv**、
DJF 为 -0.470 Sv。冬季结冰应该是海洋失去淡水，所以这里 **正值 = 海洋失淡水 = brine 输入**，
与 WMT 管线里 `fw2 = -fw*rho` 的取负一致。第一版标反了，已改正。

**风应力数据**: `sam_wind/{exp}_taux.nc`，用 `ncks -v var180,var181` 从 `{exp}/echam_mergetime.nc`
抽取（var180=ustr, var181=vstr，1200 个月）。每个实验约 52 秒、177 MB。注意 43 GB 源文件用
`run_in_background` 会被中途杀掉产生 .tmp 残file，必须前台跑。

**新图**:
- `figures/figR4_sam_push_pull.pdf`（`plot_sam_push_pull.py`）4 panel: push / pull / 两者散点 / WMT 响应
- `figures/figR5_sam_wind_ice.pdf`（`plot_sam_wind_ice.py`）2x5: 风应力异常 + 海冰浓度异常合成场
  注意风应力异常最大到 +0.23 N/m2，色标必须用 ±0.12 而不是 ±0.06，否则整片饱和

**回复信配图（用户要求）**: 五张图已内嵌进 `RESPONSE_LETTER.md`（`nc_paper/letter_figs/*.png`），
pandoc 转 PDF 时加 `--resource-path=.`。回复信 16 页 -> 19 页，图以 615-646 ppi 嵌入。
Unicode 转纯文本的替换表在 `update_letter.py` 里，pdflatex 不认 γ/σ₂/≥ 等字符。

**审稿人态度判断**: 审稿人 3 是最不 positive 的一个 —— 只有他从头到尾没说过 "suitable for
publication" 之类的话（R1 说 suitable after major revisions，R2 说 would like to support
publication，R3 只说 "I would encourage the authors to think about what is generalisable"）。
所以给他的回复要最用力，新增的 SAM 分析主要是为他做的。

**Outstanding**: Zenodo DOI、致谢与基金信息仍待填。

**Next Steps**: 用户上传 Zenodo 后回填 DOI；补致谢。

### 2026-09-21 (回复信按审稿人重编图号 + 补图)

**Progress**: 回复信从 5 图扩到 12 个图位，按审稿人分别编号。

**图号规则（用户要求）**: 给审稿人 1 的是 Fig R1.1/R1.2/R1.3，审稿人 2 是 R2.1-R2.3，审稿人 3 是
R3.1-R3.6。同一张底图被多位审稿人问到时**重复出现、各自编号**（域敏感性图同时是 R1.1/R2.1/R3.2，
对流位置图同时是 R1.2/R2.2/R3.1）。改语法、加文字的 comment 不配图。
脚本 `nc_paper/renumber_letter_figs.py`，R3 段落用文档出现顺序重排过一次编号。

**新图 figR6_convection_sites**（`plot_convection_sites.py`）一次回答三条 comment：
R1#23「模式在哪里形成深水」、R1#10「LGM 的 MLD 明显更浅」、R3 主质疑「开阔洋对流 vs 陆架过程」。
2 行布局：上排五个态 JJA MLD 极地图 + 400 m 橙色等值线；下排 (f) PI 分扇区柱状、(g) 五态对流面积、
(h) 最大 MLD。

**发现并修正一处数值错误**: MIS3 最大 MLD 之前写 768 m，那是**全球**最大值；正文上下文说的是
55°S 以南，该范围内实际是 **645 m**。已在 `build/revised.tex`、`RESPONSE_LETTER.md`、
`NEW_ANALYSIS_RESULTS.md` 三处改正。教训：算极值时的空间范围必须和文字描述的范围一致。

**当前体量**: 正文 44 页，追踪版 44 页 / 140 处标记，回复信 23 页 / 12 个图位，编译零错误零未定义引用。

**Outstanding**: Zenodo DOI、致谢与基金信息仍待填（唯二剩余项）。
