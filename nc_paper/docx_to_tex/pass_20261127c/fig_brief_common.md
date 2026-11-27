# Common brief: make every text in the reply-letter figures readable on an A4 printout

Project root: /work/ba1066/a270064/cc_projects/aabw_5exps (AWI-ESM paleoclimate paper, Nature Communications major revision).
Before starting read: root CLAUDE.md (section 全项目的坑), docs/figure_registry.md, and the CLAUDE.md of any figure folder you touch.

## Why
The user (the paper's author) asked on 2026-11-27: review every figure in the response letter; text, axis titles,
legends, subplot titles, colorbar labels, in-figure annotations are too small for some. Enlarge them and make them darker /
bolder so that a reviewer who prints the letter on A4 can read every word. After enlarging, nothing may be cropped,
overlapped or hidden.

## How big a figure prints
The letter is A4, text width 16.4 cm. Each figure is inserted at 0.9 x textwidth = 5.81 in (14.76 cm) wide, but the height
is capped at 5.93 in (keepaspectratio). Printed width = min(5.81, 5.93 * W/H) inches (W/H = saved image width/height).
Exception: figS_formation_location is inserted at 0.95 -> min(6.13, 5.93*W/H).
Printed font size (pt) = native font size (pt) x printed width / saved figure width (inches).
For PNG: saved width (in) = pixel width / savefig dpi (bbox_inches='tight' changes the width, so use the real pixels).
For letter PNGs made from a PDF with pdftoppm: saved width = PDF page width in pt / 72.

## Targets on the printed page
- Absolute floor 7 pt for every text element. Aim: tick labels >= 7.5 pt; axis titles, colorbar labels and tick labels,
  legend entries, subplot titles, in-plot annotations and region labels >= 8 pt; panel letters (a, b, ...) >= 9 pt bold.
- Lines: data lines >= 0.75 pt printed; contour lines, coastlines, axes spines, tick marks >= 0.5 pt printed.
- "加黑" (darker): all text pure black, no grey and no alpha, unless the colour itself carries meaning (e.g. a legend or
  label coloured to match its line). Subplot titles, axis titles, colorbar labels and panel letters bold. Tick labels and
  legend entries regular weight, black.
- If a figure already meets these targets leave it untouched and say so.

## Hard rules
- Data, colour ranges, colormaps, contour levels, axis limits, panel content and panel order stay exactly as they are.
  Allowed: figsize, subplot spacing, colorbar and legend placement, number of tick labels shown, line breaks in titles,
  abbreviations already used elsewhere in the figure. Do not change any number or word in a label beyond a line break.
- After enlarging: no text overlaps other text, data, a colorbar or another panel; nothing is clipped at the figure edge
  or truncated; legends do not cover data; tick labels on neighbouring axes do not collide.
- Keep the figure's existing font family.
- Height/width: keep H/W such that the printed width stays as large as before (do not make a figure taller than its width
  allows, see the height cap above), unless the layout needs it and you state the new printed width.
- Before overwriting, back up every old output with suffix _v20261127pre:
  root-script outputs -> figures/versions/; folder figures -> <folder>/versions/; letter PNGs ->
  nc_paper/SUBMISSION/letter_figs/versions/; manuscript PDFs in nc_paper/build/figures/ -> nc_paper/build/figures/versions/.
  rm is blocked by a hook; never delete anything.
- Regenerate, then put the new letter PNG at nc_paper/SUBMISSION/letter_figs/<name>.png (letter resolution: at least
  300 px per printed inch, keep about the old pixel width) and the new manuscript PDF at nc_paper/build/figures/<name>.pdf
  where the figure is also in the manuscript or supplement (names given in your task).
- Python: `module load python3/2022.01-gcc-11.2.0`, run with `python3 -s` from the project root, in the FOREGROUND
  (background matplotlib/cartopy jobs get killed silently and leave stale files). Some scripts need gsw on PYTHONPATH
  (see fig_mld_seaice/CLAUDE.md, fig_formation_location/CLAUDE.md). Check timestamps and md5 after each run so you know the
  file really changed.
- Check every changed figure yourself with the Read tool twice: at full resolution, and downscaled to its printed width at
  150 px per inch (e.g. `convert in.png -resize 872x out.png` for 5.81 in; put previews in your own scratch folder under
  /tmp/claude-22370/-work-ba1066-a270064-cc-projects-aabw-5exps/d1f4281c-1793-4a52-ae53-a207b50475a1/scratchpad/agents/<your name>/).
  Also render the manuscript PDF (pdftoppm -r 100) and look at it once.
- Do not commit, push, touch TexPage, edit any .tex file, use share_fig or contact anyone. Edit only the scripts assigned
  to you (other agents are editing other scripts at the same time). Do not edit docs/figure_registry.md (the lead does).
  You may add one History line to the CLAUDE.md of a figure folder you own (Chinese, short, no 「不是…而是」 constructions).

## Report back (in Chinese, compact)
For each figure: script and exact command; saved width (in) and printed width (in); scale factor; smallest printed font
before (which element, pt) and after; what you changed (sizes, weights, figsize, layout, line widths); new md5 of the
letter PNG and of the manuscript PDF; backup paths; anything you could not fix (e.g. an embedded image from a published
paper) or anything that looked wrong in the data while you were there (report it, do not fix it).
