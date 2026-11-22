# OPEN

- **致谢与基金信息**：`nc_paper/build/revised.tex` 约 360 行仍是 TODO 占位（2026-11-08 核过）。需要用户提供。
- **Zenodo DOI**：数据包 `zenodo_deposit/` 已备好，用户手动上传，之后把 DOI 回填到 Data Availability。是否已上传「待核」。
- **`nc_paper/NEW_ANALYSIS_RESULTS.md` §1 的「69%→61%」**：已被 git 58fb985 推翻，但该文件没改，引用前注意。是否要改这个文件「待核」。
- **合作者审阅进度**：git 0e2557d（10-13）写的是 R2、R3 部分等 Jiping Liu 审；10-15 已合并 R1+R2 的修改。R3 部分是否审过「待核」。
- **旧 AGU 稿里从图上读出的 SAM 数字**（15–20% 调制等）没经过计算，投稿版 `nc-main.tex` 仍含「15-20%」。见 `composite_sam/CLAUDE.md`「待核」。
- **D6/D2 数字待总管核对**（2026-11-08）：D6 已于 10:05 按用户批准推到 TexPage（commit 3d2ad39），D2 的两张表不在稿件里，没有可推的。
- **MOC 两个指数的定义**：间冰期深层环流最强在 19°S，「30°S 以南」的 AABW 指数落在窗口边（31°S）。上层指数 20–70°N 在 PI/MH/LIG 仍落在 580 m 顶边和 21°N 南边。正文措辞已按用户 11-08 10:21 的选择改为「does not weaken and strengthens moderately」，窗口本身是否再改由用户定（`data_moc/CLAUDE.md`）。
- **Methods「We analyze the final 100 years of each run」**：MIS3 的输出到 2616 年，WMT 用的是 2500–2599，不是最后 100 年。待用户确认措辞。
- **`nc_paper/SUBMISSION/letter_figs/figR20_wmt_merged.png` 没进 git**：回复信引用它，TexPage 上也有，但本地是 untracked。要不要提交「待核」。
- **`nc_paper/SUBMISSION/01_revised_manuscript.tex`、`03_response_to_reviewers.md`** 是 09-21 的旧快照，仍含旧 MOC 数字，现在的来源是 `build/revised.tex` 与 `RESPONSE_LETTER.md`。交稿前是否重建或删掉「待核」。
- **climdiag oracle**：`test_d2_formation_csvs` 记了两张形成率表旧版的 md5，表重算后会失败，需 climdiag 那边更新。
- **response_new.tex 待用户定**（2026-11-19）：docx 带回了 D6/D8 之前的 MOC 数字与措辞（要不要重新套用）；图号改成 R1–R15 连续、重复图各放一次是否接受；信头没有 Dear Editor、配色说明、总述、更正几段（docx 里没有）（旧 Word/PDF 已于 11-19 按用户决定从 TexPage 删除，baa76c5）；README.md 是否加 response_new.tex 一行。批注本身留到下一轮处理。
- **PI 海冰气候态文件被覆盖**（2026-11-22 发现，最要紧）：`pi/a_ice_clim.nc`、`pi/a_ice_reg.nc` 是 03-15 换进来的来历不明的 2090–2099 十年平均，正确的 2300–2399 版在 `*_beta.nc`。受影响：修订稿 Fig. 1i–l 与 Fig. 2 的 PI 海冰、Fig. 4 的 PI 海冰掩膜、`calc_wmt_siz_pellichero.py` 的 PI 海冰区（9 月冰区大 13%，牵连信里 42.7%、70%、−9.2/+6.6 Sv 与 Figs R1、R9、R12、R13 的 PI 曲线）。是否用 _beta 统一重算待用户定（`log/PITFALLS.md`）。
- **Fig. 4 淡水密度项少乘水密度约 1000 倍**（2026-11-22）：`plot_surface_density_tendency_from_raw_data_jja.py` 把 RHO_WATER 约掉了，图和正文 149 行的 5×10⁻⁹ 偏小三个量级；两图单位标的 kg m⁻³ 也应是每米水柱。改不改待用户定（`fig_sam_buoyancy/CLAUDE.md`）。
- **Figs A3/A4（SAM 热、淡水合成）实为 ±1.0σ**，图注写 1.2σ（`composite_sam/*_sam_composite_*.nc` 属性和脚本都是 1.0）。重画还是改图注待用户定。
- **response_new.tex 11-22 版待用户定**：修订稿要跟着改的段落（Discussion 245 行 Sidorenko 那段及 366 行 TODO、Fig. 1i–l 的描述、111 行 residual、241 行 they describe、249 行 53–88%、180 行 wherever、摘要 Weaker glacial transformation）；Hall & Visbeck 2002 还没进 ref.bib；批注 6、7 要看纸质修改稿；批注 8 的意思；新图 R3（海冰）、R10（SAM 密度）是否进补充材料。
