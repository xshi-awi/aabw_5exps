# OPEN

- **致谢与基金信息**：`nc_paper/build/revised.tex` 约 360 行仍是 TODO 占位（2026-11-08 核过）。需要用户提供。
- **Zenodo DOI**：数据包 `zenodo_deposit/` 已备好，用户手动上传，之后把 DOI 回填到 Data Availability。是否已上传「待核」。
- **`nc_paper/NEW_ANALYSIS_RESULTS.md` §1 的「69%→61%」**：已被 git 58fb985 推翻；修订稿 239 行 11-25 已改成分密度类结果，NEW_ANALYSIS_RESULTS.md 本身没改，引用前注意。
- **合作者审阅进度**：git 0e2557d（10-13）写的是 R2、R3 部分等 Jiping Liu 审；10-15 已合并 R1+R2 的修改。R3 部分是否审过「待核」。
- **旧 AGU 稿里从图上读出的 SAM 数字**（15–20% 调制等）没经过计算，投稿版 `nc-main.tex` 仍含「15-20%」。见 `composite_sam/CLAUDE.md`「待核」。
- **D6/D2 数字待总管核对**（2026-11-08）：D6 已于 10:05 按用户批准推到 TexPage（commit 3d2ad39），D2 的两张表不在稿件里，没有可推的。
- **MOC 两个指数的定义**：间冰期深层环流最强在 19°S，「30°S 以南」的 AABW 指数落在窗口边（31°S）。上层指数 20–70°N 在 PI/MH/LIG 仍落在 580 m 顶边和 21°N 南边。正文措辞已按用户 11-08 10:21 的选择改为「does not weaken and strengthens moderately」，窗口本身是否再改由用户定（`data_moc/CLAUDE.md`）。
- **Methods「We analyze the final 100 years of each run」**：MIS3 的输出到 2616 年，WMT 用的是 2500–2599，不是最后 100 年。待用户确认措辞。
- **`nc_paper/SUBMISSION/letter_figs/figR20_wmt_merged.png` 没进 git**：回复信引用它，TexPage 上也有，但本地是 untracked。要不要提交「待核」。
- **`nc_paper/SUBMISSION/01_revised_manuscript.tex`、`03_response_to_reviewers.md`** 是 09-21 的旧快照，仍含旧 MOC 数字，现在的来源是 `build/revised.tex` 与 `RESPONSE_LETTER.md`。交稿前是否重建或删掉「待核」。
- **climdiag oracle**：`test_d2_formation_csvs` 记了两张形成率表旧版的 md5，表重算后会失败，需 climdiag 那边更新。
- **response_new.tex 待用户定**（2026-11-19）：docx 带回了 D6/D8 之前的 MOC 数字与措辞（要不要重新套用）；图号改成 R1–R15 连续、重复图各放一次是否接受；信头没有 Dear Editor、配色说明、总述、更正几段（docx 里没有）（旧 Word/PDF 已于 11-19 按用户决定从 TexPage 删除，baa76c5）；README.md 是否加 response_new.tex 一行。批注本身留到下一轮处理。
- **PI 海冰气候态文件被覆盖**（2026-11-22 发现，最要紧）：`pi/a_ice_clim.nc`、`pi/a_ice_reg.nc` 是 03-15 换进来的来历不明的 2090–2099 十年平均，正确的 2300–2399 版在 `*_beta.nc`。受影响：修订稿 Fig. 1i–l 与 Fig. 2 的 PI 海冰、Fig. 4 的 PI 海冰掩膜、`calc_wmt_siz_pellichero.py` 的 PI 海冰区（9 月冰区大 13%，牵连信里 42.7%、70%、−9.2/+6.6 Sv 与 Figs R1、R9、R12、R13 的 PI 曲线）。是否用 _beta 统一重算：用户 EE1、FS、FX（2026-11-25 09:41）都定「先不动」，修订稿 239/249 行的 PI 数也照留（`log/PITFALLS.md`）。
- **SI 图（Fig. A6）与 Fig. 1 的 PI 海冰仍读被覆盖的文件**：用户 FS=先不动（2026-11-25）。R3 已按 FT 用正确 PI 的 SI 画法（TexPage 9115556），所以 R3 和 A6 的 PI 在威德尔海不一致，A6 重画后会一致。11-25 起 A6 编为 A2；R3（卫星海冰对比）因此没放进补充材料，讨论里只用数字并引 OSI SAF 数据集，A2 重画后可以补进去。
- **`plot_climate_patterns_jja.py` 实为 6–9 月**：11-25 只把 Fig. 1、A1、A2 图注和 2.1 节改写成 June-September，脚本和图没重算（总管定）。A4 已换成真 JJA 的 `figR7_mld_glacial_zoom.pdf`。
- **11-25 下午一轮留给用户的事**：docx 批注 8（brine rejection 导致海冰扩张？）的意思；摘要第 2/3 条引文按稿件原句（Jiping 10-13 版）对齐，没采用信里 docx 那版「set by sea ice cover, which is strongly influenced by the SAM」；第 4 条引文改为摘要原句（collapse、11→2 Sv）。PI 海冰区（EE1）仍是被覆盖的文件，信里 42.7%、70% 等和修订稿 239/249 行的 PI 数随之待定。
- **11-26 一轮留给用户的事**：A10（SAM 风与海冰图）原描述不成立已改，图本身要不要留在补充材料由她定；drift 那条她标了「回头再看」，Methods 里 LGM/MIS3 从哪里起步（她说大气 AMIP 冷启动、海洋来自转了 1500 年的旧版本）要她给出处才写；my_style 加「不用 不是…而是」一条等总管落实。R12/A13 冰缘线 08:33 两边都去掉了（GC）。11-26 08:30 一轮她问的 MOC 两问（上层环变强与 WMT 减弱是否矛盾、下层环减弱是否就是变老原因）答在信里批注，待她看要不要给审稿人加一句。「The relevant contents … reads」那句 11-27 按她按钮 GE 改成 text（TexPage 259cd2f）。
- **11-27 一轮留给用户的事**：R3 主意见 2「This is a shared model bias in most climate models」没有出处（可引 Heuzé 2021 或 Toggweiler & Samuels 1995，或改 many）；2.4 节比投稿版长一倍，要不要砍（审稿人 2 要缩短）；信里图 R3 用正确 PI，A2 的 PI 海冰仍是被覆盖文件（FS），两张看着不一样；R3 末条（图顶标态名）原来没回复，补的那句沿用 R1 第 17 条的说法，没有逐张重看。
