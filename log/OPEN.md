# OPEN

## 现状（2026-11-27 14:51 起暂停改稿和信，不推 TexPage，等她说开始）

- **合作者 11-27 收到的版本**：回复信 Word e 版 `nc_paper/SUBMISSION/response_letter_20261127e.docx`（由 TexPage 16296ee 生成）和修订追踪版 b 版 `manuscript_tracked_20261127b.docx`（原稿 TexPage 投稿版，修订稿 TexPage 1423854）。
- **e 版之后信里改过、合作者手上的 e 版里还没有的**（TexPage 9163f4f，`pass_20261127c/edits_1127i_consistency.py`）：
  - Heuzé「almost all」→「28 of 35」（两处）；
  - 给 R1 #27 的回复 37.0–37.5、12–17% → 37.1–37.6、17–19%；给 R3 #19 的回复 36.5–36.8 → 36.4–36.8；
  - 「1.9° further south」→「one grid row (1.9°)」；「same paragraph」→「next paragraph」；「at each mesh」→「at each surface node」；
  - 引文同步：LIG brine 42 → 41 mSv（两处）、「while the Last Interglacial」、A7 虚线语序、「following Berger (1977)」、68°S 句加 Supplementary Fig. A12。

  稿件在 e 版之后的改动都在 b 版里。合并合作者意见时以当时的线上版为底，对照 e 版找出他们改了什么。

## 待她定（每条一行，明细见所指文件）

- ② 信图 R20 和信里 −9.2/+6.6、ORAS5 +5.0/+4.4、+62.9/+36.8 Sv 仍是 0.2 格、12 个气候月，Fig. 8 已是 0.1 格、100 年（PI 海冰区峰值 13.0 Sv）：重算还是图注注明（`pass_20261127c/consistency_table.md`）。
- ③ 讨论「only one class … σ2 = 36.75–37.00」是旧格结果，0.1 格下是 36.70–36.90；信里热主导带 γ 27.1–27.67 也要一起定（同上）。
- ④ A11 那句「comparable annual-mean … to modern observations」：0.1 格下 PI 年均峰值 11.4 → 17.5 Sv（同上）。
- ⑤ 信里承诺了、稿件没有：R1 #14 冰期热占比（含「almost the entire」）、R1 #26「factor of 2–5」、R1 #25「We corrected them」、R3 #8 引言 ideal age 当翻转代理以及信里没提 Millet 与混合、R3 #15 扣掉均匀盐度增量的图（同上）。
- ⑥ R3 主意见 2「This is a shared model bias in most climate models」没有出处，可引 Heuzé 2021、改成 many 或删（同上）。
- ⑥′ 信里还引 Green 2022，稿件 Lhardy 段已随她的缩写删了它；要不要在稿件里留一句（AWI-ESM 冬季冰偏少，对冰期结论有利）。
- ⑦ 给 R2 大问题 1 的回复没说稿件改了什么；信说「mixed」，稿件说 PI 热主导、海冰区峰值处热占 76%（`consistency_table.md`）。
- ⑧ 2.1 节「polynyas (Fig. A3)」指向网格图，投稿版是 A2d,e；改回后 A3 要么再引一次，要么挪到补充材料最后、其余图号顺移（同上）。
- ⑨ 信里没交代的实质改动：格宽 0.2→0.1、密度倾向的单位和正负、SAM 阈值 ±1.0σ/±1.2σ、SAM 6–10 Sv 与「各态相近」、年龄三条限定、方法里的代用资料评估、Fig. 1/A1/A2 实为 6–9 月、摘要末句、方法 4.4 MOC、M(ρ) 定义（`consistency/reverse_table.md`）。
- ⑪ 可选语言小修：信「For annual mean」「Ours are fully coupled model」、给 R4 那句、「Reviewer 1 has the same concern」；2.4 节两处 systematically（`consistency_table.md`）。
- ⑫ 致谢已从投稿版恢复基金文字（2024YFF0506604、SML2023SP204、ONCE），要不要更新。
- 信里图 R3 是试看版：用的是正确 PI 海冰（2300–2399），稿件 A2 和 R2 的 PI 海冰还是被覆盖的 2090–2099 文件（FS），两张看着不一样。留不留由她定，回退把 \rfig 改回 `figR21_seaice_obs.png`（`nc_paper/docx_to_tex/pass_20261127c/`）。
- R1 模式评估那条后面引的三段讨论原文是否嫌长。
- R1、R13 的 (b) 是 Pellichero 原图，字无法放大；R13 可改成上下排。
- 稿件 A2 要不要像信里 R20 一样去掉海冰行、加经纬度（信里已只放 MLD a–e）。
- 为放大字号减少了经纬度标签：A12、A4 纬度只标 70°S、80°S，A13 经度只标 90°W/E，A7/A8/A10 只在一格标纬度。是否接受（`pass_20261127c/fig_font_checklist.md`）。
- 2.4 节比投稿版长一倍，审稿人 2 要缩短：要不要砍。
- A10（SAM 风与海冰图）原描述不成立，已改；图本身留不留在补充材料。
- 方法里 LGM/MIS3 从哪里起步（她说大气 AMIP 冷启动、海洋来自转了 1500 年的旧版本）要她给出处才写；「We analyze the final 100 years of each run」对 MIS3 不准（输出到 2616 年，用的是 2500–2599）。
- MOC：她 11-26 问的两问（上层环变强与 WMT 减弱是否矛盾、下层环减弱是否就是变老的原因）答在信里批注，要不要给审稿人加一句；两个指数的窗口要不要再调（AABW 指数落在 31°S 窗口边，`data_moc/CLAUDE.md`）。
- FS/FX：`pi/a_ice_clim.nc`、`a_ice_reg.nc` 是被覆盖的 2090–2099，正确的是 `*_beta.nc`（2300–2399）。受影响：Fig. 1i–l、Fig. 2、Fig. 4 的 PI 海冰，A2 的海冰格，PI 海冰区（信里 42.7%、70%、−9.2/+6.6 Sv，Figs R1、R9、R12、R13）。她定「先不动」（`log/PITFALLS.md`）。
- 11-25 一轮：docx 批注 8（brine rejection 导致海冰扩张？）的意思；摘要第 2–4 条引文按稿件原句对齐，没用 docx 里的说法。

## 其他未办或待核

- Zenodo DOI：`zenodo_deposit/` 已备好，她手动上传后回填 Data Availability；是否已上传「待核」。投稿版写的是 zenodo.org/records/20280950，修订稿是「DOI to be inserted upon acceptance」。
- `plot_climate_patterns_jja.py` 实为 6–9 月：只改了 Fig. 1、A1、A2 图注和 2.1 节，脚本和图没重算（总管定）。
- `nc_paper/NEW_ANALYSIS_RESULTS.md` §1 的「69%→61%」已推翻，文件本身没改，别引。
- my_style 加「不用 不是…而是」一条，等总管落实。
- 信头有「Summary of Responses to Reviewers」，没有 Dear Editor 段（11-19 待定，「待核」）。
- `nc_paper/SUBMISSION/letter_figs/figR20_wmt_merged.png` 本地 untracked；`SUBMISSION/01_revised_manuscript.tex`、`03_response_to_reviewers.md` 是 09-21 旧快照：交稿前提交、重建或删掉「待核」。
- D6/D2 数字待总管核对（11-08）；climdiag oracle `test_d2_formation_csvs` 记的旧 md5 要那边更新。
- 合作者审阅：10-13 起 R2、R3 部分等 Jiping Liu 审，R3 部分是否审过「待核」；11-27 已把 e 版信和 b 版追踪稿发给合作者。
