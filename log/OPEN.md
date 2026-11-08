# OPEN

- **致谢与基金信息**：`nc_paper/build/revised.tex` 约 360 行仍是 TODO 占位（2026-11-08 核过）。需要用户提供。
- **Zenodo DOI**：数据包 `zenodo_deposit/` 已备好，用户手动上传，之后把 DOI 回填到 Data Availability。是否已上传「待核」。
- **`nc_paper/NEW_ANALYSIS_RESULTS.md` §1 的「69%→61%」**：已被 git 58fb985 推翻，但该文件没改，引用前注意。是否要改这个文件「待核」。
- **合作者审阅进度**：git 0e2557d（10-13）写的是 R2、R3 部分等 Jiping Liu 审；10-15 已合并 R1+R2 的修改。R3 部分是否审过「待核」。
- **旧 AGU 稿里从图上读出的 SAM 数字**（15–20% 调制等）没经过计算，投稿版 `nc-main.tex` 仍含「15-20%」。见 `composite_sam/CLAUDE.md`「待核」。
- **西风急流计算脚本**没找到，季节和高度未记录（`docs/westerly_jet.md`）「待核」。
- **D6/D2 数字待总管核对**（2026-11-08）：核对前只改本地并推 GitHub，不推 TexPage。
- **MOC 两个指数的定义**：间冰期深层环流最强在 19°S，「30°S 以南」的 AABW 指数落在窗口边（31°S）；「NADW」指数在 PI/MH/LIG/LGM 落在 580 m 热带的窗口顶边，量的不是 NADW（`data_moc/CLAUDE.md`）。正文「upper cell ... changes comparatively little」靠的是这个指数。是否改定义由用户定。
- **Methods「We analyze the final 100 years of each run」**：MIS3 的输出到 2616 年，WMT 用的是 2500–2599，不是最后 100 年。待用户确认措辞。
- **TexPage 推送闸门**（2026-11-08 10:15 停在闸门前，等总管）：网页端 `response_letter_R12_1015.docx` 只在 10-15 05:58 被上传过一次，是用户带中文批注的草稿，之后的本地提交 8610cf7、86410e0、17238bf 已照做。阻断判定只报那 2 段中文批注；但闸门另打印「55 paragraph(s) differ from the built docx」，那是拿它和 R1 单独草稿 `response_letter_R1_1013.docx` 比的（`strict_diff` 比错了对象）。改和完整版 `05_response_letter.docx` 比，有 13 段不同：2 段批注，2 段是 D6 的 10→11，9 段是旧草稿与定稿之差（typo、43%→42.7%、中文占位被扩写）。线上三个 .tex 与本地 D6 之前完全相同。推送需 `--reviewed 25b3800a5e19`（远端一变即失效）。
- **`nc_paper/SUBMISSION/01_revised_manuscript.tex`、`03_response_to_reviewers.md`** 是 09-21 的旧快照，仍含旧 MOC 数字，现在的来源是 `build/revised.tex` 与 `RESPONSE_LETTER.md`。交稿前是否重建或删掉「待核」。
- **climdiag oracle**：`test_d2_formation_csvs` 记了两张形成率表旧版的 md5，表重算后会失败，需 climdiag 那边更新。
