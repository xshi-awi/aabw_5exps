# PROGRESS

按时间倒序。2026-11-08 从根 CLAUDE.md 压缩迁来，同主题已合并，只留结论；逐次细节在 git 历史里。具体数字多在 `nc_paper/NEW_ANALYSIS_RESULTS.md` 和 `docs/`。

### 2026-11-08 10:40 D8 上层环流指数改为 20–70°N（数字待总管核对）

- 查证：`plot_moc_5exps.py` 从 758dfa6（09-08）起就是 0–70°N，投稿稿 `nc-main.tex` 没有 MOC 分析，修订稿正文、图注、Methods、两封信都没写过上层窗口，唯一的「0–70N」是 D6 时照抄代码写进 `NEW_ANALYSIS_RESULTS.md` 的。无文字依据，按用户说的 20–70°N 改。
- NADW（500–4000 m、20–70°N）PI/MH/LIG/LGM/MIS3 = 18.5/19.8/21.6/20.1/21.5 Sv（0–70°N 时 19.5/20.4/22.1/20.9/21.5）。PI、MH、LIG 仍卡在 580 m 顶边和 21°N 南边；1000 m 以下参照值 16.3/16.6/17.4/19.9/21.5，全在 1040 m 顶边。
- 改了 `plot_moc_5exps.py`（及副本）、`data_moc/calc_moc_5exps.py`（重跑 job 28821814）、Source Data、`revised.tex` 2.4 节数字，新增 Methods 4.4 Overturning streamfunction 写明两个窗口；`RESPONSE_LETTER.md` R3 那段的数字。图只标 AABW，PNG 字节不变，没换。「upper cell … changes comparatively little」相对 −80–86% 仍成立（+8%、+16%），未改写。

### 2026-11-08 09:45 D6 MOC 分带更正，D2 形成率表同源（总管任务，数字待总管核对）

- D6：`production/{pi,mh,lig}_age/diag_gmoc.nc` 按旋转纬度分带，独立核实。`data_moc/verify_moc_binning.py` 不用 climdiag，从 ASCII 网格反旋转（Euler 50/15/−90）、自算面积（与 diag 文件差 2.6e-13），旋转纬度装箱把三个文件复现到 0.0000 Sv，地理纬度装箱差 30–33 Sv；跨 79°S 输运文件里有 2.3–2.5 Sv，地理上应为零。run 的网格是 `awicm2_final384`（与 `mesh_core2` 水平相同、垂向层数不同）。
- MOC 改由 `data_moc/calc_moc_5exps.py` 用 climdiag 0.5.0（46e6514）`moc_depth` 从原始 w 重算，五个态统一取 100 年分析期的最后 20 年。AABW 指数 PI/MH/LIG/LGM/MIS3 由 −10.0/−10.1/−9.2/−2.4/−1.6 变为 −11.7/−11.3/−11.4/−2.3/−1.7 Sv，降幅「75–85%」变为「80–86%」，摘要「10 to 2 Sv」变为「11 to 2 Sv」；NADW 19.1/19.1/21.3/20.0/21.7 变为 19.5/20.4/22.1/20.9/21.5 Sv。旧文件的窗口原来五个态各不相同（LGM 是 2350–2369，MIS3 是 2590–2609）。
- 更新了 `plot_moc_5exps.py`（及 SUBMISSION 副本）、Fig. 6 / Fig R2.5 / R3.4 的图、`revised.tex` 摘要与 2.4 节与图注、`RESPONSE_LETTER.md` 与 `RESPONSE_LETTER_R1.md`（引文经 `extract_quotes.py` 重抽）、`make_source_data.py` 与 `Source_Data.xlsx`、`NEW_ANALYSIS_RESULTS.md` §2。`tex_to_docx*.py` 的中间文件路径指向一个已不存在的会话目录，改为系统临时目录。
- D2：`data/aabw_formation_*_final.csv` 原是 0.25 bin 气候态文件算的，改为读主图的 100 年逐月 0.2 bin 文件（`plot_wmt_sigma_all_36.88_37.62_37.12.py`），PI 南大洋 Total 8.109→8.049，LGM 11.341→9.345 Sv。climdiag `surface_wmt` 从输入重算全部 1200 个月，文件复现到相对 5e-8，表差 ≤0.0005 Sv。两张表的数字不在修订稿和回复信里。

### 2026-10-15 07:34（据 git log 补记，旧根 CLAUDE.md 没有记录）

09-22 到 10-15 共 48 个提交，以下按提交信息整理，细节以 git log 为准。
- 58fb985（09-22）修正密度类结论：「限定到观测域后 PI 致密类热贡献 69%→61%」来自 σ2 36.6–36.8 以上的积分，那不是致密类；在转换极大处测是 68%→70%，不降。改为分密度类结果：海冰区内 PI 只有 σ2 36.75–37.00 一类热主导（70%），其余全是盐主导；两个冰期全程盐主导（热占 4–28%）。冰期类边界平移 +0.80 / +0.50 kg m⁻³，即实验设计里 +1.0 / +0.6 的盐度增量在冰点产生的密度差。
- 09-23 起 TexPage 协作：`nc_paper/sync_texpage.sh`，push 前检查网页端改动；`nc_paper/check_docx_merged.py` 作合并闸门。
- 10-13 按 Jiping Liu 当天的意见重写 R1 回复，另出 R1 单独草稿 `RESPONSE_LETTER_R1.md`。`docs/reanalysis_wmt_feasibility.md` 定稿。
- 10-15 ORAS5 WMT（`calc_wmt_oras5.py`、`calc_wmt_oras5_so60.py`，1999–2018）；Fig R1.1a 仿 Pellichero Fig. 3a 重画。海冰区年均 +5.0 vs +4.4 Sv 接近；60°S 以南 JJA +62.9 vs +36.8 Sv 不接近，写成混合层偏差。合并用户 `revision/response_letter_R12_1015.docx` 的修改和 12 处黄色批注。R3 的再分析建议改为用 ORAS5 正面回答；R3 第一条只留合并对比图，回复信共 17 张图。

### 2026-09-21 15:59 回复信成形，模式平衡诊断

- 回复信改为按审稿人分别编图号，出 LaTeX（sn-jnl.cls）和 docx，三个格式都由 `nc_paper/RESPONSE_LETTER.md` 单一来源生成，流程见 `nc_paper/CLAUDE.md`。
- 处理用户在 `05_response_letter_suggestion.docx` 的 23 处黄色批注（清单 `nc_paper/SUGGESTION_TODO.md`）：每条回复用斜体引出修订稿原文，程序化抽取；新图 figR7、figR8、figR9。
- 用户复查查出三处问题，都已修：13 条审稿意见被省略号截断，已逐字还原；两处漏配图，补图时发现 Fig.3/4 确实没有气候态列标签，已补画；图号重复和一处交叉引用错位。之后新增 `nc_paper/check_letter.py`。
- R1#13 Ross 海的回答原先附和审稿人，实算推翻（`docs/ross_sector_seaice.md`）。MIS3 最大 MLD 由 768 m 改正为 645 m。
- 模式平衡：spin-up 由 GRL Text S1 确认；全球地表温度漂移 LIG/MIS3 超标；冰期深海没有平衡。全部如实写进 Methods（`docs/model_equilibrium.md`）。
- 当日末体量：正文 46 页，追踪版 148 处标记，回复信 LaTeX 42 页，docx 19 张图。

### 2026-09-09 04:40 SAM 折中与 push/pull 分析

用户拍板：SAM 图全部进补充材料，正文加强 SAM 文字并做新分析，回复信对 R2、R3 区别表述（`nc_paper/CLAUDE.md`）。新分析 SAM push/pull 和风应力、海冰合成，结果见 `docs/sam_push_pull.md`。回复信开始内嵌图（`nc_paper/letter_figs/`）。

### 2026-09-08 19:02 NC 大修启动

对 4 份审稿意见逐条修订，产出在 `nc_paper/SUBMISSION/`。最关键的两个问题：
- **PI 热主导与 Pellichero et al. 2018 的盐主导矛盾**（R1、R2、R3 都提）。用 `calc_wmt_siz_pellichero.py` 在 9 月海冰区重算 WMT。结论：差异主要来自积分域和密度类不同。我们的热主导带 σ2 36.5–37.0 对应 σ0/γn 27.09–27.58，整体轻于他们 27.9–28.8 的下层环流；σ2 ≥ 37.0 时 PI 是盐主导（海冰 0.79 vs 热 0.06 Sv）。不是暖偏差（PI 9 月海冰区 SST 中位数 −1.71 °C）。残差是真实偏差：PI 致密水由 Weddell 环流开阔洋对流产生，而非陆架溢流。冰期与间冰期的 regime shift 在两种域下都成立（冰期热占 15% / 17%，间冰期 53–88%）。数字见 `NEW_ANALYSIS_RESULTS.md` §1、§6、§7，其中「69%→61%」已被上面 58fb985 推翻。
- **未引 Chen et al. 2025 GRL**（R2）。已补引，区别与措辞见 `nc_paper/CLAUDE.md`。

其他新分析：模式自带 MOC（AABW 环流 PI/MH/LIG 约 −10 Sv，LGM/MIS3 −2.4 / −1.6 Sv，NADW 19–22 Sv 基本不变）；全球理想年龄分海盆；共用坐标轴 WMT 补充图；西风急流（`docs/westerly_jet.md`）；边界条件 Table 1 取自真实 run 配置（CO₂ 284.3 / 264.4 / 275.0 / 190.0 / 210.5 ppm，`use_landice_water: False`）。数字见 `NEW_ANALYSIS_RESULTS.md` §2–§5。27 个脚本的旧数据路径统一改到本目录；极地图加经纬度标注，列标题 MIS 改为 MIS3。

### 2026-05-19 --:-- Zenodo 数据包，引言改写

- `zenodo_deposit/`：约 1.3 GB，141 个 nc 加 `README.md`，只含论文每张图背后的数据。用户选定的取舍：`echam_clim.nc` 只留 8 个变量（`ncks -v var92,var95,var111,var120,var165,var166,var180,var181` → `echam_clim_subset.nc`，7 MB）；`mask.nc` 只留一个时间步（`ncks -d time,0` → `mask_surface.nc`，47 层全留，要用 3900 m 那层）；100 年 WMT 全量保留（约 487 MB）；去掉 DJF 的 SAM 文件；mesh 只放 core2。用户手动上传。
- 在 `nc_paper/main.tex`（当时的工作稿）改写引言两段：去掉 this/that/such 回指，换成具体名词；WMT 段按「是什么 → 可按分量分解 → 现代观测发现」组织；新引 Adkins2002、Ferrari2014、Jansen2017。
- SAM 阈值文档统一为 1.2σ（`composite_sam/CLAUDE.md`）。

### 2026-05-18 --:-- 图件打磨

WMT 4 区图改 Nature 风格（04-28，另做了三个风格变体，未采用）；SAM 合成图、年龄图、表面密度倾向图放大字号，修空角和重叠；fig02 做了分色标变体。具体决定见 `docs/figure_registry.md` 和 `composite_sam/CLAUDE.md`。

### 2026-01-19 03:00 初稿与改向

1 月中在 AGU 模板（`paper/main.tex`，现已删）写完 SAM 合成一节和讨论。01-19 改向：模式靠开阔洋对流而非陆架过程产生致密水，而且间冰期热强迫占优与观测相反，所以论文从「AABW 形成机制」改为「南大洋 WMT 分析」。正文用 dense water formation 或 WMT 代替 AABW 形成率；讨论先讲模式局限，再讲稳健结论，AABW 的含义放在讨论里。这一框架延续到投稿版。
