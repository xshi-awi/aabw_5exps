# PROGRESS

按时间倒序。2026-11-08 从根 CLAUDE.md 压缩迁来，同主题已合并，只留结论；逐次细节在 git 历史里。具体数字多在 `nc_paper/NEW_ANALYSIS_RESULTS.md` 和 `docs/`。

### 2026-11-27 05:40
用户 11-27 08:30–12:15（+0800）线上改信，05:21 要求先把审稿人 1、2 的回复与审稿人 3 对齐（questions-14 转）。脚本 `nc_paper/docx_to_tex/pass_20261127/edits_1127_{revised,letter}.py`，对照表 `alignment_R1R2_R3.md`。稿件按她改过的引文同步（摘要删「balance … set by sea ice extent」句、A13 段、Heuzé 段去 by construction、2.4 节删「两种响应大小……correlation」段、A10 段 pack→ice、2.1 节 MLD 句删 70%/97%、2.3 节径流句截短），另把 LIG 冰缘「约 1°」标成环极平均，state 名统一大写。R1「模式局限」改成与 R3 同一套论证（冰期没有开阔洋对流，旧的「同一偏差在差值里抵消」删去），R1 第 23 条与 R2 brine 条补 A13 结论，R2 不再说 SAM 节缩短（2.4 节约 1,430 词，投稿版约 730）。图 R3（`fig_seaice_obs/`）按她的要求改成 JJA 两格。她两问的答案：A13 阈值是每态 0.02 细格曲线上转换率掉到峰值 10% 以下的等密线（PI 36.93）；Fig. 2a 在 37 处读到约 20 Sv 是 0.2 粗格把 36.89→36.93 的陡降平均进去、再连线的效果，细格穿过 37.0 只有 2.5 Sv。Pellichero 上层转换是 22 ± 4 Sv（信里原写 ±5）。TexPage 9635651（两份 tex）、d4d9c79（图）；本地编译修订稿 49 页、信 41 页，0 error。

### 2026-11-26 11:00
用户 08:48–10:46 线上改信并留批注（TexPage 33aed08 两个 tex，666f9e1、656f1bb、19326af 图；脚本 `nc_paper/docx_to_tex/pass_20261126c/`）。西风机制整段从 L144 挪到审稿人 3「westerlies 为何都增强」那条，平衡只留一段：SST 趋势（四态 ±0.03、MIS3 +0.06 K/百年）与 AMOC 趋势（仅 PI 显著），引 PMIP4 判据 Kageyama et al. 2018 §4.1（web 读原文、Crossref 核 DOI）。Methods 改回投稿版加这三句，删掉 2 m 气温与对比段。结果 2.1 节 Gray/PMIP 段删除（与讨论重复），讨论 Gray 段按她改的引文同步并补 Kageyama2021 出处。A9/R15 去掉散点 panel c 与 r = 0.77。R13 去大标题。答了 4.45/4.48 Sv 的来历（JJA、σ2 36.93、陆架；与 Fig. 2 绿线、R13 年均不是同一量）和 MOC 南极边浅环（data_moc/CLAUDE.md）。她改的「The relevant contents ... reads」语法不对，批注意思没把握，留给她定。稿 50 页、信 41 页，0 错误。

### 2026-11-26 08:35
用户 07:12–08:22 线上批注一轮（TexPage f0fddf6、a9541a6、6a8cc08、8cabb9a；脚本 `nc_paper/docx_to_tex/pass_20261126b/`）。审稿人 3 主意见回复按她的意思改成「ORAS5 的南极附近深混合层多在沿岸，模式沿岸少而浅，大部分在 Weddell 开阔洋」，数字新算（`explore/mld_coastal_oras5/`，原始网格，65°S 以南 MLD>200 m 的面积 ORAS5 87% 在离岸 200 km 内，模式 6%）。核数时发现 R2/R11 图注和审稿人 1 回复里「最深都在大西洋扇区、都向南极边缘加深」对 ORAS5 不成立（ORAS5 最深在 80–120°W 的 60–64°S 和沿岸），连同稿件讨论 L251「same large-scale pattern」一起改掉。A12/R2/R11 色标在 0–200 m 加密（BoundaryNorm）。泛化性回复加了「最密的水（σ2≥36.93）61% 在陆架、海冰项 4.45/4.48 Sv」和「中性密度只适用于现代海洋」；没写「mode water 由开阔洋热力形成」，因为 σ2 下 PI 峰值类与模式底层水同密度。讨论那段按她改的引文删掉开头两句和末尾三句，「move towards more brine-driven」改成「become more brine-driven」。A13 换成无冰缘线版（GC）。MOC 两问答在批注里。稿 51 页、信 42 页，0 错误。

### 2026-11-26 04:50 用户 11-26 线上批注一轮（questions-14 转 04:12「回复信我又更新了，请处理」）

- 拉下 38 个 TeXPage 提交（11-25 17:12 到 11-26 11:10 +0800），只改 response_new.tex：语法与措辞、把 GLORYS 意见拆成单独一条、注释掉几段，另写约 20 条 `%` 要求。按她 11-25 09:08 的授权直接改，改动在 `nc_paper/docx_to_tex/pass_20261126/`（`edits_1126_revised.py`、`edits_1126_letter.py`、`mhlig.py`、`notbut_1126.json`，可在 bb5f6eb 版上复现，md5 已对）。
- 新分析（各有文件夹和 CLAUDE.md）：密度类形成位置 `fig_formation_location/`（新 SI Fig. A13，讨论加一段；间冰期峰值类 69% 在无冰深水、最密尾部 61–88% 在陆架，冰期达到底层水密度的 0.7–1.0 Sv 全在陆架）；西风增强机制 `explore/westerly_mechanism/`（冰缘北移使近地面温度梯度变陡但仍在西风以南 10–12°，Eady 增 21–28%；LIG 是轨道强迫加热热带高空）；PMIP4 四个 LGM 模式 `explore/pmip_westerlies/`（位移 −0.6 到 +1.1°，我们偏极 3.7° 在其外，稿件原句 shared across the PMIP ensemble 改掉）；SST/AMOC 漂移与南极径流 `explore/drift_surface_amoc/`；A10 海冰形状 `explore/sam_windice_check/`（原「近岸减少、北边增加」偶极子不成立，图注与正文改）。
- 改正：push/pull 段「冰期 Ekman 响应最弱，因海冰削弱应力传递」无依据，删去（MIS3 10.1 > MH 8.9 Sv）；补 MH brine 12 mSv；Methods 的 two orders of magnitude 改 more than an order of magnitude；2.3 节和 Methods 补冰盖降水减蒸发以径流入海（PI 0.085 Sv）。A2 重画（a–c MLD 色标到 1000 m，海冰行不变）。正文与图注 MH、LIG 改全称（只在图、表 1、定义处留简称）。用户新规矩「不用 不是…而是 / rather than」：修订稿 31 处、信 29 处改写，建议写进 my_style（`docs/proposal_my_style_no_contrast.md`，交总管）。
- TexPage b30db0e（两份 tex）、4f07be5–89af427（5 张图各一提交）。本地编译修订稿 51 页、信 42 页，0 error；引文核对只剩原有 6 处写法差别。

### 2026-11-25 09:40 信与修订稿按用户行内要求和总管底线一次改完（questions-14 转用户 09:08「直接更新，不用再问」）

- 修订稿（`nc_paper/docx_to_tex/pass_20261125/edits_1125c_revised.py`，可在备份上复现）：SAM 定义与作用并进引言单独成段，补 Gong & Wang 1999、Hall & Visbeck 2002、Sen Gupta & England 2006（Crossref 核 DOI、读摘要），2.4 节开头两句删去、阈值只留在图注；2.1 节加 Fig. 1i–l 描述（不含西威德尔那句），114 行只谈 Ross 与方向，风的描述按 6–9 月风场核过后改写；冰期风应力海洋格点重算（MH 47.6°S 0.172、LGM 49.4°S 0.195 N m⁻²，强 13%），jet 改称 westerlies；Fig. 3/4 单位 kg m⁻² month⁻¹，2.3 节数换算（×2.592e6）；A3/A4 换成密度通量版（现 A7/A8，±2 同一色标，加 −15 W m⁻² 线），2.4 节逐处加子图字母并写出海区；Sidorenko 段换成信里版本（删 1.43 K），新 SI A12（MLD 对 ORAS5，引 Zuo 2019）；239 行 69→61% 改成分密度类结果，249 行 53–88% 改 53–70%；Lhardy 2021 重建 33±5、10±3 百万 km² 写进 255 行并删掉重复句；polynya 措辞；Methods 符号约定统一为 B>0 致密化；Fig. 1、A1、A2 图注改 June-September；附录按首次引用重排为 A1–A12，正文 MOC/年龄互换。
- 回复信（`edits_1125c_letter.py`）：27 段引文按句与修订稿对齐（`check_quotes_vs_revised.py`），图号随重排改；第 11、13、20、21、22 条、R2 的 Figs 5/6（R22 换成 A7/A8 两图）、R3 普适性（压短版）、Gray（风应力数）、MOC 数（D6/D8）、L25、L102 照用户 `%` 意见改并在其下写 `%claude回复`。剩两条 `\claudenote`（EE1 的 PI 冰区、批注 8）。
- TexPage da2a469（三份 tex）、4d59b5f–3a40d9f（9 张图各一提交）、bea896b（符号约定）；3 分钟后核 md5 全对，期间无 TeXPage 提交。两份都本地编译 0 error，修订稿 49 页、信 36 页。

### 2026-11-25 05:30 用户在线改信后的一轮（总管 questions-14 转，05:08 消息）

- 拉下 11-24、11-25 的 28 个 TeXPage 提交：改了第 4、9、11、13 条回复，删 R6（Ross 图，注释掉）和 Editorial requirements 一节，重写 R13 图注，另在源文件写了十来条 `%` 要求；revised.tex 155 行删掉 A3/A4 用 1.0σ 的半句。追踪版取消（用户 05:10）。
- 修：R13 图注全角括号（唯一的编译错误）和「zone(c)」；R13 子图改 a–d（`plot_wmt_merged_3rows.py rean`，旧图存 `figures/versions/figR20b_wmt_oras5_v20261122.*`）。核对结果写成 13 条 `\claudenote`，她的问题写成 13 条 `%claude回复`（LGM 海冰范围 JAS 29.8、JFM 11.1 百万 km²；Gray 2023 是地表风应力，我们 LGM 相对 MH 偏极约 1.9°、强约 15%；polynya circulation 措辞；SAM 候选文献待核）。第 13 条按她要求加了修订稿 114 行原文，并标出其中仍是旧 PI 文件的数。TexPage 4e75934、ae8bfa1。
- 发现：她给 Fig. 1i–l 的新描述「MH、LIG 西威德尔海冰减少」只对被覆盖的 PI 文件成立；`plot_climate_patterns_jja.py` 的 JJA 实为 6–9 月。R3 海冰图的原因与候选见 `fig_seaice_obs/CLAUDE.md`。

### 2026-11-23 03:45 回复信两份 latexdiff 追踪版（总管 questions-14 转用户 03:28 手机消息）

- 主版对比 Word 刚转成的 tex（TexPage 5bf3ef0）与线上现版（1fce35c）；副版旧稿取 e45004e（11-22 我们改之前、含用户当天线上改动）。脚本 `nc_paper/docx_to_tex/pass_20261123/make_diff.sh`，各 39 页，pdflatex 三遍 0 error、无未定义引用；用旧/新 PDF 的词频对查，没有丢内容。
- 主版约 120 处改动，删约 1600 词、加约 4100 词；副版只差用户自己那两处。TexPage 5971c26（两份 tex）、50384c0、e32d86b（两份 PDF），response_new.tex 没动。

### 2026-11-22 07:25 response_new.tex 黄色高亮逐条处理（总管 questions-14 派活，TexPage b90b273 起 9 个提交）

- 用户 06:41 停笔后，在干净克隆上用 `nc_paper/docx_to_tex/pass_20261122/edits_1122.py`（每个锚点断言恰好一处）改完全部 40 来处高亮，看不懂的词改成通俗说法，答非所问的段落按审稿人原问题重写；仍需用户定的 4 处保留高亮，并加品红色 `\claudenote`。
- 新分析：海冰与卫星对比 `fig_seaice_obs/`（OSI SAF CDR v3.0，1979–2008；PI 9 月冰范围 9.9 对 19.1 百万 km²，缺口几乎全在 Weddell 扇区）；SAM 合成换成同一密度通量单位 `fig_sam_buoyancy/`（冰区淡水项在 90–93% 面积上更大）；审稿人 3 的 WMT 图拆成观测（a,b）和再分析（c–f）两张；Ross 图只留两扇区；Fig. 3/4 色标合并到最下一行。
- 核实：Sidorenko et al. 2019 全文（1.43 K 是全球 SST RMSE，不是南大洋偏差），西风急流数（JJA var165 纬圈平均最大值，完全复现），Ross 扇区、域敏感性、ORAS5 峰值各数。改正：间冰期海冰区热占比 53–70%（不是 53–88%），冬季两条曲线峰值差一个密度类。
- 新发现的三处稿件问题见 `log/OPEN.md`：PI 海冰气候态文件被覆盖、Fig. 4 淡水项少乘水密度、Figs A3/A4 实为 1.0σ。
- 07:44 用户 EE1=先不动、EE2=改、EE3=图注改 1.0σ：Fig. 3/4 重出（淡水项乘回水密度，单位每米水柱），修订稿 138/149/303/305 行与 A3、A4 图注、155 行改好，信里 R10 换成 1.0σ 版（TexPage 8ef54c4–b647ba5；改稿脚本 `nc_paper/docx_to_tex/pass_20261122/edits_revised_ee23.py`）。

### 2026-11-19 08:35 回复信改写成 response_new.tex（总管 questions-14 派活）

- 用户 11-19 邮件：合作者改过的 `paper/inbox_20261119/response-JL.docx` 为准，按 texpage-sync 的 STYLE 写成 `revision/response_new.tex`，旧 04/05 不动。`nc_paper/docx_to_tex/` 拆 run、生成 tex、PDF 逐段核对：325 段有字的 docx 段落全部在编译出的 PDF 里；8 处删除线删掉；39 段高亮保留为 ulem `\hl`；丢字只补了 2 段 3 处（Berger 1977 引用、`$-\left(\alpha/C_p\right)Q_{net}$`、`$\beta S F_{fw}$`，其实是旧信生成时就丢的）；9 条 Word 批注印成红色注。图 R1–R15 连续编号，重复图因题注不同各放一次。34 页，本地 pdflatex 两遍 0 error。
- TexPage 5bf3ef0，只推了这一个文件，3.5 分钟后回读 md5 一致、无新的 TeXPage 提交、引用的 11 张图都在。
- docx 来自 D6 之前的旧信：MOC 数字和措辞是旧的，照转并在源里 `% NOTE`；3 张图 docx 里是旧版，tex 用线上现版。

### 2026-11-08 10:40 D8 上层环流指数改为 20–70°N（数字待总管核对）

- 查证：`plot_moc_5exps.py` 从 758dfa6（09-08）起就是 0–70°N，投稿稿 `nc-main.tex` 没有 MOC 分析，修订稿正文、图注、Methods、两封信都没写过上层窗口，唯一的「0–70N」是 D6 时照抄代码写进 `NEW_ANALYSIS_RESULTS.md` 的。无文字依据，按用户说的 20–70°N 改。
- NADW（500–4000 m、20–70°N）PI/MH/LIG/LGM/MIS3 = 18.5/19.8/21.6/20.1/21.5 Sv（0–70°N 时 19.5/20.4/22.1/20.9/21.5）。PI、MH、LIG 仍卡在 580 m 顶边和 21°N 南边；1000 m 以下参照值 16.3/16.6/17.4/19.9/21.5，全在 1040 m 顶边。
- 改了 `plot_moc_5exps.py`（及副本）、`data_moc/calc_moc_5exps.py`（重跑 job 28821814）、Source Data、`revised.tex` 2.4 节数字，新增 Methods 4.4 Overturning streamfunction 写明两个窗口；`RESPONSE_LETTER.md` R3 那段的数字。图只标 AABW，PNG 字节不变，没换。「upper cell … changes comparatively little」相对 −80–86% 仍成立（+8%、+16%），未改写。
- 10:21 用户选「改措辞并推 TexPage」：2.4 节改为「The upper cell does not weaken and strengthens moderately, from 18.5 Sv in PI to 20.1 Sv in LGM and 21.5 Sv in MIS3」，删掉 associated with northern-sourced deep water；信里 Fig R2.5、R3.4 图注统一为 is maintained，R3 回复跟正文措辞一致；Source Data 列名 NADW_cell_Sv 改为 upper_cell_Sv（文字从不说 NADW/AMOC）。GitHub 35e4950，TexPage c24777a，回读一致。

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
