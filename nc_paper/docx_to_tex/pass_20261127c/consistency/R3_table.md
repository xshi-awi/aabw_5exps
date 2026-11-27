# 审稿人 3：回复信 vs 修订稿核对（response_new.tex 562–884 行，27 条 \comment{R3}）

行号：信 = response_new.tex，稿 = revised.tex，原 = original.tex 或原投稿 PDF 的行号。图号按 revised.aux 核对：moc 5、age 6、age_global 7、domain 8、schematic 9、mld A2、reso A3、s1 A4、sam_heat A7、sam_fwf A8、sam_pushpull A9、sam_windice A10、mld_obs A12、formation_loc A13。
多面板图（稿 PDF 第 29–50 页）逐张看过。核对时找了 git 记录和数据源 sam_push_pull.json（只读，没有改任何文件）。

| # | 信（审稿人 3 第几条，承诺摘要） | 修订稿位置 | 状态 | 备注 |
|---|---|---|---|---|
| 1 | R3-1（信 564）总评，致谢 | — | 一致 | 无承诺 |
| 2a | R3-2（信 579）冬季 MLD 对比 ORAS5，"now Supplementary Fig. A12" | Fig. A12（mld_obs）；讨论稿 251 "Compared with the ORAS5 reanalysis…" | 一致 | 原稿没有 ORAS5，是新增 |
| 2b | R3-2（信 581）承认模式偏差，引 Orsi 1999 和 Toggweiler & Samuels 1995 | 讨论稿 247 "What differs is where dense water is made…" | 一致 | 新增 |
| 2c | R3-2（信 581）"Heuzé (2021) found that almost all CMIP6 models…" | 讨论稿 253 "…28 of 35 models form deep and bottom water…" | 不一致 | 稿件写的是 35 个里有 28 个，信说 almost all，说过头了 |
| 2d | R3-2（信 585–589）新图 A13；68–71%、16–25%、3.0–7.4 Sv、61–88%、87–90%、53–59% | 讨论稿 249 "To see where each density class is produced…"；Fig. A13 | 一致 | 数字与稿件一致（0.1 格版本） |
| 2e | R3-2 三段引文："The new Discussion paragraph"、"Another text added to the Discussion"、"Further on" | 稿 249、247、253，都在讨论 | 一致 | quote_check #29 的 MISSING 只是 \cite 写法不同 |
| 2f | R3-2（信 585）峰值类 = "the 0.1 kg m⁻³ bin of manuscript Fig. 2" | Fig. 2 图注 "in density bins 0.1 kg m⁻³ wide"；方法稿 324 | 一致 | 原稿方法是 0.2；11-27 改成 0.1（git c456e8b） |
| 3a | R3-3（信 614）按 Pellichero 的域重算 | 讨论稿 242 "Recomputing the transformation over a sea ice sector defined exactly as theirs…"；Fig. 8；方法稿 334 | 一致 | 新增，信里 R3-23 指向 Fig. 8 也对 |
| 3b | R3-3（信 621、637）−9.2/+6.6 Sv、γ 27.1–27.67、ORAS5 +5.0/+4.4、+62.9/+36.8 Sv | 只在信里，稿件没有 | 一致 | 没有可对照的稿件数 |
| 3c | R3-3（信 621、630）Pellichero 的 5±5 Sv | 讨论稿 240 | 原稿已有 | 原稿 294 行已有 5±5 |
| 3d | R3-3（信 627）"This is a shared model bias in most climate models." | 无 | 缺失 | 没有出处，稿件里也没有这句（信里 % 注已提过） |
| 4a | R3-4（信 643）"we now say so in the Discussion"（开阔洋对流是模式偏差） | 讨论稿 247、253 | 一致 | 新增 |
| 4b | R3-4（信 643）LGM 冬季冰范围在重建范围内（见给 R2 的回复） | 讨论稿 261 "29.8 million km²…33±5"；信 530 R2 引文 | 一致 | 交叉引用对 |
| 4c | R3-4（信 643、657）海冰区内冰期 16%/19%，间冰期 54–83% | 讨论稿 255；Fig. 8 f–j 格下 76/54/83/16/19% | 一致 | |
| 4d | R3-4（信 645）PI 最密类 7.4 Sv、σ₂ 36.93、陆架 61%（4.5 Sv）、200 km 内冰 < 0.7 | 讨论稿 249；A13 a、d、e | 一致 | |
| 4e | R3-4（信 645）"manuscript Fig. 2, now drawn with density bins 0.1 kg m⁻³ wide" | Fig. 2 图注；方法稿 324 | 一致 | 确是改动（原 0.2） |
| 4f | R3-4（信 657）"The relevant text in the Discussion reads" | 讨论稿 255 | 一致 | |
| 4g | R3-4（信 643）回复正文引 "Green et al., 2022" | 稿件里已删这条引用（见信 532 的 % 注） | 一致 | 信里单独引用可以，但信里要给全出处，或者删掉 |
| 5a | R3-5（信 668 R4 图注，682 引文）A9：Ekman 8.0–12.8 Sv；brine "51, 12 and 42 mSv" | 2.4 节稿 177；Fig. A9 | 不一致 | A9(b) 柱上标的是 41；sam_push_pull.json 里 LIG 是 41.47 mSv，应四舍五入成 41。稿 177、信 668、信 682 三处都写 42 |
| 5b | R3-5（信 673）各态 Ekman 11.2、8.9、12.8、8.0、10.1 Sv | A9(a) 柱标 | 一致 | json：11.19、8.89、12.81、8.04、10.11 |
| 5c | R3-5（信 673）冰下淡水项占优面积 90–94%（A7、A8） | 2.4 节稿 183 | 一致 | |
| 5d | R3-5（信 677、686）新图 A10 和引文 | 2.4 节稿 179；A10 | 一致 | |
| 5e | R3-5（信 679）"The new text in Section 2.4 reads" | 2.4 节（稿 156 起） | 一致 | 原稿没有 Ekman at 60°S，是新增 |
| 6a | R3-6（信 692）PI 45.7°S 5.83 m/s；LGM、MIS3 49.4°S 6.30/6.85；向极 3.7° | 2.1 节稿 120；讨论稿 263 | 一致 | |
| 6b | R3-6（信 692）风应力 MH 47.6°S 0.172，LGM 49.4°S 0.195，"1.9° further south"，约 13% | 只在信里 | 不一致 | 用信里给的两个数算，49.4 − 47.6 = 1.8°；13% 对 |
| 6c | R3-6（信 694）四个 PMIP4 模式 −0.6 到 +1.1° | 讨论稿 263 "less than 1.2°" | 一致 | |
| 6d | R3-6（信 702）"The Discussion now reads" | 讨论稿 263 | 一致 | quote_check #33 只是 \citet 写法不同；原稿没有 Gray 2023 |
| 7a | R3-7（信 718–721）"added a short explanation to Section 2.1" | 2.1 节稿 120 "In the glacial states the change follows…" | 一致 | 新增 |
| 7b | R3-7（信 712–714）66°S→55–57°S、65→60–62°S、24–29%、10–12°、Eady 21–28%、26–33 W m⁻²、约 1° | 稿 120 | 一致 | 信里 2.3 K、稿里 about 2 K，不矛盾 |
| 7c | R3-7（信 726–729）SST 和 AMOC 趋势，Kageyama et al. 2018，"The Methods now state" | 方法稿 296 | 一致 | 新增；bib 键 Kageyama2017 的 year 是 2018（PMIP4 Part 1） |
| 7d | R3-7（信 726）PI 已积分 1500 年 | 方法稿 296 | 原稿已有 | 原稿就有 1,500 years |
| 8a | R3-8（信 737）新图 Fig. 5 MOC；−11.7/−11.3/−11.4 → −2.3/−1.7，80–86%，18.5→20.1/21.5 | 2.6 节稿 218；Fig. 5 | 一致 | 新增 |
| 8b | R3-8（信 744–751）新图 Fig. 7 全球年龄；377→439/508、381→789/844、847→1291/1314；"text added to the Results" | 2.6 节稿 225；Fig. 7 | 一致 | 新增 |
| 8c | R3-8 审稿人要求澄清引言 L111–112（理想年龄当翻转代理） | 引言稿 94 与原稿 90 一字未改："…assessed as an integral diagnostic of overturning efficiency. This analysis serves as an integral proxy…" | 缺失 | 信里没回应这一点，引言也没改 |
| 8d | R3-8 Millet 2025（年龄未平衡）和混合影响 | 2.6 节稿 227 "We caution that ideal age…lower bound…"；讨论稿 267 两条 caveat | 一致 | 稿件做了（新增），但信里只写了 "We agree and have added the analyses."，没提这两点 |
| 9 | R3-9（信 754）"Minor comments" 小标题 | — | 一致 | 无回复，也不需要 |
| 10 | R3-10（信 758）MH、LIG 在正文、图注、方法写全称，缩写只留在面板和 Table 1 | 结果稿 105 定义；方法稿 294；图注都是全称；Table 1 用缩写 | 一致 | 正文非注释行里的 MH/LIG 只出现在两处定义 |
| 11 | R3-11（信 762）摘要 "In the model, interglacial dense water formation is driven mainly by turbulent heat loss…"，结果也加了 | 摘要稿 41；结果稿 128 "In our model…" | 一致 | 原稿 L19 没有 |
| 12 | R3-12（信 766）删掉 "clear imprint on deep-ocean ventilation"，摘要改为 abyssal overturning cell 和 abyssal water；Figs. 5、7 | 摘要稿 41 | 一致 | "clear imprint" 原 1 处、稿 0 处 |
| 13 | R3-13（信 772–775）引言扩写 SAM 段（与 R1 第 5 条同一段引文） | 引言稿 80 | 一致 | 信 248 在 R1 第 5 条下，交叉引用对 |
| 14 | R3-14（信 780）改 typo "moer" | 结果稿 109 "more than 2 °C" | 一致 | |
| 15 | R3-15（信 784）说明冰期盐度异常含全球均匀增量 | 结果稿 110 "Because that offset is applied uniformly…only their regional structure reflects local dynamics." | 一致 | 新增。审稿人要的是扣掉增量的图，信只加了说明，见「需要作者定」 |
| 16 | R3-16（信 789–792）A2 间冰期色标到 1000 m，冰期保持 400 m；正文给 >600 m 面积和最深 991 m；引文 | A2 图注；结果稿 115 | 一致 | 新增。同段 "polynyas (Fig. \ref{reso})" 指向 A3 网格图，原稿指 Fig. A2d,e，见「文字小修」 |
| 17 | R3-17（信 800）见上，2.1 节有说明 | 稿 120 | 一致 | |
| 18 | R3-18（信 804–807）"In our model, dense water formation is predominantly thermally driven…" | 结果稿 128 | 一致 | 原 L152 是 "During interglacial periods…" |
| 19 | R3-19（信 818）峰值在 "σ₂ ≈ 36.5--36.8"、γn ≈ 27.3；两段引文 | 结果稿 128（σ₂ ≈ 36.4–36.8）；讨论稿 245 | 不一致 | 信里是 0.2 格时的旧值，稿件和信 589 已是 36.4–36.8；两段引文本身对 |
| 20 | R3-20（信 834–839）冰盖融水，没有冰架下空腔，引文 | 2.3 节稿 151 | 一致 | quote_check #42 的 MISSING 是引文里夹了 % 注 |
| 21 | R3-21（信 844）SAM 定义挪到引言 | 引言稿 80；2.4 节稿 159 "introduced above" | 一致 | 原 L206 "dominant mode of atmospheric variability" 已删 |
| 22 | R3-22（信 848）"addressed at length in the above" | 讨论稿 240–255 | 一致 | |
| 23 | R3-23（信 852）"See the above response and Fig. 8" | Fig. 8 = domain；讨论稿 242 | 一致 | |
| 24 | R3-24（信 856–859）讨论不再建议用沉积物重建，引文 | 讨论稿 265 | 一致 | 原 L337–341 的建议已删。信说 "Reviewer 1 has the same concern"，但 R1 第 29 条要的是参考文献 |
| 25 | R3-25（信 865–868）改成 "radiocarbon age of dissolved inorganic carbon" | 讨论稿 267 | 一致 | 原 L346 "two times older" 已删 |
| 26 | R3-26（信 873–876）没有冰架下空腔，方法和讨论都写了，列为局限，引文 | 方法稿 277、294；讨论稿 253 | 一致 | 新增 |
| 27 | R3-27（信 881）多面板图都在每列上方标气候态 | Figs. 1–8、A1、A2、A4–A13 逐张看过，都有 | 一致 | A3 网格图和 Fig. 9 示意图是单幅，不需要 |

状态计数（54 行）：一致 46，不一致 4（2c、5a、6b、19），缺失 2（3d、8c），原稿已有 2（3c、7d）。

quote_check 里属于 R3 的 MISSING 共四条：#29、#33（\cite、\citet 写法不同），#38、#42（引文里夹了 % 注）。逐条看过，都不是真的不符。

## 文字小修

1. response_new.tex 581：`Heuzé (2021) found that almost all CMIP6 models` → `Heuzé (2021) found that 28 of 35 CMIP6 models`
2. LIG 的 brine 响应 42 → 41（json 里是 41.47 mSv，A9 图上标 41）。三处：
   - revised.tex 177：`by 51, 12 and 42~mSv` → `by 51, 12 and 41~mSv`
   - response_new.tex 668（R4 图注）：`by 51, 12 and 42 mSv` → `by 51, 12 and 41 mSv`
   - response_new.tex 682（引文）：`by 51, 12 and 42~mSv` → `by 51, 12 and 41~mSv`
   - docs/sam_push_pull.md 也写的 42，顺手改。
3. response_new.tex 692：`1.9° further south` → `1.8° further south`。如果未取整的纬度差确实是 1.9，就保留，但把两个纬度写到两位小数，免得读者自己算出 1.8。
4. response_new.tex 818：`σ\textsubscript{2} ≈ 36.5--36.8 kg m\textsuperscript{-3}` → `σ\textsubscript{2} ≈ 36.4--36.8 kg m\textsuperscript{-3}`
5. response_new.tex 856（可选）：`Reviewer 1 has the same concern.` → `Reviewer 1 also commented on this sentence (comment 29).`
6. revised.tex 115：`polynyas (Fig. \ref{reso})` → `polynyas (Fig. \ref{mld}d,e)`。现在指向的是 A3 网格图，原稿指的是 MLD 图的 d、e 两格。
7. 顺带发现，不属于 R3 承诺。revised.tex 456（Fig. 9 示意图图注）还是 0.2 格的旧数：
   - `36.5$--$36.8$~kg~m$^{-3}$ (60--80~Sv)` → `36.4$--$36.8$~kg~m$^{-3}$ (70--81~Sv)`
   - `(48--58~Sv)` → `(50--59~Sv)`
   - 示意图 frame.pdf 里如果也画了这些数，要一起改。
8. 顺带发现。revised.tex 227：`Each experiment was integrated for 1000 years` → `Each paleoclimate experiment was integrated for 1000 years`（方法里 PI 是 1500 年）。

## 需要作者定

1. **R3-8 引言 L111–112**：审稿人要求澄清"拿理想年龄当翻转代理"，信没回应，引言稿 94 原样保留 "an integral proxy for evaluating the efficiency of the overturning circulation"。建议引言改成"翻转直接用流函数诊断（Fig. 5），理想年龄作为补充、它也受混合影响"，并在信里加一句说明。
2. **R3-8 Millet 2025 和混合**：稿件已加 caveat（稿 227、267），但信里只有 "We agree and have added the analyses."。建议在信里补一段，引用稿 227 那句。注意按 nc_paper 的写作红线，Millet 的 factor of 2 讲的是 filling time，要委婉处理。
3. **信 627 "This is a shared model bias in most climate models."**：没有出处，稿件里也没有。可选：删掉；改成 "many models" 并引 Heuzé 2021（但那篇说的是底层水偏淡，不是表层）；或者找到合适的文献再引。
4. **R3-15（L129）**：审稿人要的是扣掉全球均匀盐度增量后的图，信说 "good suggestion"，但只加了一句文字说明。可选：照现在这样；在信里或方法里给出增量的数值；或者补一张扣掉增量的图。
5. **信 643 引 Green et al., 2022**：稿件已删这条引用。可选：信里保留，但给出全出处；或者删掉，只留 Lhardy et al., 2021。
