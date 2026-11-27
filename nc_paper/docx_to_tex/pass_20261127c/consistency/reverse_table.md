# 反向核对：修订稿的实质改动，信里交代了没有（2026-11-27）

依据 latexdiff.tex（逐段压成 `[-删-]{+增+}` 后逐段读）、original.tex、revised.tex、revised.aux、response_new.tex（只看回复正文与引文，% 行不算）。latexdiff 没标摘要（在 `\abstract{}` 参数里），摘要另做逐词 diff。
图号按 revised.aux：正文 Fig. 1–9（示意图为 Fig. 9），附录 A1–A13，Table 1。
「信里」一列：是（写明回复条目）/ 部分（提到了但没交代改动本身或数字）/ 否；(Q) 表示只出现在引用的稿件原文里。

| # | revised.tex 改动（位置，开头词 / label） | 原稿有？ | 信里交代？（位置） | 备注 |
|---|---|---|---|---|
| 1 | 摘要：加 "In the model, interglacial dense water formation is driven mainly by…" | 否 | 是 R3-L19 | |
| 2 | 摘要：SAM "3–8 Sv during positive phases" → "6–10 Sv between its positive and negative phases… through heat loss… through brine rejection" | 改 | 部分：新句引在 R1-2/R1-3，没说 3–8 改成 6–10 | 同 #33 |
| 3 | 摘要：加 "collapse of the simulated abyssal overturning cell from about 11 to 2 Sv" | 否 | 是 R1-4, R3-L25 | 正文 −11.3…−11.7 → −1.7…−2.3 |
| 4 | 摘要：删 "This directly links… enhanced glacial carbon storage" 和 "in coastal polynyas" | 原有（删） | 否 | R3-L25 只说删了 deep-ocean ventilation 的提法 |
| 5 | 摘要末句：加 "…the simulated pre-industrial state transforms water at lighter densities than observations indicate" | 否 | 否（R1-gen、R3-2 讲了 lighter densities，没说摘要已写入） | 摘要结论改动 |
| 6 | 引言 ¶1：删 "This rate is in turn governed by surface buoyancy forcing…" 及 sime2013、chadwick2022、volker2013、Ferrari2014、mandal2021 | 原有（删） | 否 | 删减，可不提 |
| 7 | 引言 ¶3：sea ice pump 改述为 "removes freshwater… through equatorward ice transport and melting" | 改 | 否 | 定义更正，可不提 |
| 8 | 引言 ¶4 与讨论：删 knorr2021salty（两处） | 原有（删） | 否 | 删引用，作者定是否需说明 |
| 9 | 引言 ¶4：加 Chen et al. (2025) 段 | 否 | 是 R2-major2 | |
| 10 | 引言 ¶5：SAM 定义与作用（Gong1999、Thompson2000、Marshall2003、HallVisbeck2002、SenGupta2006、Cerovecki2013、Zhou2023、Silvano2020），由 2.4 节移来 | 改+移 | 是 R1-5, R1-20, R3-L102-105, R3-L206-209 | |
| 11 | 2.1 新段 "The figures label the five simulated climate states…"，全称约定，指向 Table 1 | 否 | 是 R1-1, R2-acronyms, R3-#65 | |
| 12 | 2.1 "The maps in Fig. 1 are averaged over June to September"；Fig. 1、A1、A2 图注由 JJA 改 June-September | 改（定义更正） | 部分：R1-11、R3-L134 顺带写 June to September，没交代原图注写错 | 见重要改动 G |
| 13 | 2.1 冰期盐度含全球均匀抬升 "Because that offset is applied uniformly…" | 否 | 是 R3-L129 | |
| 14 | 2.1 新段 "In the glacial states the surface density anomalies… follow the salinity field…" | 否 | 否 | 指针错：`(Fig. \ref{sst}i-l)`、`(Fig. \ref{sst}j)` 指向 Fig. 1 海冰面板，密度在 Fig. A1a-d（`\ref{wind}`）|
| 15 | 2.1 新段 "The sea ice anomalies (Fig. 1i-l) follow the temperature changes…"（LIG Ross 退约 5°，冰期北扩 10–15°） | 否 | 是 R1-11 | |
| 16 | 2.1 MLD："are observed" → "the model produces"；PI 对流集中 Weddell；>400 m 面积 7.5→1.7/1.6×10¹¹ m²、最大 991→819/645 m、>600 m 消失；引新 A4 | 改 | 是 R1-9, R1-10, R1-23, R3-L134 | 指针：`coastal bands and polynyas (Fig. \ref{reso})` 指 A3 网格分辨率图，原稿是 A2d,e，请核 |
| 17 | 2.1 纬向风异常改写（MH 50°S 以南为负；LIG 太平洋增强；LGM 大西洋-印度洋负异常） | 改 | 部分 R1-12 | |
| 18 | 2.1 新段：各扇区冰点与海冰，LIG Ross 不对称 | 否 | 是 R1-13 | |
| 19 | 2.1 新段：西风增强机制（45.7°S 5.83 m/s、Eady +21–28%、LIG 轨道强迫） | 否 | 是 R3-6, R3-L144 | |
| 20 | 2.2 冬季极大 60–80 → 70–81 Sv，σ₂ 36.5 → 36.4–36.8；热 63–90%、haline 10–37%（原 "sea ice 15–30%"） | 改（0.1 bin） | 部分：新数字在 R1-14、R3-1，没说 bin 改了、旧数字为何变 | 见 A |
| 21 | 2.2 新句：热主导极大在中层/模态水密度；σ₂≥37.0 haline 主导（0.44 vs 0.04 Sv） | 否 | 是 R3-L175 | |
| 22 | 2.2 冰期极大 "below 60" → "50-59 Sv"，σ₂ = 37.1–37.6 | 改 | 部分：R1-27 写 "below 60 Sv… σ₂ = 37.0–37.5… thermal share 12–17%"，与稿件 37.1–37.6、17–19% 不符（R1-14 是 17–19%） | 信里旧数字要改 |
| 23 | 2.2 注明 Fig. 2 各格 y 轴独立，新 A5 共用轴 | 否 | 是 R1-15 | |
| 24 | Fig. 2 图注加 "in density bins 0.1 kg m⁻³ wide"，删线色说明 | 改 | 部分：R3-3 顺带 "now drawn with density bins 0.1" | 见 A |
| 25 | 2.3 与 Fig. 3/4 图注：单位 kg m⁻³ month⁻¹ → kg m⁻² month⁻¹（"surface density flux… 1 m thick"） | 改 | 部分：R1-17、R2-minor:fig5-6 说色标单位是 kg m⁻² month⁻¹，没说原稿单位印错 | 见 B |
| 26 | 2.3 数值：PI 热致增密 20 → 约 13；冰期热异常 10 → 约 −5；冰期淡水异常 0.02 → 约 +6.5（区域改 55–70°S 环带）；新增沿岸 >10、局地约 47 | 改 | 否 | 见 B |
| 27 | 2.3 加 "consistent with observationally based surface buoyancy budgets (Cerovecki2013, Abernathey2016)" | 否 | 否 | 小 |
| 28 | 2.3 间冰期异常 "small to modest… north of the ice-covered region" | 改 | 是 R1-18 | |
| 29 | 2.3 新段：冰盖融水（basal melt、iceberg）不在模式里，由径流代替 | 否 | 是 R3-L195, R3-#81 | |
| 30 | 2.4 合成阈值："±1.2σ" → "a fixed multiple… given in the figure caption"；A7/A8 用 ±1.0σ，A6/A9/A10 用 ±1.2σ | 改（方法） | 部分：R2-minor:fig5-6 的信图注写 1.0 SD，没解释为何与其他合成不同 | 见 C |
| 31 | 原 Figs. 5、6（SAM 热、淡水合成）移入附录 A7/A8 | 原有（移） | 是 R2-minor:SAM-short, R2-minor:fig5-6 | |
| 32 | A7/A8 改画为表面密度通量、同一色标，A7 加 15 W m⁻² 虚线 | 改 | 是 R2-minor:fig5-6, R1-22 | |
| 33 | 2.4 SAM-WMT 异常：PI 3.2→6.1、MH/LIG 约 8→8.2/10.4、LGM 4→10.2、MIS3 4.2→7.0 Sv；结论由 "magnitude… differ substantially"（间冰期最强）改为 "size… is similar in all climate states" | 改（结论） | 否 | 见 D |
| 34 | 2.4 面板指针（A6a-e、A7k-m 等） | 改 | 是 R1-21 | |
| 35 | 2.4 热通量合成区域改写（印度洋 60–140°E、太平洋 180°–120°W，冰缘约 55°S） | 改 | 是 R1-22 | |
| 36 | 2.4 brine 机制链 + Ohshima2013 | 否 | 是 R1-24 | |
| 37 | 2.4 push/pull（Ekman 8.0–12.8 Sv；brine 51/12/42/24/31 mSv），新 A9 | 否 | 是 R3-4 | |
| 38 | 2.4 空间型（冰面积 +0.07–0.57 百万 km²），新 A10 | 否 | 是 R3-4 | |
| 39 | 2.4 新段：共色标下冰区淡水项 90–94% 面积占优、大 6–10 倍 | 否 | 是 R2-minor:fig5-6, R3-4 | |
| 40 | 新小节 2.6 "Overturning circulation and deep ocean ventilation" + 新 Fig. 5（AABW −11.7/−11.3/−11.4 → −2.3/−1.7；上层环 18.5 → 20.1/21.5） | 否 | 是 R3-7 | |
| 41 | 2.6 新段：全球理想年龄 + 新 Fig. 7（2000 m 以下 377→439/508 等） | 否 | 是 R3-7 | |
| 42 | 2.6 新句："ideal age… should be read as a lower bound… integrated for 1000 years… (Millet2025)" | 否 | 否 | 见 E |
| 43 | 讨论：Pellichero 比较改写 "whereas our PI integral is thermally dominated… we examined it directly" | 改 | 是 R1-gen:model-eval, R1-26, R2-major1, R3-2 | |
| 44 | 讨论：积分域（42.7% 在 9 月冰缘外、89–91% 热主导；SIZ 内 79→76%、海冰项 2.6 Sv），新 Fig. 8 | 否 | 是 R1-26, R3-2, R3-L302-305；数字在 R1-gen(Q) | |
| 45 | 讨论："only one class… remains thermally dominated, σ₂ = 36.75–37.00" | 否 | 仅 R1-gen(Q) | 36.75–37.00 不是 0.1 bin 的边界，请核是否沿用旧分档 |
| 46 | 讨论：中性密度对照 γₙ ≈ 27.3 vs 27.9–28.8 | 否 | 是 R3-L175；R1-gen(Q) | |
| 47 | 讨论：排除冰区暖偏差（−1.08 °C、55% 在冰点 0.5 °C 内、α = 3.5×10⁻⁵） | 否 | 仅 R3-1(Q) | 可在 R2-major1 点一句 |
| 48 | 讨论：PI 深水在 68°S Weddell 开阔洋对流形成；Orsi1999、ToggweilerSamuels1995 | 否 | 是 R1-23, R3-1 | |
| 49 | 讨论：形成位置分析，新 A13 | 否 | 是 R3-1, R1-gen:limitations, R2-minor:brine | |
| 50 | 讨论：模式评估（Sidorenko 2019、ORAS5 MLD 新 A12、OSI SAF 冰范围约为观测一半、Heuzé 2021/2013、无冰架腔、"not a quantitative reconstruction"） | 否 | 是 R1-gen:limitations, R3-1 | |
| 51 | 讨论：冰区内的态转换（冰期 16/19%，间冰期 54–83%） | 否 | 是 R1-gen:limitations, R3-3 | |
| 52 | 讨论：Chen et al. 2025 一致性段 | 否 | 是 R2-major2 | |
| 53 | 讨论数字：70–81 Sv、36.4–36.8；冰期 "below 60 Sv"、37.1–37.6、热占比 17–19% | 改 | 部分：R1-14 对；R1-27 是旧数字 | 同 #22 |
| 54 | 讨论："Moreover, our paleoclimate analysis suggests that projected sea ice decline…" 一段 | 否（原稿为 % 注释） | 否 | 修订稿取消了注释；与紧随的 "These paleoclimate results carry direct implications…" 段重复（同引 Silvano2018、Li2023Nature），疑为误删 % |
| 55 | 讨论：LGM 海冰 29.8/11.1 百万 km² 在代用范围内（Lhardy2021） | 否 | 是 R2-minor:glac-conv, R3-3 | |
| 56 | 讨论：冰期西风与 Gray2023 相反，PMIP4 <1.2°，本模式极向 3.7° | 否 | 是 R3-5, R1-gen:interp | |
| 57 | 讨论 SAM 段：3–8 → 6–10 Sv；push/pull；Ross 例外；"does not separate the interglacial from the glacial states"；删代用指标重建 | 改 | 部分：Ross 与代用指标见 R1-gen:interp、R1-28、R1-29、R3-L339；数字与 "不区分" 的结论没交代 | 同 #33 |
| 58 | 讨论：Skinner2010 改为溶解无机碳的放射性碳年龄 | 改 | 是 R3-L347 | |
| 59 | 讨论新句："LGM ideal age increase of about 1000 years at 4000 m… our model somewhat overestimates the age increase" | 否 | 否 | 见 E |
| 60 | 讨论新段：年龄为下限（Millet2025），"Ideal age also responds to mixing"，MOC 更直接 | 否 | 否 | R3-7 正是问 mixing，回复没提 → 见 E |
| 61 | 讨论：加 de Lavergne 2014（开阔洋对流停止） | 否 | 否 | 小 |
| 62 | 讨论末句新增 "projections… will remain uncertain until the thermal and haline components… are evaluated separately" | 否 | 否 | 结论小补充，可不提 |
| 63 | 4.1 冰间湖分辨率、brine 按节点释放、无冰架腔、空白区 | 否 | 是 R2-minor:brine, R3-#81 | |
| 64 | 4.1 GLAC-1D 按组件、专用网格、盐度抬升、无融水、径流 0.085/0.07 Sv；加 buiron2011 | 否 | 是 R1-gen:BC(Q), R3-#81 | |
| 65 | 4.1 初始化细节；"same ensemble… Shi2025GRL"；漂移（SST ±0.03、MIS3 +0.06 K/century；AMOC PI +1.3 Sv/century）与 PMIP4 判据 | 否 | 是 R3-6 | 信写 "Kageyama et al., 2018"，稿 `\cite{Kageyama2017}`，请核年份 |
| 66 | 4.1 新段：前人代用资料评估综述（Shi2025GRL、Shi2022JCLI、Shi2023CP、Bartlein2011、TurneyJones2010），并说明没有一篇评估南大洋水文 | 否 | 否 | 见 F |
| 67 | 4.2 𝓑 单位 m² s⁻³ → kg m⁻² s⁻¹；公式淡水项 +βSF → −βSF；正 𝓑 = 增密（原为负 𝓑 = 增密） | 改 | 部分：R1-19 说加了 sign convention，没说约定改了 | 见 B |
| 68 | 4.2 thermal / sea-ice-driven density tendency 定义 | 否 | 是 R1-19 | |
| 69 | 4.3 bin 宽 0.2 → 0.1 kg m⁻³ | 改 | 部分（R3-1、R3-3 顺带） | 见 A |
| 70 | 4.3 新增形成率 M(ρ) = −∂F/∂ρ；原稿 "A positive F(ρ) indicates… accumulate" 改为 M(ρ) | 改（定义更正） | 否 | 见 J |
| 71 | 4.3 SIZ 及其补集的辅助计算、TEOS-10 换中性密度 | 否 | 是 R1-26, R3-2 | |
| 72 | 新小节 4.4 Overturning streamfunction（地理纬度 2° 分带、末 20 年、两个指数窗口） | 否 | 否 | 见 I |
| 73 | Table 1 边界条件 | 否 | 是 R1-gen:BC, R1-32 | |
| 74 | Data availability：原 Zenodo 链接 records/20280950 改为 "DOI to be inserted upon acceptance" 并列出存档内容 | 改 | 否 | 编辑事项，可不提 |
| 75 | Code availability：脚本公开于 GitHub | 改 | 否 | 可不提 |
| 76 | Acknowledgements：原基金文字删除，现为占位 "This research was supported by…" | 改 | — | 投稿前必须填 |
| 77 | 新 Fig. 8 积分域敏感性 | 否 | 是 R1-26, R3-L302-305 | |
| 78 | 示意图（现 Fig. 9）图注加 "σ₂ ≈ 36.5–36.8 (60–80 Sv)"、"σ₂ > 37.0"、"(48–58 Sv)" | 改 | 否 | 与正文 0.1 bin 数字 70–81、36.4–36.8、50–59 不符 |
| 79 | A2 MLD 色标 a–c 到 1000 m | 改 | 是 R3-L134 | |
| 80 | 新 A4（冰期 MLD 0–300 m） | 否 | 是 R1-10 | |
| 81 | A11 年均 WMT 改 0.1 bin（PI SO 年均峰 11.4 → 17.5 Sv，各格升 1.2–2.0 倍） | 改（数据） | 否 | 讨论 "comparable annual-mean transformation magnitudes to modern observations (Fig. A11)" 要按新值复核 |
| 82 | 附录图重排 A1–A13 | 改 | 是 R1-7 | |
| 83 | 新 A12（PI MLD 对 ORAS5） | 否 | 是 R1-gen:limitations, R3-1 | |
| 84 | 地图加经纬度标签、各列状态名 | 改 | 是 R1-gen:labels, R1-17, R3-#82 | |

计数（84 条）：信里已交代 47，部分交代 14（其中 #45、#47 只在引文里出现），没交代 22，不适用 1（#76）。

## 信里没交代的重要改动

- **A. WMT 密度 bin 0.2 → 0.1（#20、#22、#24、#53、#69、#81）**：建议在信开头的总述加一句，或挂在 R1-14 / R1-15 下。写明全部 WMT 改用 0.1 kg m⁻³ bin 重算，列出主要数字变化：冬季极大 60–80 → 70–81 Sv，冰期 "below 60" → 50–59 Sv，年均 A11 升 1.2–2.0 倍。同时改掉信里残留的旧数字：R1-27 的 "σ₂ = 37.0–37.5… 12–17%" 应为 37.1–37.6、17–19%；R3-L175 的 "σ₂ ≈ 36.5–36.8" 应为 36.4–36.8。
- **B. 表面密度倾向的单位、符号与数值（#25、#26、#67）**：挂 R1-19（定义）或 R1-17（Fig. 3 单位），一句即可。原稿单位印作 kg m⁻³ month⁻¹（Methods 为 m² s⁻³、负值为增密），现统一为表面密度通量 kg m⁻² month⁻¹、正值为增密；正文数字相应更正（20 → 13，10 → 5，0.02 → 6.5）。
- **C. SAM 合成阈值不统一（#30）**：A7/A8 用 ±1.0σ，A6/A9/A10 和原稿用 ±1.2σ。挂 R2-minor:fig5-6，说明为何不同；也可以统一阈值。
- **D. SAM-WMT 异常的数字与结论变了（#2、#33、#57）**：3–8 → 6–10 Sv，"间冰期最强、各态差别大" 改为 "各态相近、只是机制不同"。挂 R3-4，或 R1-gen:interp（已说 "moderated the SAM claims"），交代重算后的数字和结论变化。变化原因是 0.1 bin 还是合成重算，需作者确认。
- **E. 理想年龄的新限定（#42、#59、#60）**：年龄是下限（1000 年未平衡，Millet2025）；年龄也反映混合，MOC 是更直接的量；模式高估 LGM 年龄增幅。审稿人 3 第 7 条正问到 "ideal age is strongly affected by mixing"，回复却没提这三句。建议挂 R3-7，高估一句可同时挂 R3-L347。
- **F. Methods 新增前人代用资料评估综述与 "同一组实验"（#65 前半、#66）**：挂 R1-gen:model-eval 或 R3-3，一句。写明 Methods 已概述 Shi et al. 2022、2023、2025 的代用资料评估，并说明这些评估都不涉及南大洋水文。
- **G. Fig. 1、A1、A2 实为 6–9 月平均（#12）**：一般性一句（或挂 R1-11），说明原图注写 JJA，现已更正为 June–September。
- **H. 摘要的新结论句（#5）**：挂 R2-major1 / R3-2，说明摘要已写明 PI 转化的密度比观测轻。
- **I. Methods 4.4 MOC 算法（#72）**：R3-7 加半句指向 Methods 4.4。
- **J. 形成率 M(ρ) 的定义更正（#70）**：R1-19 加半句。

## 顺带发现（稿件内部，非信的问题，供作者定）

- 示意图 Fig. 9 图注的数字是旧的（#78）：60–80 Sv、36.5–36.8、48–58 Sv，正文是 70–81、36.4–36.8、50–59。
- 讨论里 "Moreover, our paleoclimate analysis suggests…" 一段在原稿是注释（#54）。修订稿取消了注释，与下一段 "These paleoclimate results carry direct implications…" 重复，疑为误删 %。
- 2.1 的两处指针（#14、#16）：密度异常写成 `\ref{sst}i-l` 和 `\ref{sst}j`，应为 Fig. A1a-d 和 A1b；冰期 MLD 处写成 `\ref{reso}`（A3 网格图）。
- 讨论 "comparable annual-mean… (Fig. A11)"（#81）要按 0.1 bin 的新年均值复核。
- "σ₂ = 36.75–37.00" 的分档请核（#45）。
- Kageyama 引文年份请核（#65）。
- 致谢和基金仍是占位（#76）。
