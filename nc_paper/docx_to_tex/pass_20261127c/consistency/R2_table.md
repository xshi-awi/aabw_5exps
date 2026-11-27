# 审稿人 2（信 477–561 行）与审稿人 4（885–893 行）：信的承诺对照修订稿

依据：response_new.tex、revised.tex/.aux、original.tex、nc-main_submitted.aux（原稿 Fig. 5 = sam_heat，Fig. 6 = sam_fwf）。
数字复算用的数据：项目根目录 `wmt_siz_pi.nc`（0.2 格，12 个气候月，信图 figR20 用的就是它）、`wmt_siz_pi_bin0p1.nc`、`wmt_siz100_pi_bin0p{1,2}.nc`（稿件 Fig. 8 现在用 0.1 格的 100 年逐月）。

| # | 信（审稿人、条目、承诺） | revised.tex 位置 | 状态 | 说明 |
|---|---|---|---|---|
| 1 | R2 大问题 1：「the original single sentence was inadequate」，暗示稿件已改 | 原稿讨论句「This discrepancy likely reflect model biases or regional differences in analysis domains.」已删。讨论里新增三段：「Two factors contribute, and they can be separated.」「The second factor is the density range…」「A residual discrepancy remains…」；新增 Fig. 8（`domain`）；方法 L334 加了海冰区的计算 | 一致 | 稿件确实改了，但这条回复一句都没指向稿件的改动（见需要作者定 1） |
| 2 | R2 大问题 1：信图 (a) 和正文数字「minimum of -9.2 Sv … peaks at +6.6 Sv」 | 稿件里没有这两个数。稿件 Fig. 8 (f) PI 画的是同一个量：海冰区年均 | 不一致 | 信里的数与 `wmt_siz_pi.nc` 相符（0.2 格：−9.17、+6.55）。稿件 11-27 起改用 0.1 格的 100 年逐月数据，Fig. 8f 的 PI 峰值是 13.0 Sv（σ2 36.85）。同一个量，信图约 6.6 Sv，稿件图约 13 Sv（见需要作者定 2） |
| 3 | R2 大问题 1：「freshwater term dominates … below γ = 27.1 and … above γ = 27.67, thermal signature confined to a band」 | 讨论：「only one class … remains thermally dominated, $\sigma_2 = 36.75$--$37.00$」，折合 γ ≈ 27.33–27.59；「transformation maximum sits near $\gamma_n \approx 27.3$」 | 不一致 | 两边口径不同：信按画出的曲线分（0.2 格复算得 27.10 和 27.674，舍入到 27.67/27.68 正好在边界上），稿件按单个格分。稿件这句在 0.1 格下可能已经过时（见需要作者定 3） |
| 4 | R2 大问题 1：Pellichero「5±5 Sv」，数据来自「Argo floats, ship-based profiles and instrumented seals」 | 讨论首段：「approximately 5$\pm$5 Sv」「based on Argo floats and instrumented seals」 | 一致 | 信多写了 ship-based profiles，与稿件不矛盾 |
| 5 | R2 大问题 1：「The comparable calculation shows that the driving component is mixed」 | 讨论：「whereas our PI integral is thermally dominated」；海冰区峰值处「thermal share of PI falls from 79% to 76%」 | 一致 ⚠ | 不算错。信说驱动是混合的，稿件强调峰值处仍有 76% 来自热，读者两边对照可能觉得口径不一（见需要作者定 4） |
| 6 | R2 大问题 1：ORAS5，「+5.0 Sv for us and +4.4 Sv for ORAS5」；「ours is stronger … winter mixed layers being deeper and more extensive」 | ORAS5 的 WMT 只在信里，稿件没有。稿件只拿 ORAS5 比混合层（Supp. Fig. A12）：「too deep in the interior of the Weddell gyre and too shallow along the Antarctic coast」 | 一致 | 数字与 0.2 格数据相符（4.99、4.38）。混合层那半句可以加「(Supplementary Fig. A12)」；稿件说的是 Weddell 内部偏深、沿岸偏浅，信里的「more extensive」只对开阔洋成立 |
| 7 | R2 大问题 2：Chen et al. 2025「now cited in both the Introduction and the Discussion」 | 引言 L78「A recent study using the same ocean model in a forced, ocean-only configuration … \cite{Chen2025}」；讨论 L257 | 一致 | 原稿 aux 里没有 Chen2025，确为新增 |
| 8 | R2 大问题 2：引文「The passage added to the Discussion reads」 | 讨论段「Our conclusions about the glacial state are consistent with recent work…」 | 一致 | 位置在讨论，原稿没有，确为新增。quote_check #24 的 MISSING 只是「Chen et al. (2025)」和 `Chen et al. \cite{}` 的写法差别 |
| 9 | R2 大问题 2：回复正文（两个态、纯海洋、大气给定；本文五个耦合态） | 同上那段 | 一致 | |
| 10 | R2 小问题 1：「expanded the Methods」：25 km、只表现大型冰间湖、热力学结冰和析盐、冰间湖尺度要几 km 网格、无冰架腔 | 方法 4.1 L277「At this resolution the larger coastal polynyas are represented as features…」「…thermodynamically at each surface node…」「…of the order of a few kilometres.」「The configuration does not include ice shelf cavities…」 | 一致 | 原稿 L256 只有「…25 km in polar regions…」，确为新增。信写的是「at each mesh」，与稿件措辞不同（见文字小修 a） |
| 11 | R2 小问题 1：「Supplementary Fig. A13 … 61--88%, most of it within 200 km of the coast」 | A13 = `formation_loc`；讨论「form mainly on the shelf and slope (61--88\%)」 | 一致 | 「within 200 km」正文没写，只能从 A13 图注 (e) 看出来 |
| 12 | R2 小问题 2：Lhardy 2021「added and discussed」，引文 | 讨论 L261「Many PMIP models underestimate glacial Southern Ocean sea ice…」 | 一致 | 原稿未引 Lhardy，确为新增；29.8、11.1、33±5、10±3 两边相同。quote_check #25 只是 ± 的写法差别 |
| 13 | R2 小问题 3：结果开头介绍五个实验和缩写，正文写全称，MH/LIG 只留在图和 Table 1 | 结果 2.1 首句「The figures label the five simulated climate states as PI (pre-industrial), MH (mid-Holocene), LIG…」；方法 L294 也有定义 | 一致 | MH、LIG 在正文里只出现在这两处定义和 Table 1 里，原稿结果开头没有这段 |
| 14 | R2 小问题 4：原稿 Figs 5、6 移入补充材料 | 原稿 5 = sam_heat、6 = sam_fwf，现为 A7、A8 | 一致 | |
| 15 | R2 小问题 4：「removed the repetition that Reviewer 1 flagged」 | 讨论 SAM 段：原稿连说两次「SAM … systematically enhances …」「composite analysis reveals that positive SAM phases systematically enhance」，现在合成一句，并补了 Ross、Weddell 的限定 | 一致 | |
| 16 | R2 小问题 4：「removed … the speculative passage about marine proxy reconstruction」 | 原稿「…implies potential for paleoclimate proxy development … marine-based SAM reconstructions…」已删，换成「Whether this coupling could be used for proxy reconstruction is unclear. … We do not expect it to be recoverable from marine sediments.」 | 一致 | 准确说是换成了一段告诫，见文字小修 b。2.4 节从约 700 词增到约 1230 词；信没有说缩短，符合 SAM 折中的既定写法 |
| 17 | R2 小问题 5：A7、A8 统一单位 kg m⁻² month⁻¹、共用一条色标、单位与 Figs. 3、4 相同 | A7/A8 图注「shares one colour scale with Fig. A8/A7」「Units: kg m$^{-2}$ month$^{-1}$」；Figs. 3、4 单位相同 | 一致 | 原稿单位是 W/m²，确为新改。信图注里的 ±1.0σ、15 W m⁻² 与 A7 图注一致；信和稿件的图文件同时出（11-27 11:16） |
| 18 | R2 小问题 5：引文「The corresponding statement in the Results reads」 | 结果 2.4 L183「Because Figs.~\ref{sam_heat} and \ref{sam_fwf} share one unit…」 | 一致 | 位置在结果，原稿没有，确为新增 |
| 19 | R2 小问题 5：回复正文（冰下淡水项大，开阔水热项大） | 同上 | 一致 | |
| 20 | R4：只致谢，没有承诺 | 无 | 一致 | 句子不通，见文字小修 e |

统计：一致 18（其中 #5 带 ⚠），不一致 2（#2、#3），缺失 0，原稿已有 0。

## 文字小修（都只改 response_new.tex）

a. 第 520 行：`computed thermodynamically at each mesh and enter the ocean as a salt flux,` → `computed thermodynamically at each surface node and enter the ocean as a salt flux at that node,`（与方法的措辞一致）
b. 第 540 行：`removed the repetition that Reviewer 1 flagged and the speculative passage about marine proxy reconstruction,` → `removed the repetition that Reviewer 1 flagged and replaced the speculative passage about marine proxy reconstruction with a short caveat,`（稿件里留了一段告诫，没有整段删掉）
c. 第 498 行（第 97 行给 R1、第 637 行给 R3 的同一句也一样）：`For annual mean over the sea ice zone` → `For the annual mean over the sea ice zone`；可选：`deeper and more extensive than in the reanalysis.` → `deeper and more extensive than in the reanalysis (Supplementary Fig. A12).`
d. 第 508 行：`Ours are fully coupled model, so` → `Ours are fully coupled, so`
e. 第 889 行：`We thank Reviewer 4 for the contribution to the review, and see corresponding responses.` → `We thank Reviewer 4 for contributing to the review. Our responses to the joint report are given above.`

## 需要作者定

1. **R2 大问题 1 的回复没提稿件改了什么。** 这是审稿人 2 的第一个大问题，稿件里原来那一句已经换成讨论里的三段，加了 Fig. 8 和方法里的海冰区计算，可回复只讲信里的新图。建议加一两句指向稿件，例如「In the manuscript, the single sentence is replaced by three paragraphs in the Discussion and by the new Fig. 8」，或者把给审稿人 1 的那段引文（信第 105 行一带）也放一份在这里。
2. **格宽口径不一致。** 信图 figR20（这里和 R3 那里的 figR20a/b 都是）和信里的 −9.2、+6.6、+5.0、+4.4、+62.9/+36.8 Sv，用的是 0.2 格、12 个气候月（`wmt_siz_pi.nc`、`wmt_oras5_*.nc`）。稿件 11-27 起改用 0.1 格、100 年逐月：Fig. 8f 的 PI 海冰区峰值是 13.0 Sv，0.1 格气候月是 11.9 Sv，信图是 6.6 Sv。审稿人把信图 (a) 和 Fig. 8f 放在一起看，会看到差了一倍。13 Sv 也已经超过 Pellichero 5±5 Sv 的上限。两条路：(i) 信图和 ORAS5 都按 0.1 格重算，ORAS5 也要换格，才能一对一比较，信里这些数随之改；(ii) 信图注写明「0.2 kg m⁻³ bins, as in the submitted version」，并说明为什么和 Fig. 8 不同。
3. **热主导带的说法与稿件对不上，稿件那句可能已经过时。** 信写 γ 27.1–27.67 一带，稿件讨论写「only one class … σ2 = 36.75–37.00」。我用 100 年逐月的海冰区 PI，按 |热|/(|热|+|冰|+|其他|) > 50% 且 |G| > 0.1 Sv 逐格判断：0.1 格是中心 36.75、36.85 两格（σ2 36.70–36.90）；0.2 格是 36.6–37.0 两格。两种都对不上「一个格 36.75–37.00」。请用原来的判据在 0.1 格下核一遍这句，再决定信里的带怎么写。
4. **信和稿件的侧重点不同。** 信说「the driving component is mixed」，稿件讨论说「our PI integral is thermally dominated」，海冰区峰值处热占比 76%。建议在信这句后面补上「at the transformation maximum the thermal share in the sea ice zone is still 76% (Fig. 8)」，让信和稿件讲的是同一件事。

附带一条，不在我的范围内：修订稿 L115（结果 2.1）「becomes confined to narrow coastal bands and polynyas (Fig. \ref{reso})」指向的是 A3（分辨率图）。原稿这里是 `\ref{mld}d,e`，看起来是指错了图。
