# 用户在 05_response_letter_suggestion.docx 中的 23 处黄色批注 — 处理清单

提取方式：解压 docx，扫 `w:highlight w:val="yellow"` 的 run，按段落还原上下文。

## A. 需要"引用修改后的正文原文（斜体）"的条目 — 这是最主要的一类

批注反复要求：**每一条回复都要把 revised version 里新加/改过的文字用斜体引出来**，
让审稿人只读回复信、不翻稿子也能明白改了什么。原话：
「给出abstract里我们修改成了什么。。。斜体字引用出来，以下每一条都要这样」

涉及段落：50(abstract/SAM)、68(Fig1 i-l 密度)、70(LIG/LGM 异常)、75(thermal 定量)、
39(SAM 措辞收敛)、138、145、151(Lhardy)、以及「以下每一条都是，我不再重复」。

做法：对每条已做实质修改的回复，在末尾加
`According to the comment, we have added the following to the revised manuscript: "..."`
并用蓝色粗斜体引出 revised.tex 里的真实句子（必须与稿子逐字一致，不能另写一版）。

## B. 需要新图或挪图

- **para 65 / R1#10**：把 LGM/MIS3 的 MLD colorbar 收紧，让沿岸冰间湖的深对流显出来，
  作为 **R1.4** 放在这条回复里。→ 已做 `figures/figR7_mld_glacial_zoom.pdf`
  （0–300 m，p99=280/296 m，橙线 200 m；上排对照原 0–400 m 标度）
- **para 22 / R1 G1**：把现有 MLD 图从 G1 挪走，换成上面那张收紧版，移到 para 42（model limitation）
- **para 42**：G1 与 model-limitation 两条重复，图要分散；这条强调 bias 而非 validation
- **para 82 / R1#17**：加了 climate-state 列标签的新图也要放出来给审稿人看
- **para 123 / R2 major1**：先给一张**区域图** —— 我们 <60°S 的范围 vs Pellichero 的海冰区，
  说明看起来像但其实差很多。→ 已做 `figures/figR8_domain_map.pdf`
  （(a) 2.07e13 m²，(b) 1.22e13 m²，(c) 橙色 42.7% 是我们有他没有的常年开阔水）
- **para 157 / R2 units**：把新图给出来
- **para 71 / R1#13**：Ross 海问题「这里有图支撑吗？有的话放这里」→ 需判断

## C. 需要引用你自己已发表文章的 model validation（**待核实，已派 agent**）

- para 22：「把之前发表文章里的 model-data comparison 总结一下」
  - 现代场 → Sidorenko 那篇标题含 ECHAM/FESOM2 的
  - LGM/MIS3 → 你的非洲季风 GRL，supplementary 里有 validate
  - MH/LIG → 你的 JC 文章，谈第二代 AWIESM
- para 35：实验设置要更详细，「去我的 GRL 那篇看边界条件、初始场、转了多少年、取最后 100 年」

**状态：已核实完毕（papercheck agent + 我自己查 Crossref）。四篇都找到了：**
- Sidorenko et al. 2019 JAMES 11, 3794-3815 — ref.bib 里本来就有 `sidorenko2019evaluation`，沿用旧 key
- Shi et al. 2022 J. Climate 35(23), 7811-7831 → 新 key `Shi2022JCLI`（MH/LIG，两代 AWIESM）
- Shi et al. 2025 GRL 52(9), e2024GL112717 → 新 key `Shi2025GRL`（非洲季风，同样五个实验 PI/MH/LIG/LGM/MIS3）
- Shi et al. 2023 Clim. Past 19, 2157-2175 → 新 key `Shi2023CP`（LGM 设置，开放获取，spin-up 措辞可逐字引用）

**Sidorenko 2019 的重要限制（必须遵守）**：全文没有 polynya / convection / deep water formation
字样，没有南大洋 MLD 评估，没有跟任何卫星海冰产品比较。所以只能引用它确实写了的：
AABW cell ~10 Sv、南大洋暖偏差 SST RMSE 1.43 K（vs PHC）、9 月威德尔海冰厚度不到 0.25 m、
以及作者自己那句 "biases ... are present, they still result in a reasonable density distribution"。
开阔洋对流这个判断**仍然只能作为我们自己的诊断**。

**顺带查出一个真错误**：ref.bib 里的 `Lhardy2022` 条目 DOI/卷/页都对，但**作者写错了** ——
10.5194/cp-18-845-2022 实际是 **Green et al.**（你是共同作者），不是 Lhardy。已经 Crossref 核实并改正，
key 也改成 `Green2022`。该条目此前未被引用，所以稿子里没有出错，但这是个定时炸弹。

## D. 需要注意一致性/口径的条目

- **para 102 / R1#26**：「这个好像就是第一个 comment，咱们的回答前后要一致，别出现矛盾」
  → R1 G1、R1#26、R2 major1、R3 都在问同一件事，四处口径必须统一
- **para 162 / R3**：「又是跟前面 2 个审稿人相同的 concern，请小心应对」
- **para 21 / R1 G1**：澄清 Fig A4「跟观测一致」的依据其实只是 Pellichero 那个 5±5 Sv 的数字，
  文章里在那句话后面补个引用即可
- **para 124 / R2**：承认 MLD/开阔洋对流的 bias，但强调**我们讨论的是相对变化**，
  这对结论有利 —— 既认 bias 又保住研究价值。要展开得让审稿人信服
- **para 45 / R1 G5**：加个具体例子「for example, we added the labels in fig. XXX (also
  referred to as Fig. R1.3), we treated other figs likewise」
- **para 48 / R1#1**：「thank you, and now we have ... in the revised manuscript」

## E. 可选的额外分析

- **para 168 / R3**：GLORYS 再分析「这个数据可以下载试试，做一些 wmt 分析，说不定有惊喜」
  → 之前在回复信里婉拒了。若要做，需下载 Copernicus 数据，是独立的一大块工作，
    且表面通量与再分析海洋态不自洽的问题依然存在。**需要你拍板。**

---

## ⚠️ 未决问题：spin-up 年数在已发表文献之间互相矛盾（2026-09-21 查出）

三篇文章对"同一个" PI 给了三个不同的积分长度：
- Shi et al. 2022 JCLI：**1000 年**（PI/MH/LIG），判据 ±0.05 K/century
- Shi et al. 2023 CP：**1300 年以上**（只有 PI 和 LGM；该文没有 MH/LIG/MIS3）
- Shi et al. 2023 GMD（据 KB awiesm2.md）：PI **1500 年**，MH 再续 1500 —— 但那是 wiso 分支，不同血统
- 本文投稿版写的：PI 1500 年，古气候实验各 1000 年

这四个数没有一个能互相对上。其中有些可能确实是不同的模拟（CP 那篇的 LGM 是从
ECHAM5-MPIOM 初始化的，不是从 PI 分叉；GMD 是同位素分支），但**读者不会知道，除非我们写明**。

**我尝试从 run config 核实，失败了**：`production/{exp}_age/` 只是带 ideal age 的续算段
（pi/mh/lig 从模式年 2000 起算，lgm 到 2499，mis 到 2616），parent 实验
（pi_beta / mh_beta / lig_beta / glac1d_final / mis）的 outdata 已经是后处理产物，
原始年份追不回来。pi_beta 的 restart 停在模式年 2899，说明血统很长但跨了重编号的段落。
Shi 2025 GRL 的 Supporting Information Text S1 本来能一锤定音，但 Wiley 付费墙打不开。

**已做的处理**：把我先前加的 "following the protocol used for the same model in earlier work
\cite{Shi2022JCLI,Shi2023CP,Shi2025GRL}" 这句**删掉了** —— 那等于宣称我们的 1500/1000 跟那三篇
一致，而实际并不一致。现在 Methods 只保留投稿版原有的 1500/1000（那是用户自己写的数），
引用只挂在它们确实支持的论断上（proxy validation）。

**需要用户确认**：这五个实验真实的 spin-up 长度是多少？如果 1500/1000 是对的就保持现状；
如果实际是别的数，现在改还来得及。审稿人 3 已经在问平衡态问题，这个数被查出来对不上会很被动。

### 文献核实的技术记录（备查）

**PDF 获取路径**（本轮实测，其他站点都被挡）：
- AWI EPIC：需要带浏览器 user-agent 和 cookie jar
- Copernicus（Clim. Past / GMD）：直连 PDF 可下
- **Wiley、AMS、GEOMAR 一律封锁自动访问** —— GRL 和 J. Climate 的正文只能靠摘要 + 索引文本

**Sidorenko 2019 的三条 "NOT DISCUSSED" 是全文 grep 出来的**，不是从摘要推断：
没有 mixed layer depth 评估、没有 open-ocean convection / polynya 字样、
没有跟任何卫星海冰产品比较。引用时不得越过这三条界线。

**仍未拿到**：Shi et al. 2025 GRL 的 Supporting Information Text S1（付费墙），
那是五个实验权威实验设计的所在，也是能一锤定音解决 spin-up 年数问题的文件。
若用户记不清年数，可以从这里入手（作者本人有权限）。

**已清理**：`awi_validation_refs.bib` 暂存文件已删除，6 个条目中 5 个已并入
`build/ref.bib` 并被引用（`Shi2023GMD` 是同位素那篇，并入但未引用，留着备用），
`Sidorenko2019` 是与 `sidorenko2019evaluation` 的重复，已弃用。
