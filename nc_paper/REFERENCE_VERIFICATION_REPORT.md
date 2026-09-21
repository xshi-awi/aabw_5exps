# 参考文献核查报告（含原文原句）

**文件**：`nc-main.tex` 引用的全部文献，对照 `Ref.bib`
**核查日期**：2026-05-17
**方法**：逐篇上网检索原文，核对元数据；并对**内容引用正确**的文献，回到原文（开放获取抓正文，付费墙抓摘要）摘录与你引用位置直接相关的**英文原句**。

---

## 目录

- [第一部分：元数据/内容存在问题的文献（需修改）](#第一部分需修改)
- [第二部分：内容引用正确的文献——主要内容 + 原文原句](#第二部分原文原句)
  - [A. WMT 框架与方法论](#a-wmt-框架)
  - [B. 南大洋环流与碳/热](#b-南大洋环流)
  - [C. 海冰与水团变换（观测/理论）](#c-海冰与水团变换)
  - [D. 古气候与放射性碳代用](#d-古气候与代用)
  - [E. 模式与方法（AWI-ESM 组件）](#e-模式与方法)
  - [F. 温室气体冰芯重建](#f-温室气体冰芯)
  - [G. 软件工具](#g-软件工具)

---

<a name="第一部分需修改"></a>
## 第一部分：需修改的文献

### 🔴 元数据严重错误（作者列表错误）

| 文献键 | bib 现状 | 正确作者 |
|---|---|---|
| **Iudicone2008b** | Iudicone, Rodgers, Stendardo, Aumont, Madec, Bopp, Mangoni, Ribera d'Alcalà | **D. Iudicone, G. Madec, B. Blanke, S. Speich** |
| **Pellichero2018** | Pellichero, Sallée, Schmidtko, Roquet, Charrassin | **Pellichero, V.; Sallée, J.-B.; Chapman, C. C.; Downes, S. M.** |
| **Bailey2023** | Bailey, Jones D.C., Abernathey, Josey, Sallée | **Bailey, S. T.; Jones, C. S.; Abernathey, R. P.; Gordon, A. L.; Yuan, X.** |
| **Koeve2015** | Koeve, Gutknecht, Oschlies, Wagner W. | **Koeve, W.; Wagner, H.; Kähler, P.; Oschlies, A.** |

### 🟡 元数据次要问题

| 文献键 | 问题 | 建议 |
|---|---|---|
| otto2017pmip4 | 缺期号 | 补 `number={11}` |
| Zhou2023 | 末位作者拼写 | `Osterhus` → `Østerhus` |
| fluckiger2002high | 页码 `10--1` 错误 | 改 `pages={1010}` 或补 `doi={10.1029/2001GB001417}` |
| drake2025water | 缺 DOI | 补 `doi={10.1029/2024MS004383}` |
| rafter2022global | 缺 DOI | 补 `doi={10.1126/sciadv.abq5434}` |
| shi2020early | 缺 DOI | 补 `doi={10.1177/0959683620908634}` |

### 🔴 内容错引/夸大（需改写正文）

| 引用位置 | 问题 | 建议 |
|---|---|---|
| **第80行**："potentially explaining 50-80% of the glacial CO2 drawdown \cite{Ferrari2014,Skinner2017}" | **Ferrari2014 实际只给"10–20 ppm（约11–22%）"**，并回避单一机制主导归因；Skinner2017 说"超过一半（约50–70%）"，**不支持上限80%** | 改为 "explaining more than half of the ~90 ppm glacial–interglacial amplitude \cite{Skinner2017}"，Ferrari2014 改作机制关联引用 |
| **第231、250行**：太平洋年龄~1500年 / >750年 "consistent with radiocarbon proxy reconstructions \cite{Skinner2017}" | Skinner2017 标志数值是**全球平均 ~689±53 ¹⁴C-yr**（中深层鼓包），并非太平洋1500年具体值 | 软化为 "of the same order as the ~690 ¹⁴C-yr global-mean glacial increase reported by \cite{Skinner2017}"，或为1500年另引太平洋专门文献 |
| **第82行**："SAM is known to modulate present-day Southern Ocean surface buoyancy fluxes \cite{Sallee2010}" | Sallée2010 核心结果是 SAM 对**混合层深度**的影响，摘要用词为 "heat flux" 而非 "buoyancy flux" | 改为 "modulate present-day Southern Ocean mixed-layer depth and air-sea heat exchange \cite{Sallee2010}" 或补一篇直接关于 SAM–浮力通量的文献 |

### 🟡 措辞偏强（可接受，建议注明为本文解读）

- **第80、82行**："glacial sea ice expansion shifts the dominant surface buoyancy forcing from thermal to haline \cite{Ferrari2014,Marzocchi2017}" —— 两文支持"海冰增强卤水/盐度强迫"，但均未明确表述为"主导项从热到盐的转换"，该框架是作者综合。

### ⚪ 正文与 bib 不一致

- **第332行**代码可用性写 `ocean-eddy-cpt/xwmt`，规范仓库应为 **`NOAA-GFDL/xwmt`**（bib 正确，改正文）。
- **buiron2011taldice**（第273行 GHG 引用群）实为**冰芯定年/年代标尺**论文，非 GHG 浓度重建。建议单独作为年代标尺引用或加说明。

---

<a name="第二部分原文原句"></a>
## 第二部分：内容引用正确的文献——主要内容 + 原文原句

> 说明：以下每篇给出（1）主要内容中文概述；（2）你文章中的引用位置与论点；（3）支持该论点的**英文原文原句**（带来源）。开放获取文献尽量取正文原句，付费墙文献取已核实的摘要原句。

---

<a name="a-wmt-框架"></a>
### A. WMT 框架与方法论

#### Walin1982 — Walin (1982), *Tellus* 34, 187–195

**主要内容**：水团变换（WMT）理论的奠基之作。推导了微分加热、扩散热通量与表层漂流的关系，证明在中低温度区间向极的表层体积输运可直接由海表热通量确定。

**引用位置**：第65、73、244、293行 —— "Surface buoyancy fluxes fundamentally control the transformation of water masses"；"This method relates surface heat fluxes to cross-isothermal mass transport [Walin1982]"。

**原文原句**：
- > "A theoretical framework for the description of the thermal state and circulation in the ocean is presented."
- > "We find that for low and medium temperatures, the poleward surface drift can be determined directly from a knowledge of the heat flux through the sea surface."
- 旁证（Speer & Tziperman 1992）：> "The transformation F(p) is the analogue of the derivative with respect to temperature of the thermal forcing function Q(T) discussed by Walin (1982)."

来源：https://tellusjournal.org/articles/10.3402/tellusa.v34i2.10801 （正文付费墙；摘要经 Tellus/ADS/Wiley 三处一致核实）

---

#### Speer1992 — Speer & Tziperman (1992), *JPO* 22, 93–104

**主要内容**：将 Walin 框架扩展为同时包含热通量与淡水通量（E−P）的表层密度通量，计算北大西洋密度坐标下的水团生成率。

**引用位置**：第65、73、244、293行 —— WMT 框架 "extended to incorporate both thermal and freshwater contributions to surface density flux [Speer1992]"。

**原文原句**：
- > "North Atlantic air-sea heat and freshwater flux data from several sources are used to estimate the conversion rate of water from one density to another throughout the range of sea surface density."
- > "Air-sea fluxes of heat and freshwater at the ocean's surface change temperature and salinity characteristics and convert water from one density to another."

来源：https://oceanrep.geomar.de/42207/1/1520-0485(1992)022_0093_rowmfi_2.0.co;2.pdf （开放仓库全文）

---

#### Groeskamp2019 — Groeskamp et al. (2019), *Annual Review of Marine Science* 11, 271–305

**主要内容**：系统综述 WMT 框架，将环流、热力学与生物地球化学统一起来，作为欧拉/拉格朗日方法的补充。

**引用位置**：第73、244行 —— "The water mass transformation (WMT) framework provides a thermodynamic approach to quantifying how air-sea fluxes drive changes in water mass properties"。

**原文原句**：
- > "The water mass transformation (WMT) framework weaves together circulation, thermodynamics, and biogeochemistry into a description of the ocean that complements traditional Eulerian and Lagrangian methods."
- > "We show how it provides a robust methodology to characterize and quantify the impact of physical processes on buoyancy and other thermodynamic fields."

来源：https://pubmed.ncbi.nlm.nih.gov/30230995/ （摘要，经 PubMed/NORA/Soton 三处一致核实）

---

#### drake2025water — Drake et al. (2025), *JAMES* 17, e2024MS004383

**主要内容**：给出有限体积广义垂直坐标海洋模式中诊断闭合 WMT 收支的理论与开源 Python 软件栈（xbudget + xwmt）。

**引用位置**：第65、73、244、280、293行 —— WMT 框架；"we employ the xbudget diagnostic tool"。

**原文原句**：
- > "Water Mass Transformation (WMT) theory provides conceptual tools that in principle enable innovative analyses of numerical ocean models; in practice, however, these methods can be challenging to implement and interpret, and therefore remain under-utilized."
- > "The lightweight xbudget package (https://github.com/hdrake/xbudget) provides helper functions for wrangling complicated multi-level tracer budget diagnostics and is used to verify budget closure and decompose high-level terms into constituent processes."

来源：https://opensky.ucar.edu/system/files/2025-04/...drake...pdf （开放全文）；DOI 10.1029/2024MS004383

---

#### Iudicone2008a — Iudicone, Madec, McDougall (2008), *JPO* 38, 1357–1376

**主要内容**：提出中性密度框架下评估跨密度输运的新方法，将穿透性短波辐射等内部源纳入海表浮力通量估算（即对表层通量做分量分解）。

**引用位置**：第75行 —— "WMT implementations for the modern climate decompose surface fluxes into multiple components to isolate specific physical processes [Iudicone2008a,Iudicone2008b]"。

**原文原句**：
- > "A new formulation is proposed for the evaluation of the dianeutral transport in the ocean. The method represents an extension of the classical diagnostic approach for estimating the water-mass formation from the buoyancy balance."
- > "The inclusion of internal sources such as the penetrative solar shortwave radiation (i.e., depth-dependent heat transfer) in the estimate of surface buoyancy fluxes has a significant impact in several oceanic regions, and the former simplified formulation can lead to a 100% error in the estimate of water-mass formation due to surface buoyancy fluxes."

来源：https://eprints.soton.ac.uk/59050/ （摘要，与 AMS 10.1175/2007JPO3464.1 一致）

---

#### Iudicone2008b — Iudicone, Madec, Blanke, Speich (2008), *JPO* 38, 1377–1400

> ⚠️ **bib 作者列表错误**（见第一部分）。内容引用本身正确。

**主要内容**：基于冰-海耦合模式，将海表密度通量分解为热驱动、淡水驱动、海表盐度恢复等独立分量，量化各分量对南大洋水团变换的贡献。

**引用位置**：第75行 —— 同上（多分量分解隔离物理过程）。

**原文原句**：
- > "A quantitative dynamical analysis of the water-mass transformation has been performed using a new method."
- 图3标题：> "Surface density fluxes (10⁻⁶ kg m⁻² s⁻¹): total density flux (thick solid line); heat-driven density flux (dashed line); freshwater-driven density flux (dotted–dashed line); and density flux due to the surface salinity restoring (thin solid line)."

来源：https://horizon.documentation.ird.fr/exl-doc/pleins_textes/2023-06/010083533.pdf （开放全文）

---

<a name="b-南大洋环流"></a>
### B. 南大洋环流与碳/热

#### Marshall2012 — Marshall & Speer (2012), *Nature Geoscience* 5, 171–180

**主要内容**：综述论证南大洋西风驱动上升流是全球 MOC 的关键返回路径。

**引用位置**：第60行 —— "the primary region where deep waters return to the surface through wind-driven upwelling"。

**原文原句**：
- > "A key part of the overturning puzzle, however, is the return path from the interior ocean to the surface through upwelling in the Southern Ocean. This return path is largely driven by winds."
- > "It has become clear over the past few years that the importance of Southern Ocean upwelling for our understanding of climate rivals that of North Atlantic downwelling, because it controls the rate at which ocean reservoirs of heat and carbon communicate with the surface."

来源：https://www.whoi.edu/cms/files/marshall_speer_natgeo2012_270284.pdf （开放全文）

---

#### Talley2013 — Talley (2013), *Oceanography* 26(1), 80–97

**主要内容**：给出全球翻转环流示意与输运量；深层水向南上升于南大洋，AABW 在南大洋形成并向北输入深渊。

**引用位置**：第60行 —— 与 Marshall2012 并列（上升流 + 致密水通风深渊）。

**原文原句**：
- > "The global overturning circulation (GOC) includes both large wind-driven upwelling in the Southern Ocean and important internal diapycnal transformation in the deep Indian and Pacific Oceans."
- > "All three northern-source Deep Waters (NADW, IDW, PDW) move southward and upwell in the Southern Ocean. AABW is produced from the denser, salty NADW and a portion of the lighter, low oxygen IDW/PDW..."

来源：https://tos.org/oceanography/assets/docs/26-1_talley.pdf （开放全文）

---

#### Speer2000 — Speer, Rintoul, Sloyan (2000), *JPO* 30, 3212–3222

> ⚠️ **引用偏弱**：本文聚焦"上层"diabatic Deacon cell，摘要用"potential vorticity gradients"而非"density gradients"，未涉及下层环。"both upper and lower cells" 主要由 Lumpkin2007 支持。

**主要内容**：分析南大洋海气浮力通量与北向 Ekman 输运的相容性，UCDW 上升后向北流动获浮力。

**引用位置**：第60行 —— "strong meridional density gradients that drive both the upper and lower cells"。

**原文原句**：
- > "An analysis in density classes points to an upwelling of Upper Circumpolar Deep Water and subsequent buoyancy gain from air–sea exchange as water flows northward in the Ekman layer."
- > "...an eddy mass flux mechanism for southward transport in this layer to replenish the upwelling is advanced, based on observed large-scale potential vorticity gradients."

来源：https://journals.ametsoc.org/view/journals/phoc/30/12/1520-0485_2000_030_3212_tddc_2.0.co_2.xml （摘要）

---

#### Lumpkin2007 — Lumpkin & Speer (2007), *JPO* 37, 2550–2562

**主要内容**：逆方法估算十年平均全球环流，明确给出上层环（北极下沉、南大洋上升）与下层环（南极下沉、深渊上升）两个全球环。

**引用位置**：第60行 —— 与 Speer2000 并列（上、下两个环）。**精确支持**。

**原文原句**：
- > "The model obtains a global overturning circulation consistent with the various observations, revealing two global-scale meridional circulation cells: an upper cell, with sinking in the Arctic and subarctic regions and upwelling in the Southern Ocean, and a lower cell, with sinking around the Antarctic continent and abyssal upwelling mainly below the crests of the major bathymetric ridges."
- > "The magnitude of the two principal cells is about the same, 17.2 ± 3.3 Sv (at 48°N) for the upper cell and 20.9 ± 6.7 Sv (at 32°S) for the lower cell."

来源：https://www.aoml.noaa.gov/phod/docs/LumpkinSpeer07.pdf （开放全文）

---

#### Marinov2006 — Marinov, Gnanadesikan, Toggweiler, Sarmiento (2006), *Nature* 441, 964–967

> ⚠️ **部分支持**：支持"碳"调控，但全文为生物地球化学，**不涉及"热量储存"**。热量部分应靠 Gray2024/Marshall2012。

**主要内容**：南大洋"生物地球化学分界"——分界以南控制大气 CO₂ 交换，以北控制全球海洋生产力。

**引用位置**：第60、62、67行 —— "the Southern Ocean regulates the global heat and carbon storage"。

**原文原句**：
- > "The Southern Ocean has central roles in carbon dioxide exchange between the oceans and the atmosphere, and in nutrient supply to the rest of the world's oceans — but these are physically separated due to the nature of ocean circulation, creating a biogeochemical divide."
- > "The Southern Ocean is the most important high latitude region in controlling pre-industrial atmospheric CO2."

来源：https://pubmed.ncbi.nlm.nih.gov/16791191/

---

#### Gray2024 — Gray, A.R. (2024), *Annual Review of Marine Science* 16, 163–190

> ⚠️ **注意**：摘要明确支持"主导海洋热与碳吸收"，但**摘要中无"~40%"定量数字**（你正文已删该句，仅注释中保留——无影响）。

**主要内容**：综述南大洋在全球碳循环中的根本作用，主导人为热与碳的海洋吸收。

**引用位置**：第60、62、67行 —— SO 调控全球热与碳储存。

**原文原句**：
- > "The Southern Ocean plays a fundamental role in the global carbon cycle, dominating the oceanic uptake of heat and carbon added by anthropogenic activities and modulating atmospheric carbon concentrations in past, present, and future climates."

来源：https://pubmed.ncbi.nlm.nih.gov/37738480/

---

#### Sallee2010 — Sallée, Speer, Rintoul (2010), *Nature Geoscience* 3, 273–279

> ⚠️ **措辞问题**（见第一部分）：摘要用 "heat flux"，未用 "buoyancy flux"；核心结果是 MLD 响应。

**主要内容**：用 Argo 数据证明 SAM 导致南大洋混合层深度纬向不对称异常，机制为经向风引起的热通量异常。

**引用位置**：第82行 —— "SAM is known to modulate present-day Southern Ocean surface buoyancy fluxes [Sallee2010]"。

**原文原句**：
- > "The Southern Annular Mode (SAM), the dominant mode of atmospheric variability in the Southern Hemisphere, leads to large-scale anomalies in mixed-layer depth that are zonally asymmetric."
- > "From a simple heat budget of the mixed layer, meridional winds associated with departures of the SAM from zonal symmetry cause anomalies in heat flux that can explain the observations."

来源：https://www.nature.com/articles/ngeo812 （摘要）

---

<a name="c-海冰与水团变换"></a>
### C. 海冰与水团变换（观测/理论）

#### Abernathey2016 — Abernathey et al. (2016), *Nature Geoscience* 9, 596–601

**主要内容**：用南大洋状态估计（SOSE）量化海冰淡水通量；差异性卤水排泄与冰融化以约 22 Sv 转换上涌 CDW；海冰是上层翻转支主导项，与风驱海冰输运紧密耦合（即"海冰淡水泵"）。

**引用位置**：第65、75、246行 —— 海冰卤水排泄强迫；"acts as a freshwater pump ... concentrating salt in formation regions"；Discussion "aligns with the sea ice pump hypothesis"。

**原文原句**：
- > "We find that sea ice is a dominant term, with differential brine rejection and ice melt transforming upwelled Circumpolar Deep Water at a rate of ~22 × 10⁶ m³ s⁻¹."
- > "These results imply a prominent role for Antarctic sea ice in the upper branch and suggest that residual overturning and wind-driven sea-ice transport are tightly coupled."
- > "While brine rejection from sea ice is thought to contribute to the lower branch, the role of sea ice in the upper branch is less well understood..."

来源：https://www.nature.com/articles/ngeo2749 （摘要，经 BAS 出版记录核实；正文付费墙）
*注："concentrating salt in formation regions" 是对 "differential brine rejection" 的转述，非逐字原句。*

---

#### Pellichero2018 — Pellichero, Sallée, Chapman, Downes (2018), *Nature Communications* 9, 1789

> ⚠️ **bib 作者列表错误**（见第一部分）。内容引用正确。

**主要内容**：基于船测、Argo 浮标、标记海洋哺乳动物的冰下观测库，给出南大洋海冰区表面浮力通量驱动翻转的观测估计；海冰季节性生长/融化主导水团变换，如"泵"般驱动上涌 27±7 Sv，其中 22±4 Sv 转为更轻、5±5 Sv 转为更密的水。

**引用位置**：第65、77、244行 —— 淡水通量（尤其海冰热力学）在冰区主导；Argo+船+海洋哺乳动物；约 5±5 Sv 转为更密底层水。

**原文原句**：
- > "In this region, the seasonal growth and melt of sea-ice dominate water-mass transformations."
- > "Both sea-ice freezing and melting act as a pump, removing freshwater from high latitudes and transporting it to lower latitudes, driving a large-scale circulation that upwells 27 ± 7 Sv of deep water to the surface. The upwelled water is then transformed into 22 ± 4 Sv of lighter water and 5 ± 5 Sv into denser layers that feed an upper and lower overturning cell, respectively."
- > "...combining observations from ships, autonomous floats, and animal-born sensors."
- > "...it is the freshwater contribution that dominates the seasonal variation of the net buoyancy flux. The heat flux contributes only marginally."

来源：https://pmc.ncbi.nlm.nih.gov/articles/PMC5934442/ （开放全文）

---

#### Bailey2023 — Bailey, Jones, Abernathey, Gordon, Yuan (2023), *Ocean Science* 19, 381–402

> ⚠️ **bib 作者列表错误**（见第一部分）。内容引用正确（"haline forcing" 为忠实转述）。

**主要内容**：用 3 套海洋再分析（ECCOv4/SOSE/SODA）首次诊断威德尔海 AABW 的闭合水团收支；表面盐通量主要由卤水排泄驱动，E−P−R 贡献极小，盐析在大部分季节主导。

**引用位置**：第77、244行 —— "the closed-budget WMT analysis in the Weddell Sea reveals that sea ice brine rejection dominates the surface salt flux for most seasons"；"dominant role of haline forcing in the Weddell Sea"。

**原文原句**：
- > "From the model outputs, we diagnose a closed form of the water mass budget for AABW that explicitly accounts for transport across the WG boundary, surface forcing, interior mixing and numerical mixing."
- > "We found a minimal contribution of E–P–R to surface salt fluxes and a dominant role from sea ice activity. Specifically, we saw brine rejection activity throughout most of the season until summer when there were spikes in ice melt..."
- > "...this indicates that brine rejection is the dominant process behind AABW formation, as was seen in the annual mean budget..."

来源：https://os.copernicus.org/articles/19/381/2023/os-19-381-2023.pdf （开放全文）
*注：本文亦强调粗分辨率再分析"did not realistically capture AABW formation"，盐度主导为稳健定性结论。*

---

#### Ferrari2014 — Ferrari et al. (2014), *PNAS* 111(24), 8753–8758

> ⚠️ **CO₂ 数值错引**（见第一部分）：本文给南极源水扩张贡献"10–20 ppm"，冰期总差"80–90 ppm"，**未支持"50–80%"**。

**主要内容**：LGM 夏季海冰线北移≥500 km，与现代"8字形"环流分裂为上下两个独立环相关；南极沿岸浮力通量为负（冷却+卤水排泄变咸）。

**引用位置**：第80、82行 —— 海冰扩张使浮力强迫由热转盐；50–80% CO₂ 下降。

**原文原句**：
- > "The buoyancy flux is negative around coastal Antarctica where the relatively warm subsurface waters that upwell in the Southern Ocean are cooled to the freezing point and become saltier through brine rejection as new ice is formed."
- > "the 5° latitude expansion of summer sea ice at the LGM was accompanied by the splitting of the modern figure eight overturning circulation in two separate overturning cells"
- > "The expansion of Antarctic-origin abyssal waters, richer in nutrients and metabolic carbon than the deep Arctic-origin waters it replaced, is believed to have reduced atmospheric CO2 by 10–20 ppm"

来源：https://pmc.ncbi.nlm.nih.gov/articles/PMC4066517/ （开放全文）

---

#### Marzocchi2017 — Marzocchi & Jansen (2017), *GRL* 44(12), 6286–6295

**主要内容**：用 PMIP LGM/PI 模拟研究南极海冰通过卤水排泄驱动表面浮力损失，塑造现代与冰期深海环流和层结；强 LGM 海冰模式同时表现增强层结与更浅 AMOC。

**引用位置**：第80、82行 —— 与 Ferrari2014 并列（海冰扩张改变浮力强迫与深水形成）。

**原文原句**（出版商摘要/US CLIVAR 高亮；Wiley 正文付费墙）：
- > "the role of Antarctic sea ice in shaping deep ocean circulation and stratification, by driving surface buoyancy loss associated with brine rejection"
- > "models simulating strong LGM sea ice formation also exhibit enhanced stratification and a shallower Atlantic Meridional Overturning Circulation (AMOC)"

来源：https://usclivar.org/research-highlights/role-antarctic-sea-ice-shaping-modern-and-glacial-deep-ocean-circulation ；DOI 10.1002/2017GL073936

---

#### Silvano2018 — Silvano et al. (2018), *Science Advances* 4, eaap9467

**主要内容**：冰架基底融水部分抵消冰间湖海冰盐通量，阻止全水柱对流与致密陆架水形成；变暖下融水增加将减少 AABW 形成。

**引用位置**：第246行 —— "concurrent Antarctic ice sheet mass loss could substantially modify this response through enhanced freshwater input [Li2023Nature,Silvano2018]"。

**原文原句**：
- > "freshwater input from basal melt of ice shelves partially offsets the salt flux by sea ice formation in polynyas found in both regions, preventing full-depth convection and formation of DSW."
- > "increased glacial meltwater input in a warming climate will both reduce Antarctic Bottom Water formation and trigger increased mass loss from the Antarctic Ice Sheet."

来源：https://pmc.ncbi.nlm.nih.gov/articles/PMC5906079/ （开放全文）

---

#### Silvano2020 — Silvano et al. (2020), *Nature Geoscience* 13, 780–786

**主要内容**：观测到罗斯海 AABW 形成/盐度近期恢复，由 2015–2018 正 SAM 与极端厄尔尼诺异常组合引发的风致海冰生成增强驱动。

**引用位置**：第248行 —— "the recovery of Ross Sea convection during 2015--2019 from positive SAM combined with El Niño forcing [Silvano2020]"。
*注：原文驱动窗口为 2015–2018，恢复属性观测至 2018–2019；"2015–2019" 为合理取整。*

**原文原句**（摘要）：
- > "Recent recovery of Antarctic Bottom Water formation in the Ross Sea driven by climate anomalies"（标题）；正文摘要指出 dense shelf water 形成恢复源于 2015–2018 正 SAM 与极端厄尔尼诺组合下的异常风致海冰生成。

来源：https://www.nature.com/articles/s41561-020-00655-3

---

#### Zhou2023 — Zhou et al. (2023), *Nature Climate Change* 13, 701–709

> 🟡 末位作者拼写 `Osterhus`→`Østerhus`（见第一部分）。

**主要内容**：威德尔海底层水自 1992 年以来减少约 30%，由北风趋势导致海冰生成下降 >40% 驱动。

**引用位置**：第248行 —— "observational evidence linking Weddell Sea bottom water reduction to wind-driven sea ice decline [Zhou2023]"。

**原文原句**（标题/摘要）：
- > "Slowdown of Antarctic Bottom Water export driven by climatic wind and sea-ice changes"（标题，直接支持引用论点）

来源：https://www.nature.com/articles/s41558-023-01695-4

---

<a name="d-古气候与代用"></a>
### D. 古气候与放射性碳代用

#### SigmanBoyle2000 — Sigman & Boyle (2000), *Nature* 407, 859–869

**主要内容**：综述冰期—间冰期大气 CO₂ 变化机制；冰期较间冰期低约 80–100 ppmv。

**引用位置**：第80行 —— "Glacial-interglacial cycles feature dramatic changes in atmospheric CO2 [SigmanBoyle2000,Kohfeld2005]"。

**原文原句**：
- > "The concentration of carbon dioxide (CO2) in the atmosphere has varied in step with glacial/interglacial cycles."
- > "During peak glacial periods, atmospheric CO2 is 80–100 p.p.m.v. lower than during peak interglacial periods, with upper and lower limits that are reproduced in each of the 100-kyr cycles."

来源：https://courses.washington.edu/pcc588/readings/Sigman_Boyle-Glacial_CO2_Review-Na00.pdf

---

#### Kohfeld2005 — Kohfeld, Le Quéré, Harrison, Anderson (2005), *Science* 308, 74–78

**主要内容**：分析海洋生产力沉积记录检验生物泵贡献；铁施肥等机制最多解释约一半的 CO₂ 下降。

**引用位置**：第80行 —— 与 SigmanBoyle2000 并列（冰期—间冰期 CO₂ 变化）。

**原文原句**：
- > "It has been hypothesized that changes in the marine biological pump caused a major portion of the glacial reduction of atmospheric carbon dioxide by 80 to 100 parts per million..."
- > "Iron fertilization and associated mechanisms can be responsible for no more than half the observed drawdown."

来源：https://pubmed.ncbi.nlm.nih.gov/15802597/

---

#### huybrechts1990antarctic — Huybrechts (1990), *Annals of Glaciology* 14, 115–119

**主要内容**：用 3D 热力-力学冰盖模式模拟整个南极冰盖在末次冰期—间冰期旋回中的响应；最大波动在西南极，全新世去冰川化贡献全球海平面约 12–15 m。

**引用位置**：第80行 —— "ice sheet extent [huybrechts1990antarctic]"。

**原文原句**：
- > "A complete three-dimensional thermo-mechanical ice-sheet model for the entire Antarctic ice sheet ... is employed to simulate the response of the ice sheet during the last glacial-interglacial cycle with respect to changing environmental conditions."
- > "In line with glacial geological evidence, the most pronounced fluctuations are found in the West Antarctic ice sheet..."

来源：https://www.cambridge.org/core/journals/annals-of-glaciology/article/0F7F21233D89BA8EB3E682F2CAB6B876

---

#### rafter2022global — Rafter et al. (2022), *Science Advances* 8, eabq5434

**主要内容**：用海洋化石放射性碳建立末次冰期以来深海环流基准；冰期最贫¹⁴C 水体位于太平洋海底深度（非现今中层），指示冰期深海翻转减缓+太平洋环流构型"翻转"。

**引用位置**：第80行 —— "deep ocean circulation [rafter2022global]"。

**原文原句**：
- > "Using new and published marine fossil radiocarbon (14C/C) measurements ... we establish several benchmarks for Atlantic, Southern, and Pacific deep-sea circulation and ventilation since the last ice age."
- > "...best explained by a slowdown in glacial deep-sea overturning in addition to a \"flipped\" glacial Pacific overturning configuration."

来源：https://pmc.ncbi.nlm.nih.gov/articles/PMC9668286/ （开放全文）

---

#### Skinner2010 — Skinner, Fallon, Waelbroeck, Michel, Barker (2010), *Science* 328, 1147–1151

**主要内容**：南大洋大西洋扇区放射性碳证据；末次冰期环南极深水相对大气年龄比现今老两倍以上。

**引用位置**：第250行 —— "glacial deep ocean radiocarbon ages increased globally, with strongest signals in the Pacific and Southern Ocean [Skinner2010,Skinner2017]"。

**原文原句**：
- > "We show that during the last glacial period, deep water circulating around Antarctica was more than two times older than today relative to the atmosphere."
- > "...the dissipation of this old and presumably CO2-enriched deep water played an important role in the pulsed rise of atmospheric CO2..."

来源：https://pubmed.ncbi.nlm.nih.gov/20508128/

---

#### Skinner2017 — Skinner et al. (2017), *Nature Communications* 8, 16010

> ⚠️ **数值范围注意**：标志数值是**全球平均 ~689±53 ¹⁴C-yr**；正文"太平洋1500年"具体值不直接来自本文（见第一部分）。

**主要内容**：用全球海-气放射性碳失衡阵列证明 LGM 深海碳平均滞留时间增加约 689±53 ¹⁴C-yr，中深层出现广泛年龄极大值（"鼓包"），最贫¹⁴C 水位于南大洋。

**引用位置**：第80、231、250行 —— ~689±53 ¹⁴C-yr；模拟约1000年与之同量级；与放射性碳代用一致。

**原文原句**：
- > "Here we use a global array of ocean–atmosphere radiocarbon disequilibrium estimates to demonstrate a ∼689±53 14C-yr increase in the average residence time of carbon in the deep ocean at the LGM."
- > "The most striking aspect of the observed global LGM radiocarbon ventilation profile is the existence of a mid-depth bulge in ocean interior ventilation ages."
- > "the most radiocarbon-depleted waters at the LGM can be found in the Southern Ocean, rather than the North Pacific as today."

来源：https://pmc.ncbi.nlm.nih.gov/articles/PMC5511348/ （开放全文）

---

#### Koeve2015 — Koeve, Wagner, Kähler, Oschlies (2015), *GMD* 8, 2079–2094

> ⚠️ **bib 作者列表错误**（见第一部分）。内容引用精确正确。

**主要内容**：用模式与理想示踪剂探讨天然¹⁴C 定年；整体¹⁴C 年龄由环流(老化)分量与预成¹⁴C 年龄分量（大气-海洋平衡不完全）两部分主导。

**引用位置**：第250行 —— "benthic-planktonic 14C ages include both circulation age and preformed age arising from incomplete atmosphere-ocean equilibration [Koeve2015]"。

**原文原句**：
- > "globally, bulk 14C-age is dominated by two equally important components, one associated with ageing, i.e. the time component of circulation, and one associated with a 'preformed 14C-age'."
- > "The latter quantity exists because of the slow and incomplete atmosphere–ocean equilibration of 14C particularly in high latitudes where many water masses form."

来源：https://gmd.copernicus.org/articles/8/2079/2015/ （开放全文）

---

#### Weber2007 — Weber et al. (2007), *Climate of the Past* 3, 51–64

**主要内容**：9 个 PMIP 耦合模式中分析 AMOC 对 LGM 的响应（经流函数/水团密度差诊断），各模式分歧大；主控因子是 AABW–NADW 密度差。

**引用位置**：第82行 —— "Existing paleoclimate modeling studies have largely diagnosed circulation changes through overturning streamfunctions, water mass volumes, or deep-ocean tracer distributions [Weber2007,Wainer2012]"。

**原文原句**：
- > "This study analyses the response of the Atlantic meridional overturning circulation (AMOC) to LGM forcings and boundary conditions in nine PMIP coupled model simulations..."
- > "a major controlling factor for the AMOC response is the density contrast between Antarctic Bottom Water (AABW) and North Atlantic Deep Water (NADW)."

来源：https://cp.copernicus.org/articles/3/51/2007/cp-3-51-2007.html （开放全文）

---

#### Wainer2012 — Wainer, Goes, Murphy, Brady (2012), *Paleoceanography* 27, PA3101

**主要内容**：用 NCAR-CCSM3 分析 LGM/MH/PI 三大洋水团形成率变化；LGM NADW 显著减弱，被增强 AAIW/GNAIW/AABW 替代。

**引用位置**：第82行 —— 与 Weber2007 并列（经水团形成率/环流诊断）。

**原文原句**：
- > "The paleoclimate version of the National Center for Atmospheric Research Community Climate System Model version 3 (NCAR-CCSM3) is used to analyze changes in the water formation rates in the Atlantic, Pacific, and Indian Oceans for the Last Glacial Maximum (LGM), mid-Holocene (MH) and pre-industrial (PI) control climate."

来源：https://www.aoml.noaa.gov/phod/docs/Wainer_et_al_2012.pdf （开放全文）

---

#### shi2020early — Shi, Lohmann, Sidorenko, Yang (2020), *The Holocene* 30(7), 996–1015

> 注：仅出现在被注释掉的备选句中，但仍在 bib。

**主要内容**：用 AWI-ESM 在 PI 与早全新世情景下试验，考察模拟 AMOC 对早全新世日射/GHG/地形/融水的敏感性。

**引用位置**：第86行（注释句）—— 古气候模拟侧重环流强度。

**原文原句**：
- > "This paper performs experiments under pre-industrial and different early-Holocene regimes with AWI-ESM ... to examine the sensitivity of the simulated Atlantic meridional overturning circulation (AMOC) to early-Holocene insolation, GHGs, topography ... and glacial meltwater perturbation."

来源：https://epic.awi.de/51623/1/shi2020.pdf （开放全文）

---

#### Tierney2020Paleo — Tierney et al. (2020), *Nature* 584, 569–573

> 注：仅出现在被注释掉的句中。

**主要内容**：数据同化重建 LGM 温度场；全球平均冷却 −6.1°C，对应 ECS 3.4°C。

**引用位置**：第91行（注释句）—— 古气候提供超出现代观测范围的约束。

**原文原句**：
- > "The Last Glacial Maximum (LGM), one of the best studied palaeoclimatic intervals, offers an excellent opportunity to investigate how the climate system responds to changes in greenhouse gases and the cryosphere."
- > "Our assimilated product provides a constraint on global mean LGM cooling of -6.1 degrees Celsius..."

来源：https://pubmed.ncbi.nlm.nih.gov/32848226/

---

#### Li2023Nature — Li, England, Hogg, Rintoul, Morrison (2023), *Nature* 615, 841–847

> ⚠️ "40% by 2050" 注意：摘要为 "accelerate over the next 30 years"，"40%/30年" 来自 ANU 新闻稿引述。正文若用精确"40%"应核对正文页或软化措辞。

**主要内容**：瞬变强迫高分辨率模式表明南极融水将在未来数十年减缓深渊翻转、致深渊变暖老化；高排放下未来30年加速变暖。

**引用位置**：第246行 —— "concurrent Antarctic ice sheet mass loss could substantially modify this response through enhanced freshwater input [Li2023Nature,Silvano2018]"。

**原文原句**：
- > "We find that meltwater input around Antarctica drives a contraction of Antarctic Bottom Water (AABW) ... The reduction in AABW formation results in warming and ageing of the abyssal ocean, consistent with recent measurements."
- > "under a high-emissions scenario, abyssal warming is set to accelerate over the next 30 years."

来源：https://pubmed.ncbi.nlm.nih.gov/36991191/

---

<a name="e-模式与方法"></a>
### E. 模式与方法（AWI-ESM 组件）

> 这些为模式/方案描述性引用，每篇给一条确认原句即可。

#### sidorenko2019evaluation — Sidorenko et al. (2019), *JAMES* 11(11), 3794–3815
**引用位置**：第256行（AWI-ESM2 耦合模式）
- > "A new global climate model setup using FESOM2.0 for the sea ice‐ocean component and ECHAM6.3 for the atmosphere and land surface has been developed."

来源：https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019MS001696

#### stevens2013atmospheric — Stevens et al. (2013), *JAMES* 5(2), 146–172
**引用位置**：第256行（ECHAM6 大气分量）
- > "ECHAM6, the sixth generation of the atmospheric general circulation model ECHAM, is described."

来源：https://www.inscc.utah.edu/~reichler/publications/papers/Stevens2013cn.pdf

#### brovkin2009global — Brovkin et al. (2009), *GRL* 36(7), L07405
**引用位置**：第256行（JSBACH 动态植被）
- > "Large-scale biogeophysical interactions between forests and climate are explored using the Earth System Model of the Max Planck Institute for Meteorology (MPI-ESM) that includes interactive atmosphere, ocean, and vegetation modules."

来源：https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2009GL037543 *(摘要经索引核实；如有机构权限建议复核 PDF)*

#### reick2021jsbach — Reick et al. (2021), *Berichte zur Erdsystemforschung* 240
**引用位置**：第256行（JSBACH 陆面分量/PFTs）
- > "JSBACH is the land component of the atmospheric component ECHAM, and thereby part of the Max Planck Institute Earth System Model (MPI-ESM)."
- > "The diversity of vegetation is represented in JSBACH by so-called \"Plant Functional Types\" (PFTs)."

来源：https://core.ac.uk/download/387902891.pdf

#### danilov2017finite — Danilov, Sidorenko, Wang, Jung (2017), *GMD* 10(2), 765–789
**引用位置**：第256行（FESOM2 有限体积）
- > "It builds upon FESOM1.4 ... but differs by its dynamical core (finite volumes instead of finite elements)"

来源：https://gmd.copernicus.org/articles/10/765/2017/ （开放全文）

#### large1994oceanic — Large, McWilliams, Doney (1994), *Rev. Geophys.* 32(4), 363–403
**引用位置**：第258行（KPP 方案，整体 Richardson 数，非局地输送）
- > "It includes a scheme for determining the boundary layer depth h, where the turbulent contribution to the vertical shear of a bulk Richardson number is parameterized."
- > "This nonlocal \"K profile parameterization\" (KPP) is then verified and compared to alternatives..."

来源：https://ui.adsabs.harvard.edu/abs/1994RvGeo..32..363L

#### timmermann2004parameterization — Timmermann & Beckmann (2004), *Ocean Modelling* 6(1), 83–100
**引用位置**：第265行（Monin-Obukhov 长度尺度混合）
- > "Combinations of the Pacanowski-Philander parameterization or the Ocean Penetrative Plume Scheme with a simple diagnostic model depending on the Monin-Obukhov length yield particularly good results."

来源：https://epic.awi.de/5526

#### gent1990isopycnal — Gent & McWilliams (1990), *JPO* 20(1), 150–155
**引用位置**：第268行（Gent-McWilliams 厚度扩散）
- > "A subgrid-scale form for mesoscale eddy mixing on isopycnal surfaces is proposed for use in non-eddy-resolving ocean circulation models. The mixing is applied in isopycnal coordinates to isopycnal layer thickness, or inverse density gradient..."

来源：https://journals.ametsoc.org/view/journals/phoc/20/1/1520-0485_1990_020_0150_imiocm_2_0_co_2.xml

#### ferrari2008parameterization — Ferrari et al. (2008), *J. Climate* 21(12), 2770–2789
**引用位置**：第268行（近边界涡通量参数化）
- > "Near the bottom and near the surface, however, microscale boundary layer turbulence overcomes the adiabatic, isopycnal constraints for the mesoscale transport."

来源：https://www.giss.nasa.gov/pubs/abs/fe06000k.html

#### redi1982oceanic — Redi (1982), *JPO* 12(10), 1154–1158
**引用位置**：第269行（Redi 等密度面扩散）
- > "...the isopycnal mixing tensor has been transformed from a diagonal second-rank tensor in the isopycnal coordinate system to a tensor containing off-diagonal elements in the geopotential coordinate system."

来源：https://journals.ametsoc.org/view/journals/phoc/12/10/1520-0485_1982_012_1154_oimbcr_2_0_co_2.xml

#### otto2017pmip4 — Otto-Bliesner et al. (2017), *GMD* 10(11), 3979–4003
> 🟡 bib 缺 `number={11}`
**引用位置**：第273行（PMIP4 边界条件）
- > "...mid-Holocene (midHolocene, 6000 years before present) and the Last Interglacial (lig127k, 127 000 years before present)"

来源：https://gmd.copernicus.org/articles/10/3979/2017/ （开放全文）

#### berger1977long — Berger (1977), *Celestial Mechanics* 15(1), 53–74
**引用位置**：第273行（轨道参数计算）
- > "Using these results, a new solution for the long-term variations of the Earth's orbital elements is obtained."

来源：https://link.springer.com/article/10.1007/BF01229048

#### werner2016glacial — Werner et al. (2016), *GMD* 9(2), 647–670
**引用位置**：第275行（LGM/MIS3 由先前冰期态初始化）
- > "The model consists of the fully coupled ECHAM5/MPI-OM atmosphere–ocean model, enhanced by the JSBACH interactive land surface scheme..."
- > "Simulation results under Last Glacial Maximum boundary conditions also fit to the wealth of available isotope records..."

来源：https://gmd.copernicus.org/articles/9/647/2016/ （开放全文）

#### roeckner2004atmospheric — Roeckner et al. (2004), *MPI Report* 354
**引用位置**：第275行（AMIP 大气条件初始化）
- > "The most recent version of the Max Planck Institute for Meteorology atmospheric general circulation model, ECHAM5, is used to study the impact of changes in horizontal and vertical resolution on seasonal mean climate."

来源：https://pure.mpg.de/pubman/faces/ViewItemOverviewPage.jsp?itemId=item_995221

#### levitus2010world — World Ocean Atlas 2009 (Levitus 等, NOAA Atlas NESDIS)
**引用位置**：第275行（WOA 海洋气候态初始化）
- > "...horizontal maps of annual, seasonal, and monthly climatological distribution fields of temperature ... computed by objective analysis of all scientifically quality-controlled historical temperature data in the World Ocean Database 2009."

来源：https://www.ncei.noaa.gov/sites/default/files/2020-04/woa09_vol1_text.pdf

#### tarasov2012data — Tarasov et al. (2012), *EPSL* 315, 30–40
**引用位置**：第273行（GLAC1D 北美冰盖）
- > "...a distribution of high-resolution glaciologically-self-consistent deglacial histories for the North American ice complex calibrated against a large set of RSL, marine limit, and geodetic data."

来源：https://www.atmosp.physics.utoronto.ca/~peltier/pubs_recent/Tarasov...EPSL%20315,%2030-40,%202012.pdf

#### briggs2014data — Briggs, Pollard, Tarasov (2014), *QSR* 103, 91–115
**引用位置**：第273行（GLAC1D 南极冰盖）
- > "...this article presents results from a large-ensemble data-constrained study of Antarctic evolution over the Last Glacial cycle."

来源：https://ui.adsabs.harvard.edu/abs/2014QSRv..103...91B/abstract *(如有机构权限建议复核 Elsevier PDF)*

#### tarasov2003greenland — Tarasov & Peltier (2003), *JGR Solid Earth* 108(B3), 2143
**引用位置**：第273行（GLAC1D 格陵兰冰盖）
- > "We examine the extent to which observations from the Greenland ice sheet combined with three-dimensional dynamical ice sheet models ... can be used to constrain inferences of the Eemian evolution of the ice sheet..."

来源：https://www.atmosp.physics.utoronto.ca/people/lev/g6jgrpub.pdf （开放全文）

---

<a name="f-温室气体冰芯"></a>
### F. 温室气体冰芯重建

> 第273行共同引用："greenhouse gas concentrations are taken from multi-archive reconstructions from ice core records and recent measurements of firn air and atmospheric samples [fluckiger2002high,monnin2004evidence,schilt2010glacial,buiron2011taldice,schneider2013reconstruction,kohler2017156]"

#### fluckiger2002high — Flückiger et al. (2002), *GBC* 16(1), 1010
> 🟡 页码错误（见第一部分）
- > "Here we fill this gap with a high-resolution N2O record measured along the European Project for Ice Coring in Antarctica (EPICA) Dome C Antarctic ice core. On the same ice we obtained high-resolution methane and carbon dioxide records."

来源：https://climatehomes.unibe.ch/~stocker/papers/flueckiger02gbc.pdf

#### monnin2004evidence — Monnin et al. (2004), *EPSL* 224(1-2), 45–54
- > "High resolution records of atmospheric CO2 concentration during the Holocene are obtained from the Dome Concordia and Dronning Maud Land (Antarctica) ice cores."

来源：https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/epica_domec/edc-taylor-ice-sync-noaa.txt

#### schilt2010glacial — Schilt et al. (2010), *QSR* 29(1-2), 182–192
- > "We present records of atmospheric nitrous oxide obtained from the ice cores of the European Project for Ice Coring in Antarctica (EPICA) Dome C and Dronning Maud Land sites shedding light on the concentration of this greenhouse gas on glacial-interglacial and millennial time scales."

来源：https://ui.adsabs.harvard.edu/abs/2010QSRv...29..182S/abstract

#### buiron2011taldice — Buiron et al. (2011), *Climate of the Past* 7(1), 1–16
> ⚠️ 实为**冰芯定年/年代标尺**论文，非 GHG 浓度重建（见第一部分，建议加说明或单列）
- > "Using stratigraphic markers and a new inverse method, we produce the first official chronology of the ice core, called TALDICE-1."

来源：https://cp.copernicus.org/articles/7/1/2011/cp-7-1-2011.html （开放全文）

#### schneider2013reconstruction — Schneider et al. (2013), *Climate of the Past* 9(6), 2507–2523
- > "The record was derived with a well established sublimation method using ice from the EPICA Dome C (EDC) and the Talos Dome ice cores in East Antarctica."

来源：https://cp.copernicus.org/articles/9/2507/2013/ （开放全文）

#### kohler2017156 — Köhler et al. (2017), *ESSD* 9(1), 363–387 ★最佳匹配
- > "Here, we document our best possible data compilation of published ice core records and recent measurements on firn air and atmospheric samples spanning the interval from the penultimate glacial maximum ( ∼ 156 kyr BP) to the beginning of the year 2016 CE."

来源：https://essd.copernicus.org/articles/9/363/2017/essd-9-363-2017.html （开放全文）
*注：此文摘要措辞与你正文几乎逐字对应，是该引用群的最强锚点。*

---

<a name="g-软件工具"></a>
### G. 软件工具

#### xbudget2024 — Drake, *xbudget*, github.com/hdrake/xbudget
**引用位置**：第280、332行 —— 表面密度倾向分解工具。**URL 正确**。
- GitHub About：> "Helper functions and meta-data conventions for wrangling finite-volume ocean model budgets."

来源：https://github.com/hdrake/xbudget

#### xwmt2024 — *xwmt*
**引用位置**：第307、332行 —— WMT 计算包。
> ⚠️ 规范仓库为 **`github.com/NOAA-GFDL/xwmt`**（bib 正确；正文第332行写 `ocean-eddy-cpt/xwmt` 应改为 `NOAA-GFDL/xwmt`）。
- 仓库描述：> "Python package for water mass transformation analysis that leverages xarray functionality"
- README：> "xWMT is a Python package that provides a framework for calculating water mass tranformations in an xarray-based environment."

来源：https://github.com/NOAA-GFDL/xwmt

---

## 附：核查覆盖总结

- **被引文献总数**：56（去重）
- **元数据完全正确**：约 46
- **元数据严重错误（作者列表）**：4 — Iudicone2008b, Pellichero2018, Bailey2023, Koeve2015
- **元数据次要问题**：6 — otto2017pmip4, Zhou2023, fluckiger2002high, drake2025water/rafter2022global/shi2020early(缺DOI)
- **内容错引/夸大（需改正文）**：3 — Ferrari2014(50–80%CO₂), Skinner2017(1500yr数值), Sallee2010(buoyancy flux措辞)
- **内容措辞偏强（可接受）**：1 — Ferrari2014/Marzocchi2017 thermal→haline
- **正文/bib 或定位不一致**：2 — xwmt 仓库路径, buiron2011taldice 归类
- **内容引用正确**：其余全部（已在第二部分给出原文原句）

> 所有原文原句均逐字摘自上述来源 URL。付费墙文献（Walin1982, Speer2000, Marinov2006, Gray2024, Sallee2010, Abernathey2016, Marzocchi2017 等）摘自经多源交叉核实的官方摘要；开放获取文献摘自正文。
