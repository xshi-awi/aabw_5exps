# 回复信 ↔ 修订稿终查（2026-11-27 13:38 用户要求，经 questions-ef）

基线：TexPage `original_submission/nc-main_submitted.tex`（真正的投稿版，审稿人引的行号对得上）。
修订稿 `revision/revised.tex`，信 `revision/response_new.tex`，核查时为 TexPage 16296ee，改完推成 9163f4f。
逐条明细在 `consistency/`：`R1_table.md`（审稿人 1，49 行）、`R2_table.md`（审稿人 2 与 4，20 行）、`R3_table.md`（审稿人 3，54 行）、`reverse_table.md`（反向：修订稿相对投稿版的 84 处实质改动，信里交代了没有）。下面是汇总，以及每条在 9163f4f 之后的状态。

## 投稿版与本地 nc-main.tex

两者有 35 处不同，本地这份不是投稿版。投稿版有 `\linenumbers`，作者块带两位通讯作者的邮箱，引言前两段、第四段，结果、讨论和方法里各有几句措辞不同，还多引了 sime2013southern、chadwick2022compilation、volker2013responses、Ferrari2014、mandal2021roles、knorr2021salty 六篇。修订稿除作者块外几乎都跟着本地草稿，投稿版最后一轮的措辞在修订稿里丢了 56 处（逐词片段），五篇文献没了（Ferrari2014 别处还在引）。逐处对照在 `submitted_vs_draft.md`。这些在 tracked Word 里显示为改动，信里没有交代。

## 正向：信的条目 ↔ 修订稿

| 审稿人 | 行数 | 一致 | 不一致 | 缺失 | 原稿已有 |
|---|---|---|---|---|---|
| 1 | 49 | 37 | 7 | 4 | 1 |
| 2 与 4 | 20 | 18 | 2 | 0 | 0 |
| 3 | 54 | 46 | 4 | 2 | 2 |
| 合计 | 123 | 101 | 13 | 6 | 3 |

「原稿已有」的三条（Pellichero 5±5 Sv、PI 积分 1500 年等）只是信里提到，不算错。信里引修订稿的 45 段引文逐句和 revised.tex 对过，剩下的 12 个 MISSING 只是写法不同：信里 Author (year) 对应稿件里的数字编号，σ₂、± 的写法，以及她在引文里留的两条 % 注。

## 已直接改（9163f4f，`edits_1127i_consistency.py`）

| 位置 | 原 | 改为 | 依据 |
|---|---|---|---|
| 稿 2.1 | Here the enhanced glacial salinification | Here, the … | R1 #8 信说已改 |
| 稿 2.1 | glacial density anomalies (Fig. \ref{sst}i-l)；compensated (Fig. \ref{sst}j) | (Fig. \ref{wind}c,d)；(Fig. \ref{wind}b) | Fig. 1i-l 是海冰，密度在 A1a-d |
| 稿 2.3 与信引文 | while The Last Interglacial | while the Last Interglacial | 大小写 |
| 稿 2.4 与信引文 | …between 50 and 70°S, north of the winter ice edge (dashed lines in Fig. A7a-c) | …between 50 and 70°S (dashed lines in Fig. A7a-c), north of the winter ice edge | A7 图注：虚线是 15 W m⁻² 等值线，不是冰缘 |
| 稿 2.4、信两处 | brine 51, 12 and 42 mSv | 51, 12 and 41 mSv | sam_push_pull.json 中 LIG 为 41.47，A9 柱标 41 |
| 稿方法 | following \cite{timmermann…}；following \cite{berger1977long} | following Timmermann and Beckmann \cite{…}；following Berger \cite{…}（信引文 following Berger (1977)） | R1 #31 信说已改 |
| 稿 9 处 | \citet{…} 印成「(author?) [40]」 | Sidorenko et al.、Shi et al.（5 处）、Bartlein et al.、Turney and Jones + \cite | 数字编号格式下 \citet 印不出作者 |
| 稿方法 4.1 | AWI-ESM2、ECHAM6、JSBACH、FESOM2 没有全称 | 加全称 | R1 #30 信说已改 |
| 稿 2.6 | Each experiment was integrated for 1000 years | Each paleoclimate experiment… | 方法：PI 1500 年，古气候各 1000 年 |
| 稿 Fig. 9 图注 | σ₂ ≈ 36.5–36.8 (60–80 Sv)，(48–58 Sv) | 36.4–36.8 (70–81 Sv)，(50–59 Sv) | 0.1 格，同 2.2 节；示意图里没有这些数 |
| 稿讨论与信引文 | …Weddell gyre near 68°S. | …68°S (Supplementary Fig. A12). | A13 原先比 A12 先被引，R1 #7 信说补充图按引用顺序排 |
| 信 2 处 | Heuzé (2021) found that almost all CMIP6 models | …28 of 35 CMIP6 models | 稿件讨论写 28 of 35 |
| 信 R1 #27 | σ₂ = 37.0–37.5 … 12–17% | 37.1–37.6 … 17–19% | 0.1 格，引文与稿件已是新数 |
| 信 R3 #19 | σ₂ ≈ 36.5--36.8 | 36.4--36.8 | 同上 |
| 信 R3 #6 | 1.9° further south | one grid row (1.9°) further south | T63 格点 47.564→49.429°S，差 1.865°；信里纬度取一位小数，读者自己减会得 1.8 |
| 信 R1 #11 | described in the same paragraph | described in the next paragraph | 修订稿里海冰和 MLD 是两段 |
| 信 R2 小问题 1 | computed thermodynamically at each mesh | at each surface node | 与方法措辞一致 |

改后：服务器状态编译 0 错误，稿 50 页，信 43 页，无未定义引用，无「(author?)」。补充图首次引用顺序 A1–A13（A6–A10 是一次区间引用）。

## 需要她定（没动）

1. **修订稿底子是投稿前的草稿**（见上节和 `submitted_vs_draft.md`）。要不要把那 56 处恢复成投稿版措辞、补回五篇文献？恢复后 tracked Word 里这些假改动就消失了。
2. **信图 R20 与 Fig. 8 的格宽不一致**（R2 #2）。信图和信里的 −9.2/+6.6 Sv、ORAS5 +5.0/+4.4 Sv、+62.9/+36.8 Sv 用的是 0.2 格、12 个气候月；Fig. 8f 现为 0.1 格、100 年逐月，PI 海冰区峰值 13.0 Sv。审稿人把两张放在一起看会看到差一倍，13 Sv 也超出 Pellichero 5±5 Sv 的上限。可以按 0.1 格重算信图和 ORAS5，也可以在信图注写明「0.2 kg m⁻³ bins as in the submitted version」。
3. **热主导带的说法**（R2 #3）。讨论「only one class … σ₂ = 36.75–37.00」（约 γ 27.33–27.59）是旧格的结果，信写 γ 27.1–27.67 一带。0.1 格、100 年海冰区 PI 按热占比 > 50% 逐格判断，是 σ₂ 36.70–36.90 两格。要不要按 0.1 格重核这句和信里的带。
4. **A11 那句「comparable annual-mean transformation magnitudes to modern observations」**。0.1 格下 PI 南大洋年均峰值从 11.4 升到 17.5 Sv，「comparable」还成不成立。
5. **信里承诺了、稿件没有的**：
   - R1 #14 冰期热占比。2.2 节冰期段仍只写「Thermal forcing operates only secondarily.」，17–19% 只在讨论里。同段「sea ice processes contributing almost the entire of the total WMT」也和 17–19% 有冲突。
   - R1 #26 信说讨论里报告了 Pellichero 的淡水项为热项 2–5 倍并作了比较，稿件里没有。可以补进讨论（先核原文），也可以删掉信里这部分。
   - R1 #25 信说原稿 291、294 行的引文格式「We corrected them」，修订稿这两处没变（只拆开了 302 行的合并引用）。
   - R3 #8 引言里「ideal age … integral proxy for the efficiency of the overturning circulation」没改，信也没回应。稿件里新加了年龄是下限（Millet 2025）和受混合影响的两条说明，信里没提。
   - R3 #15 审稿人要扣掉全球均匀盐度增量的图，稿件只加了一句说明。
6. **信里没有出处的话**：R3 回复「This is a shared model bias in most climate models.」；信里引 Green et al. 2022，稿件已删这条引用。
7. **给 R2 大问题 1 的回复**没说稿件改了什么（讨论三段、Fig. 8、方法里的海冰区计算）。信说「the driving component is mixed」，稿件说 PI 热主导，海冰区峰值处热仍占 76%，两边侧重不同。
8. **结果 2.1「polynyas (Fig. A3)」指向网格分辨率图**，投稿版是「(Fig. A2d,e)」。改回去以后 A3 就只在方法里被引，排到 A13 之后。可以加半句「(Fig. A2d,e), which the ocean mesh resolves at about 25 km near the coast (Fig. A3)」，也可以把网格图挪到补充材料最后、其余补充图号前移一位。
9. **信里没交代的实质改动**（`reverse_table.md` A–J）：
   - WMT 格宽 0.2→0.1，以及随之变的数；
   - 表面密度倾向的单位与符号约定（原 kg m⁻³ month⁻¹、负为增密，现 kg m⁻² month⁻¹、正为增密，2.3 节数字 20→13 等）；
   - SAM 合成阈值，A7/A8 用 ±1.0σ，A6/A9/A10 用 ±1.2σ；
   - SAM 异常的数字和结论（3–8→6–10 Sv，「各态相近」）；
   - 年龄的三条限定；
   - 方法里新增的代用资料评估综述；
   - Fig. 1、A1、A2 实为 6–9 月平均；
   - 摘要末句新结论；
   - 方法 4.4 的 MOC 算法；
   - M(ρ) 定义的更正。

   建议在信开头加一段「其他改动」概括，或分别挂到相关条目下。
10. **讨论里「Moreover, our paleoclimate analysis suggests that projected sea ice decline…」一段**在投稿版是 % 注释，修订稿里成了正文，和紧接的「These paleoclimate results carry direct implications…」重复（同引 Silvano 2018、Li 2023）。
11. 可选的语言小修，没改：
    - 信「For annual mean over the sea ice zone」（三处），「Ours are fully coupled model, so」，给 R4 的那句；
    - 信「Reviewer 1 has the same concern」，R1 第 29 条要的是文献；
    - 2.4 节仍有两处「systematically」（R1 说 SAM 措辞要收）。
12. 已知未办：致谢与基金占位；PI 海冰文件问题（FS/FX）照规矩没碰。

## 修订追踪 Word

`nc_paper/SUBMISSION/manuscript_tracked_20261127.docx`（40 MB，原稿删掉的 5 张图带原图）。
`make_tracked_docx.py` 从两份 tex2docx 转出的 Word 写出：
- 修订标记是 Word 的 w:ins/w:del，作者 Xiaoxu Shi，共 1858 处插入、791 处删除。
- 段落和表格按文字对齐，配对段落逐词标改动。
- 图按图注配对：留在原位的 6 张图用修订稿的图，图注逐词标改动；挪位、新增和删掉的图在原处标删除、新处标插入。
- 公式按 Word 自己的方式在公式内部标改动。

核对结果（`consistency/tracked_check.txt`，用 `check_tracked_docx.py`）：
- 接受全部修订后与修订稿的 Word 逐段相同（293 段、22 张图），拒绝全部修订后与原稿相同（168 段、13 张图）。
- docx skill 的 validate.py --author 通过（没有漏标的改动），XSD 校验通过。
- LibreOffice 能打开并显示修订，接受全部后正文与修订稿一致。只有 6 处改过的公式在 LibreOffice 里显示成空框：式 (1) 新旧两版、新加的 M(ρ) 式、三处行内符号。LibreOffice 不支持公式内部的修订标记，这里没有 Word 可以实测。
