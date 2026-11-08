# nc_paper：投稿稿、修订稿与回复信

## 哪个文件是什么

- `nc-main.tex` 投稿版（Nature 格式 sn-jnl.cls，5 位作者含 Lohmann）。`main.tex` 是更早的 AGU 格式版，**不是**投稿稿。
- `build/revised.tex` 修订稿。`build/` 自包含（ref.bib、sn-jnl.cls、sn-nature.bst、figures/），整个目录 gitignored。
- `SUBMISSION/` 交付物：01 修订稿、02 latexdiff 追踪版、04 回复信 tex/pdf、05 回复信 docx、`Source_Data.xlsx`（`../make_source_data.py` 生成）、`letter_figs/`（真实目录，非软链）、`sn-jnl.cls`。在该目录直接 `pdflatex 04_response_letter.tex` 两遍即可。
- 分析笔记：`NEW_ANALYSIS_RESULTS.md`（域敏感性、MOC、理想年龄、对流位置、边界条件、漂移的全部数字），`SUGGESTION_TODO.md`（用户 23 处黄色批注清单），`REVISION_PLAN.md`，文献核对 `LITERATURE_NOTES.md`、`LITERATURE_VERIFIED_BY_MAIN.md`、`reviewer_refs_verified.bib`。
- 初版修订用幂等脚本 `apply_revision.py`、`apply_minor.py`、`apply_minor2.py`、`add_figures.py` 套用，每个都断言锚点恰好匹配一次。

## 回复信流水线（单一来源，别手改生成物）

`RESPONSE_LETTER.md` → `md_to_tex.py` + `tex_header.tex` → `SUBMISSION/04_response_letter.tex` → `tex_to_docx.py` → `SUBMISSION/05_response_letter.docx`。
R1 单独草稿（据 git 0e2557d）：`RESPONSE_LETTER_R1.md` → `md_to_tex_r1.py` / `tex_to_docx_r1.py`。
旧的 `build_response_tex.py`（从 `comments.txt` 直接生成 tex）已被 markdown 路线取代。

- 引文：`extract_quotes.py` 从 `build/revised.tex` 程序化抽取原文存 `quotes.json`，`apply_quotes.py` 插入回复信。**绝不手打**引文。`\ref{sst}` 一类要映射成审稿人看到的图号（如 Fig. 1i-l），不能留占位符。
- 审稿意见原文来自 `comments.txt`，必须逐字全文，不许省略号截断（用户查出过 13 条被截断）。
- **每次改完必跑 `check_letter.py`**：图号是否连号、交叉引用是否有对应、标号数与图片数是否相等、有无省略号、引文能否追溯回 quotes.json。
- TexPage 协作：`sync_texpage.sh pull/push`，push 前 `check_docx_merged.py` 闸门要过；网页端说明 `texpage_README.md`。用户在 docx 或网页上的改动先并回 markdown 再重建（详见项目 memory）。
- Unicode 转纯文本替换表在 `update_letter.py`（pdflatex 不认 γ、σ₂、≥ 等）。pandoc 把 md 转 PDF 时加 `--resource-path=.`。

## 格式约定（仿 `SUBMISSION/example_response.tex`）

- 审稿人原文普通黑色；我们的回复 `\textcolor{blue}{\textbf{...}}`；引用修订稿原文 `\textcolor{blue}{\textbf{\textit{``...''}}}`。
- example 用 copernicus.cls，本机 texlive 没有且用户要求 NC 模板，所以用 sn-jnl.cls，视觉约定不变。sn-jnl.cls 自带 hyperref，再 `\usepackage{hyperref}` 会 option clash，用 `url` 包。
- 图号按审稿人分别编（R1.x、R2.x、R3.x）。同一张底图被多位审稿人问到时重复出现、各自编号。只改语法或加文字的 comment 不配图。重编号脚本 `renumber_letter_figs.py`，按文档出现顺序编。

## docx 的坑

- `pandoc --from=latex --to=docx` 静默丢掉 `\textcolor`，还漏图。必须走 HTML 中间格式。
- pandoc 2.18 连 HTML 的 inline CSS color 也丢。做法：蓝色文字插不可见哨兵 U+2060，转完解压 `word/document.xml`，给含哨兵的 run 注入 `<w:color w:val="0000CC"/>` 再打包，同时删哨兵。
- 核查 docx：pandoc 写的是 `<w:b />`（带空格），`grep -c '<w:b/>'` 得 0 会误判样式全丢。7 个 media 文件对应 12 个 `r:embed` 是重复图共用文件，不是丢图。

## 用户拍板的决定

- **SAM 折中**（2026-09-09）：SAM 图全部移入补充材料（满足 R2），正文加强 SAM 文字并新做 push/pull 分析（满足 R3）。回复信区别对待：给 R2 只说图已移走，给 R3 只说加强了分析，两段不许串味。
- **Chen et al. 2025 GRL**（Lohmann 两篇都是合作者）：引言和讨论都引。区别如实写：他们纯海洋、两个态，诊断 AABW 体积、流函数、MLD、海冰，不做 WMT 和浮力通量分解。措辞是「我们检验并量化了他们提出的机制」，不是「他们漏了」。
- 热主导带用中性密度表述：画出的带 γ 27.10–27.68，分 bin 诊断为单 bin γ 27.33–27.59（据 git 8610cf7）。
- R3 第一条只留合并对比图，原 R3.1–R3.3 的结论保留为文字（据 git 17238bf）。
- 审稿人态度：R3 最不 positive，只有他从没说过 suitable 之类的话（R1 说 suitable after major revisions，R2 说 would like to support），给他的回复要最用力。

## 写作红线（都被查出过或纠正过）

- 不写「五个实验全部满足 ±0.05 K/century 漂移判据」，LIG 和 MIS3 超标（`../docs/model_equilibrium.md`）。
- MIS3 最大 MLD 在 55°S 以南是 645 m，768 m 是全球值。
- 不引 `NEW_ANALYSIS_RESULTS.md` §1 的「69%→61%」，已推翻。
- 没有任何已发表的 AWI 论文说本模式在开阔洋对流，这一点只能作为我们自己基于输出的诊断来写。Heuze 的 35 个 CMIP6 模式里也没有 AWI 模式。
- 审稿人三处张冠李戴，回复时委婉处理：Millet 2025 的 factor of 2 讲的是 filling time 不是 ideal age spin-up；Toggweiler & Samuels 1995 有两篇，审稿人指的那篇讲海冰 brine 而非冰架融水；Gray 2023 标题的 poleward 讲冰消期转变，审稿人说的是 LGM 状态，审稿人没错（`LITERATURE_NOTES.md` §3、§5）。
- GLORYS 系列（`GLOBAL_MULTIYEAR_PHY_001_030` 月均只有 thetao/so/uo/vo/zos/mlotst/siconc/sithick/bottomT）没有任何表面通量场，做不了 WMT。现在 R3 的再分析建议用 ORAS5 回答，限定条件写明（`../docs/reanalysis_wmt_feasibility.md`）。
