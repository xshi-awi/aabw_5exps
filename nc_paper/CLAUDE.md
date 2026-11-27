# nc_paper：投稿稿、修订稿与回复信

## 哪个文件是什么

- 真正的投稿版在 TexPage `original_submission/nc-main_submitted.tex`（有行号）。本地 `nc-main.tex` 是投稿前的草稿（Nature 格式 sn-jnl.cls），和投稿版有 35 处不同，修订稿是从它改出来的，逐处对照在 `docx_to_tex/pass_20261127c/submitted_vs_draft.md`。`main.tex` 是更早的 AGU 格式版。
- `build/revised.tex` 修订稿。`build/` 自包含（ref.bib、sn-jnl.cls、sn-nature.bst、figures/），整个目录 gitignored。
- `SUBMISSION/` 交付物：01 修订稿、02 latexdiff 追踪版、04 回复信 tex/pdf、05 回复信 docx、`Source_Data.xlsx`（`../make_source_data.py` 生成）、`letter_figs/`（真实目录，非软链）、`sn-jnl.cls`。在该目录直接 `pdflatex 04_response_letter.tex` 两遍即可。
- 分析笔记：`NEW_ANALYSIS_RESULTS.md`（域敏感性、MOC、理想年龄、对流位置、边界条件、漂移的全部数字），`SUGGESTION_TODO.md`（用户 23 处黄色批注清单），`REVISION_PLAN.md`，文献核对 `LITERATURE_NOTES.md`、`LITERATURE_VERIFIED_BY_MAIN.md`、`reviewer_refs_verified.bib`。
- 初版修订用幂等脚本 `apply_revision.py`、`apply_minor.py`、`apply_minor2.py`、`add_figures.py` 套用，每个都断言锚点恰好匹配一次。

## 旧回复信流水线（只管 04/05 文件，response_new.tex 不走这条）

`RESPONSE_LETTER.md` → `md_to_tex.py` + `tex_header.tex` → `SUBMISSION/04_response_letter.tex` → `tex_to_docx.py` → `SUBMISSION/05_response_letter.docx`；R1 草稿走 `RESPONSE_LETTER_R1.md` 和 `*_r1.py`。引文由 `extract_quotes.py`/`apply_quotes.py` 从修订稿程序化截取，绝不手打；`\ref` 要换成审稿人看到的图号；审稿意见原文取自 `comments.txt`，不许省略号截断；改完跑 `check_letter.py`。Unicode 替换表在 `update_letter.py`。

## response_new.tex（2026-11-19 起的新回复信）

- 用户 11-19 要求：回复信改按 texpage-sync skill 的 STYLE 写，内容以合作者改过的 `paper/inbox_20261119/response-JL.docx` 为准，新开 `revision/response_new.tex`，旧的 04/05 文件不动。以后用户在 TexPage 上直接改它，**线上是主版本**：本地 `response_new.tex` 只是推送时的副本，动手前先 pull 并以线上为准。上面那条 markdown 流水线不管这个文件。
- 由 `docx_to_tex/` 生成一次：`dump_docx.py` 拆 docx 的 run（删除线、高亮、批注、上下标），`build_response_new.py` 按 STYLE 写 tex（段落编号写死，只适用于这一份 docx），`check_paras2.py` 拿 PDF 文本逐段核对。删除线删掉；高亮用 ulem 版 `\hl`（soul 3.0 以前在 pdfLaTeX 下碰 UTF-8 会挂）；Word 批注印成红色 `\docxcomment`；图按出现顺序 R1–R15 连续编号，同一张图在不同审稿人处各放一次（题注不同）。
- 这份 docx 是从 11-08 D6 之前的回复信转来的，MOC 数字（−10.0 等）和 MOC 图都是旧的；tex 照 docx 转写，在源文件里用 `% NOTE` 标了 D6/D8 之后的值，等用户定。
- 11-19 用户定：旧回复信的 Word 和 PDF 已从 TexPage 的 revision/ 删除（baa76c5），本地副本保留。
- 每轮的改动都由 `docx_to_tex/pass_<日期>/` 里的脚本套用（锚点断言恰好一处，可在新拉的线上版上重跑）：`edits_1122.py`；`pass_20261125/edits_1125c_{revised,letter}.py`；`pass_20261126{,b,c}/`；`pass_20261127/`；`pass_20261127c/`（另有图字号清单 `fig_font_checklist.md`）。仍需用户定的地方加 `%claude回复（日期）：` 批注。信里引文从修订稿程序化截取（\ref 换 aux 号、\cite 换作者年份），每轮用 `pass_20261125/check_quotes_vs_revised.py <信> <修订稿> <aux>` 逐句核，剩下的 MISSING 只是 ±、σ₂、\citet 之类写法差别。`mhlig.py` 是 MH/LIG 改全称的规则。追踪版脚本 `pass_20261123/make_diff.sh`，11-25 用户说不再需要。
- Word 版：`docx_to_tex/letter_to_word.py <revision 目录> <out.docx>`，用 tex2docx 转换，前后各加一步：先把信里自定义的宏展开成普通 LaTeX 并插入颜色标记，转完再按标记上色（回复蓝、引文蓝斜体缩进、claudenote 品红，图注和表保持黑）。核对用 `compare_word_pdf.py`（逐词比文字和颜色、按 md5 比图序）。11-27 出 `SUBMISSION/response_letter_20261127.docx`，核对结果在 `pass_20261127c/word_check.txt`。
- 修订追踪 Word：`docx_to_tex/make_tracked_docx.py <orig.docx> <rev.docx> <out.docx>`，两份都先用 tex2docx 转（原稿用 TexPage 的 original_submission 编译），Word 修订标记作者 Xiaoxu Shi，图按图注配对；`check_tracked_docx.py` 核「接受全部=修订稿、拒绝全部=原稿」。LibreOffice 把带修订的公式显示成空框，Word 用的是它自己的公式修订写法。11-27 出 `SUBMISSION/manuscript_tracked_20261127.docx`（40 MB）；发邮件用 `_small.docx`（8 MB），由 `shrink_docx_images.py <in> <out> 200 85` 只换图：按 Word 显示宽度降到 200 dpi，地图类转 JPEG 85、线图和示意图留 PNG，document.xml 逐字节不变。
- 附录图按首次引用排序，现为 A1–A13；信里写死的图号（A1–A13、Fig. 5 MOC）改图序时要一起改。
- 11-27 用户要先把审稿人 1、2 的回复和审稿人 3 对齐，对照表在 `pass_20261127/alignment_R1R2_R3.md`。SAM 折中规矩仍有效；2.4 节已比投稿版长一倍，给审稿人 2 的回复不能再说缩短了。
- 写 LaTeX 的编辑脚本一律用原始字符串：旧脚本把 `\t`、`\r` 当成转义，印出过制表符和 esponse。
- **别再用 `sync_texpage.sh push`**：它先 `rm -rf revision` 再按旧清单重建，会把上面删掉的文件推回去，并把 `response_new.tex` 删掉（只保留 docx/txt 上传）。现在改线上文件一律用干净克隆：pull，只 `git add`/`git rm` 写明的文件，推完等 3 分钟核 md5。

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
