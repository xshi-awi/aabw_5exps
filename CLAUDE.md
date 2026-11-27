# aabw_5exps

AWI-ESM 五个气候态（PI / MH / LIG / LGM / MIS3）南大洋表面水团转换（WMT）的热、盐强迫分解与 SAM 调制。论文已投 Nature Communications，现处大修阶段。远程 origin = https://github.com/xshi-awi/aabw_5exps

## 当前状态（2026-11-27 14:45）

- **在做什么**：NC 大修。修订稿 `nc_paper/build/revised.tex`（同步自 TexPage `revision/revised.tex`）；回复信是 TexPage 上的 `revision/response_new.tex`（线上是主版本，用户在线改，本地副本 `nc_paper/response_new.tex`）。旧的 `RESPONSE_LETTER.md` 流水线和 04/05 文件留着不动。
- **上一步做完了什么**：11-27 终查信与修订稿（`nc_paper/docx_to_tex/pass_20261127c/consistency_table.md`）；她 GQ 定以 TexPage 投稿版为准，投稿版最后一轮的措辞、5 篇文献和基金文字已补回修订稿（TexPage 1423854）。修订追踪 Word `nc_paper/SUBMISSION/manuscript_tracked_20261127b.docx`（发邮件用 `_b_small.docx`，7.8 MB）；信 Word g `response_letter_20261127g.docx`。
- **下一步**：等她定一致性表里余下的条目（其中 2.1 节 A3 指针、信图 R20 的格宽口径最要紧）；之后回填 Zenodo DOI。
- **动手前必须知道的**：
  - 真正的投稿版是 TexPage `original_submission/nc-main_submitted.tex`（有行号，审稿人行号对得上）。本地 `nc_paper/nc-main.tex` 是投稿前的草稿，和投稿版有 35 处不同，修订稿就是从它改出来的（`log/PITFALLS.md`）。`nc_paper/main.tex` 是旧 AGU 版。
  - 改 response_new.tex 前先 pull，以线上为准；用户在改的文件等她停手 10 分钟再推；只推自己改的文件，推完至少等 3 分钟核 md5。别用 `sync_texpage.sh push`。
  - 用户写作新规矩：不用「不是…而是 / X rather than Y / X, not Y」句式；正文 MH、LIG 写全称。
  - 稿件用数字编号引文（sn-nature），`\citet` 印成「(author?)」，写作者名加 `\cite`。
  - `nc_paper/NEW_ANALYSIS_RESULTS.md` §1 的「69%→61%」已被推翻，别引（见 `log/PITFALLS.md`）。

## 数据路径

实验目录名 `{pi,mh,lig,lgm,mis}`，`mis` 即论文里的 MIS3。
- 气候态月均：`./{exp}/`；100 年逐月：`./{exp}/100years/`（SAM 用其中的 `slp*`）
- 表面密度倾向（100 年逐月）：`./{exp}/surface_density_tendency_100years_{exp}.nc`
- WMT：气候态 `./{exp}/wmt*nc`；100 年逐月 `./{exp}/wmt_results_bin0p1/`（σ2 格 0.1，11-27 起论文用这套），旧的 0.2 格在 `./{exp}/wmt_results/wmt_*_100years_{exp}.nc`；9 月海冰区 WMT `./wmt_siz_{exp}.nc`
- 理想年龄：`./{exp}/age_reg.nc`（全球 1°×47 层，单位已是年）
- ECHAM 全量：`./{exp}/echam_mergetime.nc`（约 43 GB，1200 个月）；风应力抽取 `sam_wind/{exp}_taux.nc`；t2m 抽取 `drift_check/{exp}_t2m.nc`
- 原始模式输出：`/work/ba1066/a270064/production/{exp}_age/`（`outdata/fesom/` 逐年 3D 温盐与 w；`diag_gmoc.nc` 模式自带 MOC，PI/MH/LIG 那份分带错了，别用，MOC 用 `data_moc/`）；run 配置 `production/{exp}_age.yaml` 与 `config/*echam*`
- ORAS5 WMT 结果：`wmt_oras5_siz.nc`、`wmt_oras5_so60.nc`（数据说明 `docs/oras5_data_description.md`）
- 网格（用其中的 `fesom.mesh.diag.nc`）：PI/MH/LIG 的 run 用 `/home/a/a270064/bb1029/inputs/awicm2_final384`，它与 `mesh_core2` 水平网格相同但垂向层数不同，表层分析两者都行，三维积分必须用 `awicm2_final384`；LGM `.../mesh_glac1d`；MIS3 `.../mesh_glac1d_38k`

## 环境

- Python：`module load python3/2022.01-gcc-11.2.0`（取自 `composite_sam/slurm_*.sh`）
- LaTeX：`module load texlive/live2025-gcc-13.3.0`（含 pdflatex、bibtex、latexdiff）

## 文件夹索引

- `nc_paper/` 投稿稿、修订稿、回复信及其生成和检查脚本 → `nc_paper/CLAUDE.md`
- `composite_sam/` SAM 指数与 JJA 合成（阈值 ±1.2σ），两个 fwflux 脚本只有一个是现役 → `composite_sam/CLAUDE.md`
- `docs/` 长篇分析，一题一文件，含图与脚本对照表 `docs/figure_registry.md` → `docs/CLAUDE.md`
- `log/` PROGRESS、OPEN、SOLVED、PITFALLS
- `data_moc/` 五个态的全球 MOC（地理纬度分带，D6）→ `data_moc/CLAUDE.md`
- `data/` AABW 形成率表（与 WMT 主图同源，D2）→ `data/CLAUDE.md`
- `figures/` 论文引用的图（gitignored）；根目录 `plot_*.py` 写到这里
- `zenodo_deposit/` Zenodo 数据包（约 1.3 GB，说明见其 `README.md`），别往里放别的文件，会被一起上传
- `revision/` 从 TexPage 取回的合作者 docx 等原件（如 `response_letter_R12_1015.docx`）「待核」

## 术语

- **SIZ（海冰区）**：9 月海冰密集度 15% 等值线以内，复现 Pellichero et al. 2018 的积分域（面积差 2.5%）。论文原域是 60°S 以南。
- **thermal / haline 主导**：WMT 中热通量项与淡水项（海冰项加其他）谁占优；「regime shift」指间冰期偏热、冰期偏盐。
- **PUSH / PULL**：SAM 的两条路径，60°S 向北 Ekman 输运与 65°S 以南沿岸 brine 输入，见 `docs/sam_push_pull.md`。
- **Fig R1.x / R2.x / R3.x**：回复信里按审稿人分别编的图号；`figR*.pdf` 是文件名，两者不对应。

## 全项目的坑（静默出错类，案例细节见 `log/PITFALLS.md`）

- 数据已从 `/work/ba0989/a270064/bb1029/aabw_5exps` 搬到本目录，旧脚本里的硬编码路径读不到东西。新脚本别抄旧路径。
- `awicm2_final384/nod2d.out` 存的是旋转坐标（与地理纬度差 ±15°），直接拿它的 lat 分带或画图就错，地理坐标取 `fesom.mesh.diag.nc`（弧度）。mesh 文件里 lat/lon 可能是弧度。冰期网格的深度轴叫 `nz` 不叫 `zbar`（数值相同）。
- `diag_gmoc.nc` 与 `age_reg.nc` 的深度为负；`diag_gmoc.nc` 用 netCDF4 或 `decode_times=False` 打开。
- `./{exp}/fw_clim.nc` 的 fw **正值 = 海洋失淡水 = brine 输入**（PI 沿岸 JJA +0.164 Sv，DJF −0.470 Sv），与 WMT 管线 `fw2 = -fw*rho` 的取负一致。
- 43 GB 源文件的 ncks 抽取、matplotlib/cartopy 画图脚本都要前台跑。放后台会被中途杀掉或空跑，留下 .tmp 或旧时间戳文件，看起来像成功了。
- 后台长任务早看输出，别等跑完才发现第一个实验就崩了。
- 引用极值时空间范围要和文字一致；附和审稿人的猜测前先算。
- `plot_climate_patterns_jja.py` 的 `jja_months=[5,6,7,8]` 是 6–9 月，不是 JJA（Fig. 1、A1、A2 受影响；11-25 起图注与正文写 June-September，脚本未改）。
- 图按信里的印出宽出图（`\rfig{0.9}` 为 5.81 in，稿件版心 5.15 in），脚本里直接写印出磅数，刻度不小于 7.5 pt。大画布配小字再缩进去，印出来只有 3 pt 左右（11-27 信里 18 张图全部返工，`docs/figure_registry.md`）。
- `pi/a_ice_clim.nc` 和 `pi/a_ice_reg.nc` 是被覆盖的 2090–2099 别的 run，PI 海冰用 `pi/a_ice_*_beta.nc`（2300–2399）。本机 python 要 `python3 -s`，否则 ~/.local 的 pandas 让 xarray 解不开时间（`log/PITFALLS.md`）。
