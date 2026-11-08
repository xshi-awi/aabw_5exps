# aabw_5exps

AWI-ESM 五个气候态（PI / MH / LIG / LGM / MIS3）南大洋表面水团转换（WMT）的热、盐强迫分解与 SAM 调制。论文已投 Nature Communications，现处大修阶段。远程 origin = https://github.com/xshi-awi/aabw_5exps

## 当前状态（2026-11-08 压缩时整理，旧文档最后一条为 2026-09-21）

- **在做什么**：NC 大修。修订稿 `nc_paper/build/revised.tex`，回复信单一来源 `nc_paper/RESPONSE_LETTER.md`，生成 tex 与 docx，合作者在 TexPage 上看和改。
- **上一步做完了什么**：旧文档止于 09-21（深海平衡诊断、回复信按用户复查改了三处）。git log 显示 09-22 到 10-15 还有 48 个提交没写进旧文档，摘要见 `log/PROGRESS.md` 首条：TexPage 同步与防覆盖闸门，按 Jiping Liu 意见重写 R1 回复，ORAS5 WMT 对比，R3 的再分析建议改为正面回答，回复信 17 张图。10-15 之后的进展「待核」，以 git log 为准。
- **下一步**：填致谢与基金（`revised.tex` 约 360 行仍是 TODO），用户上传 Zenodo 后回填 DOI。其余见 `log/OPEN.md`。
- **动手前必须知道的**：
  - 投稿稿是 `nc_paper/nc-main.tex`（sn-jnl.cls，5 位作者）；修订在 `nc_paper/build/revised.tex`。`nc_paper/main.tex` 是旧 AGU 版，不是投稿稿；根目录 `paper/` 已删。
  - 回复信只改 markdown 源再生成，改前先合并用户在 docx 和 TexPage 上的改动。流程、闸门、写作红线全在 `nc_paper/CLAUDE.md`。
  - `nc_paper/NEW_ANALYSIS_RESULTS.md` §1 的「69%→61%」已被推翻，别引（见 `log/PITFALLS.md`）。

## 数据路径

实验目录名 `{pi,mh,lig,lgm,mis}`，`mis` 即论文里的 MIS3。
- 气候态月均：`./{exp}/`；100 年逐月：`./{exp}/100years/`（SAM 用其中的 `slp*`）
- 表面密度倾向（100 年逐月）：`./{exp}/surface_density_tendency_100years_{exp}.nc`
- WMT：气候态 `./{exp}/wmt*nc`；100 年逐月 `./{exp}/wmt_results/wmt_*_100years_{exp}.nc`；9 月海冰区 WMT `./wmt_siz_{exp}.nc`
- 理想年龄：`./{exp}/age_reg.nc`（全球 1°×47 层，单位已是年）
- ECHAM 全量：`./{exp}/echam_mergetime.nc`（约 43 GB，1200 个月）；风应力抽取 `sam_wind/{exp}_taux.nc`；t2m 抽取 `drift_check/{exp}_t2m.nc`
- 原始模式输出：`/work/ba1066/a270064/production/{exp}_age/`（`outdata/fesom/` 逐年 3D 温盐，`diag_gmoc.nc` 模式自带 MOC）；run 配置 `production/{exp}_age.yaml` 与 `config/*echam*`
- ORAS5 WMT 结果：`wmt_oras5_siz.nc`、`wmt_oras5_so60.nc`（数据说明 `docs/oras5_data_description.md`）
- 网格（用其中的 `fesom.mesh.diag.nc`）：PI/MH/LIG `/home/a/a270064/bb1029/inputs/mesh_core2`；LGM `.../mesh_glac1d`；MIS3 `.../mesh_glac1d_38k`

## 环境

- Python：`module load python3/2022.01-gcc-11.2.0`（取自 `composite_sam/slurm_*.sh`）
- LaTeX：`module load texlive/live2025-gcc-13.3.0`（含 pdflatex、bibtex、latexdiff）

## 文件夹索引

- `nc_paper/` 投稿稿、修订稿、回复信及其生成和检查脚本 → `nc_paper/CLAUDE.md`
- `composite_sam/` SAM 指数与 JJA 合成（阈值 ±1.2σ），两个 fwflux 脚本只有一个是现役 → `composite_sam/CLAUDE.md`
- `docs/` 长篇分析，一题一文件，含图与脚本对照表 `docs/figure_registry.md` → `docs/CLAUDE.md`
- `log/` PROGRESS、OPEN、SOLVED、PITFALLS
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
- mesh 文件里 lat/lon 可能是弧度。冰期网格的深度轴叫 `nz` 不叫 `zbar`（数值相同）。
- `diag_gmoc.nc` 与 `age_reg.nc` 的深度为负；`diag_gmoc.nc` 用 netCDF4 或 `decode_times=False` 打开。
- `./{exp}/fw_clim.nc` 的 fw **正值 = 海洋失淡水 = brine 输入**（PI 沿岸 JJA +0.164 Sv，DJF −0.470 Sv），与 WMT 管线 `fw2 = -fw*rho` 的取负一致。
- 43 GB 源文件的 ncks 抽取、matplotlib/cartopy 画图脚本都要前台跑。放后台会被中途杀掉或空跑，留下 .tmp 或旧时间戳文件，看起来像成功了。
- 后台长任务早看输出，别等跑完才发现第一个实验就崩了。
- 引用极值时空间范围要和文字一致；附和审稿人的猜测前先算。
