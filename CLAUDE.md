# aabw_5exps

AWI-ESM 五个气候态（PI / MH / LIG / LGM / MIS3）南大洋表面水团转换（WMT）的热、盐强迫分解与 SAM 调制。论文已投 Nature Communications，现处大修阶段。远程 origin = https://github.com/xshi-awi/aabw_5exps

## 当前状态（2026-11-08 09:45）

- **在做什么**：NC 大修。修订稿 `nc_paper/build/revised.tex`，回复信单一来源 `nc_paper/RESPONSE_LETTER.md`，生成 tex 与 docx，合作者在 TexPage 上看和改。
- **上一步做完了什么**：总管任务 D6、D2（任务书 `~/.claude/manager/tasks/D_round_20261108.md`）。D6：PI/MH/LIG 的 `diag_gmoc.nc` 按旋转纬度分带，已独立核实；MOC 改为从原始 w 在地理网格上重算（`data_moc/`），Fig. 6、Fig R2.5/R3.4、修订稿与回复信里的 MOC 数字全部更新。D2：形成率表改与主图同源（100 年逐月、0.2 bin），climdiag 交叉核对通过（`data/CLAUDE.md`）。这些数字在总管核对前都算初步。
- **下一步**：D6 已推 TexPage（10:05，commit 3d2ad39，回读一致）。D8（上层指数改 20–70°N、Methods 加 4.4 节）只在本地和 GitHub，等总管说推再推。之后填致谢与基金（`revised.tex` 约 360 行仍是 TODO），用户上传 Zenodo 后回填 DOI。其余见 `log/OPEN.md`。
- **动手前必须知道的**：
  - 投稿稿是 `nc_paper/nc-main.tex`（sn-jnl.cls，5 位作者）；修订在 `nc_paper/build/revised.tex`。`nc_paper/main.tex` 是旧 AGU 版，不是投稿稿；根目录 `paper/` 已删。
  - 回复信只改 markdown 源再生成，改前先合并用户在 docx 和 TexPage 上的改动。流程、闸门、写作红线全在 `nc_paper/CLAUDE.md`。docx 闸门的「N paragraph(s) differ from the built docx」是拿 R1 单独草稿比的，数量大不代表有未合并的改动，看阻断判定那一段（见 `log/SOLVED.md`）。
  - `nc_paper/NEW_ANALYSIS_RESULTS.md` §1 的「69%→61%」已被推翻，别引（见 `log/PITFALLS.md`）。

## 数据路径

实验目录名 `{pi,mh,lig,lgm,mis}`，`mis` 即论文里的 MIS3。
- 气候态月均：`./{exp}/`；100 年逐月：`./{exp}/100years/`（SAM 用其中的 `slp*`）
- 表面密度倾向（100 年逐月）：`./{exp}/surface_density_tendency_100years_{exp}.nc`
- WMT：气候态 `./{exp}/wmt*nc`；100 年逐月 `./{exp}/wmt_results/wmt_*_100years_{exp}.nc`；9 月海冰区 WMT `./wmt_siz_{exp}.nc`
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
