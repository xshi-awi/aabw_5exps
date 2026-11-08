# 图与脚本对照表

本项目的图没有各自的 `fig_*` 文件夹：脚本在根目录，产物写到 `figures/`（gitignored），`nc_paper/build/figures/` 是编译用的副本。这里只记「哪个脚本出哪张图」和改动中做过的决定，具体字号、线宽等以代码为准。SAM 合成图见 `composite_sam/CLAUDE.md`。

## 正文图

- **WMT 4 区 × 5 实验**：`plot_wmt_4regions_5exps_from_100years.py` → `plot_wmt_4regions_5exps_{winter,annual}_from_100years.{pdf,png}`。2026-04-28 改 Nature 风格：Arial、嵌入 TrueType（`pdf.fonttype=42`）、顶部共用图例、只在边缘行列标轴、配色 Total `#111111` / Heat `#D7263D` / Sea-ice FW `#1B998B` / Other FW `#2E86AB`。正文和修订稿都引这一版。`_v1_editorial`、`_v2_minimal`、`_v3_dramatic` 三个变体脚本是备选，没有被采用。
- **表面密度倾向 JJA**：`plot_surface_density_tendency_from_raw_data_jja.py` → `surface_{heat,freshwater}_density_tendency_from_raw_jja.pdf`。这版每张图统一色标范围；`_ok.py` 是按变量分色标的旧版，保留不动。单位由 per second 改为 per month（`SEC_PER_MONTH = 30*86400`），色标刻度用 `pick_tick_format` 自动选小数位。第一列 PI 色标窄，用不带括号的短标题 `var_titles_pi`，字号 13（15 会把开头的「PI:」切掉）。FW 图第三行标题 `Other FW (net precipitation + runoff)`。2026-09-21 应审稿人 1 第 17 条补了气候态列标题（原图确实没有）。
- **理想年龄 3900 m**：`plot_age_horizontal_3900m_combined.py` → `age_horizontal_3900m_combined.pdf`。数据在 `LAT_CUTOFF = -50` 以北被有意屏蔽，所以用 `set_circular_boundary(ax)` 裁成圆盘，避免方框四角空白。输出路径原为不存在的 `paper/figures/`，已改为 `figures/`。`age_vertical*.pdf` 在稿子里被注释掉。
- **fig02 MLD 与海冰 JJA**：`plot_climate_patterns_jja.py`。稿子引的是原版 `fig02_climate_mld_seaice_jja.pdf`。另有分色标变体 `plot_figure2_split()`（开关 `seaice_split_only`）→ `fig02_climate_mld_seaice_jja_split.pdf`：MLD 分两组色标，PI/MH/LIG 用 0–400 m，LGM/MIS3 用 0–200 m（用户要求；五个态 JJA MLD p95 约 260–300 m，单一 0–400 色标会饱和）。变体目前没有进稿子。

## 修订新增图（文件名 figR*，与回复信里的 Fig R1.x 编号无关）

- `figR1_wmt_domain_sensitivity.pdf` ← `plot_wmt_domain_sensitivity.py`（数据 `wmt_siz_{exp}.nc`，由 `calc_wmt_siz_pellichero.py` 算）
- `figR2_moc_5exps.pdf` ← `plot_moc_5exps.py`（模式自带 `diag_gmoc.nc`）
- `figR3_age_global.pdf` ← `plot_age_global_basins.py`
- `figS_wmt_common_axis_winter.pdf` ← `plot_wmt_common_axis_supp.py`（审稿人 1 第 15 条要的共用坐标轴版）
- `figR4_sam_push_pull.pdf`、`figR5_sam_wind_ice.pdf` → 见 `docs/sam_push_pull.md`
- `figR6_convection_sites` ← `plot_convection_sites.py`：上排五个态 JJA MLD 极地图加 400 m 等值线；下排 PI 分扇区柱状、五态对流面积、最大 MLD。一次回答 R1#23、R1#10 和 R3 的开阔洋对流质疑。
- `figR7_mld_glacial_zoom` ← `plot_mld_glacial_zoom.py`：LGM/MIS3 MLD 色标收紧到 0–300 m，因为冰期冬季 MLD 在 55°S 以南的 p99 只有 280 / 296 m。按用户批注放在模式局限那一条。
- `figR8_domain_map` ← `plot_domain_map.py`：60°S 以南域 2.07e13 m²，9 月海冰区 1.22e13 m²，两者之差（42.7%）是常年开阔水。
- `figR9_ross_sector` → `docs/ross_sector_seaice.md`；`figR10_deep_trend` → `docs/model_equilibrium.md`
- 2026-09-22 以后据 git 新增：`plot_density_class_comparison.py`、`plot_pi_vs_pellichero.py`、`plot_wmt_merged_3rows.py`、`plot_wmt_oras5_*.py` 等，旧文档未记录「待核」。
