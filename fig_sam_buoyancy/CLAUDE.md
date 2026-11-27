# fig_sam_buoyancy：SAM 合成的热项与淡水项放在同一密度标尺上

回应 R2「Figs 5 和 6 单位不同，没法比绝对大小」。2026-11-22 建 Fig R2.2（figR22，2 行，只在信里）；2026-11-25 用户「可以统一，确认」后，补充材料 A3/A4 整张换成密度通量版，按引用顺序现编为 **Supplementary Figs. A7、A8**，信里用这两张图替换了 R22。

## 产物

- `plot_sam_density_components.py`（2026-11-25，约 15 s 前台）→ `figA3_sam_heat_density.{png,pdf}`（热：总、辐射、湍流）与 `figA4_sam_fw_density.{png,pdf}`（淡水：总、海冰、其他），各 3 行 × 5 列，1.0σ 合成，**两图同一色标 ±2 kg m⁻² month⁻¹**（30 面板 |值| 98 分位 1.99），A3 第一行紫色虚线是 ΔQ = −15 W m⁻²（直接画 W m⁻² 场）。导入 figR22 脚本的网格、gsw 系数、PI _beta 冰和陆地补点；分量的符号与总量相同（`calc_sam_fwflux_composite.py`：total = −fw，other = evap+prec+snow+runoff，sea ice = total − other，分量和等于总量到 1e-5）。稿件与 TexPage 里改名为 `figures/figS_sam_{heat,fw}_density.pdf`、信里 `letter_figs/figS_sam_{heat,fw}_density.png`。同时写 `numbers_components.txt`（六行各态冰区/无冰区平均 |值|）和 `numbers_heatloss15.txt`（正文 15 W m⁻² 海区与冰期冰缘纬度的出处）。

- `plot_sam_density_tendency.py`：登录节点前台跑约 30 s。`--validate` 用逐月数据按 1.0σ 重算合成并和现成文件比。
- `figR22_sam_density_tendency.{png,pdf}`：2 行 × 5 列（PI MH LIG LGM MIS3），上行热项、下行总淡水项（含海冰），JJA 高 SAM 减低 SAM，**共用一个色标**（±3 kg m⁻² month⁻¹，取两行合并后 |值| 98 分位）。南极立体投影到 50°S，黑线是各态自己的 JJA 15% 冰缘，灰底表示该态网格里没有海洋（冰期显示 GLAC-1D 的陆地，没叠现代海岸线）。原生网格 tricontourf，填色栅格化，PDF 约 0.7 MB。
- `figR22_sam_density_tendency_1p2sigma.{png,pdf}`：同上，±1.2σ 重算（色标 ±4）。
- `numbers.txt`：两种阈值下的面积统计、平均态对照表、按规则自动写的结论，重跑会整份覆盖。

## 数据

- 1.0σ 合成：`composite_sam/heatflux_sam_composite_<exp>.nc` 的 `total_heat_diff`（W m⁻²，T63 最近邻插到网格），`composite_sam/fwflux_sam_composite_<exp>.nc` 的 `total_fw_diff`（mm/day）。节点数与网格一致（126858/92416/102065），脚本有 assert。
- 1.2σ 合成：`<exp>/100years/{92,95,111,120}_mergetime_echam.nc` 与 `fw_mergetime.nc`，年份来自 `composite_sam/sam_index_<exp>.nc` 的 `sam_jja`（std 用 ddof=0，与合成脚本一致，PI 得 10/13 年，等于该文件属性）。
- α、β、cp 与 S：各态 `<exp>/{sst,sss}_clim.nc` 的 JJA（索引 5,6,7，时间戳是月末）平均，gsw 在 p=0 计算，与 Fig 3/4 脚本做法一致。
- 冰区掩膜：`<exp>/a_ice_clim.nc` JJA 平均，0–1 小数（单位写的 % 是错的）。**PI 用 `pi/a_ice_clim_beta.nc`**，见下面的坑。
- 平均态对照：`<exp>/echam_clim.nc`（ymonmean）与 `<exp>/fw_clim.nc`。
- 网格：`/home/a/a270064/bb1029/inputs/{mesh_core2,mesh_glac1d,mesh_glac1d_38k}/fesom.mesh.diag.nc`，`nodes` 是弧度，`elem` 从 1 开始，面积取 `nod_area` 第一层。

## 换算与符号

- 热项 H = −(α/cp)·ΔQ，ΔQ = SW+LW+SH+LH，**正值 = 进入海洋**（核对过：JJA 南大洋 SW 为正，LW、LH、SH 为负）。
- 淡水项 F = −ρ_fw·β·S·ΔFW，ΔFW = −fw（m/s），**正值 = 淡水进入海洋**，ρ_fw = 1000，与 WMT 管线 `wfo = -fw*1000` 相同。FESOM 原始 `fw` 正值 = 海洋失淡水（副热带 JJA −fw 约 −2.7 mm/day，即蒸发大于降水，符合）。
- 两项都是 kg m⁻² s⁻¹（表面密度通量），图里乘 30 天，**正值 = 增密**。
- 统计区域为 50°S 以南全部网格节点，冰区 = JJA 冰密集度 > 0.15，无冰区 = ≤ 0.15，按节点面积加权。

## 结论（2026-11-22，PI 已换 _beta 冰）

逐点比较时，这句话在五个态里都成立。冰区 |F| > |H| 的面积占 90–94%（1.2σ 下 90–93%），面积平均 |F| 是 |H| 的 6–10 倍；无冰区 |H| > |F| 的面积占 71–77%，|H| 是 |F| 的 2.0–2.8 倍。「wherever」略强，冰区有 6–10% 面积热项更大。换成面积积分的净异常就不成立了：F 是近岸增密、外海变轻的偶极子，H 在各扇区正负相抵，所以无冰区的净 |∫F| 在 PI、MH、LIG、MIS3 都大于 |∫H|，LGM 冰区两者相当（−0.40 与 −0.35 ×10⁶ kg s⁻¹）。平均态分区方向相同但弱一些（冰区 |F| > |H| 占 69–81%）。平均态冰区的净 ∫H 在 PI、LIG 大于 ∫F，MH 两者持平。改稿时说「逐点量级」，别说成净贡献。

## 坑

- **Fig 3/4 淡水项差 1000 倍**：`plot_surface_density_tendency_from_raw_data_jja.py` 把 RHO_WATER 约掉，淡水项按 −β·S·FW[m/s] 算，单位是 m/s，比热项小 ρ 倍。正文第 145 行热项 −2×10⁻⁶ 与第 149 行淡水项 +5×10⁻⁹ 相差约 400 倍就是这么来的。乘上 ρ 后淡水项量级与热项相当或更大。WMT 本身不受影响（xbudget 拿到的 wfo 是 kg m⁻² s⁻¹）。本图乘了 ρ_fw，照抄那个脚本会让热项处处占优。
- **单位标签**：α·Q/cp 量纲是 kg m⁻² s⁻¹。Fig 3/4 色标写 kg m⁻³ month⁻¹，caption 写 kg m⁻³ s⁻¹，都只在除以 1 m 水层时才对。本图标 kg m⁻² month⁻¹，要统一的话改 `UNIT` 常量。
- **阈值**：Figs 5/6 画的是 ±1.0σ 合成，caption 写 1.2σ。依据是 15 个合成文件的全局属性 `threshold_std = 1.`、四个计算脚本里的 `THRESHOLD_STD = 1.0`，以及年数（PI 18/17，1.2σ 是 10/13）。`fw_data/*_highsam_*.nc` 的 `n_years` 同为 18/17。push/pull 与 WMT 合成用的是 1.2σ。本图两种阈值都算了，结论不变。
- **ECHAM 码 92/95/111/120 是 over-water 通量**（`trflwac soflwac ahflwac ahfswac`，见 `production/pi_age/outdata/echam/*_echam.codes`），从数值看已按开阔水比例加权（冰密集度 > 0.8 的格点 LW 约 −9 W m⁻²），无冰处与格点平均通量相差 < 0.05 W m⁻²。ECHAM 陆地格点上为 0，网格节点最近邻落到陆地格点时热通量就成了 0（50°S 以南约 1% 面积，集中在南极沿岸）。脚本用 T63 合成里最近的海洋格点（大圆距离）补上。
- **重复计数**：冰区的热项沿用论文做法，取 ECHAM 开阔水通量。冰点海水失热会结冰，这部分作用已经以 brine 形式进了 F。冰点附近 α 很小，H 本来就小，所以影响有限，但冰区的 H 偏大而不是偏小。
- **`pi/a_ice_clim.nc` 已被覆盖**：2026-03-15 被另一个 run 2090–2099 的 10 年平均替换（history 里文件名是 `a_ice.fesom.209x`，没有路径）。论文时段 pi_age 2300–2399 在 `pi/a_ice_clim_beta.nc`，与 `pi/100years/a_ice_mergetime.nc` 的 JJA 平均最大差 8×10⁻⁷。用错文件时 PI 冰区面积是 10.03 而不是 9.02 ×10⁶ km²，冰区比例从 0.91 变成 0.86。脚本 `read_clim` 会检查 history 里有没有 `production/<exp>_age/`，没有就报错。
- `seaice_fw` 在无冰纬度也不为零（高 SAM 平均约 −0.1 到 −0.17 mm/day），说明它是残差项而不是纯海冰，所以这里用 total FW。
- 合成文件 `heatflux_sam_composite_<exp>.nc` 是在经纬度空间（度）里做最近邻，和 `--validate` 的大圆最近邻在个别节点上差到 17 W m⁻²，逐点相关 ≥ 0.9997，面积统计到小数点后两位相同。

## History

- 2026-11-27 A4 可读性：图按信里印出的 5.81 in 宽直接画，字号就是印刷磅值（刻度和图例 8 pt，色标标题和行标题 8.5 pt 粗体，面板字母 9 pt 粗体挪到面板上方，列标题 10 pt），纬度标注只在 (k) 留 50°S、70°S，行标题加换行，图例改一行放色标下；PNG 585 dpi。数据、色标、两个 numbers txt 不变（md5 相同）。旧版在 `versions/*_v20261127pre.*`。需要 gsw 在 PYTHONPATH 上。
- 2026-11-25 新增 `plot_sam_density_components.py` 与 A3/A4 密度版（TexPage 4d59b5f、638d640，信 6e32b68、3a40d9f）。正文 2.4 节的海区据 `numbers_heatloss15.txt`：间冰期失热多 15 W m⁻² 的面积 90–97% 在冰缘以北开阔洋，主要在 60–140°E 与 180–120°W；冰期冰缘纬圈平均 55.2/56.2°S，冰下超过 15 W m⁻² 的只占 1–3%。figR22 不再进信，文件留着。

- 2026-11-22 新建脚本、图和 numbers.txt。PI 冰掩膜由 `a_ice_clim.nc` 换成 `a_ice_clim_beta.nc` 并加 history 闸门，增加 1.2σ 重算和平均态对照。
