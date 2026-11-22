# fig_seaice_obs：PI 南大洋海冰对比卫星观测（回 R1）

## 画什么

`figR21_seaice_obs.{png,pdf}`，2×3。(a) PI 9 月海冰密集度，(b) OSI SAF 1979–2008 9 月，(c) PI 减 OSI SAF（9 月），黑实线为观测 15% 线、朱红虚线为模式 15% 线，点线是三个扇区分界（60°W、79°E、180°）；(d)(e) 2 月；(f) 南半球 extent（实线）与 area（虚线）的季节循环，灰带是观测 30 年逐月 min–max，浅朱红带是 PI 100 年 min–max。a/b/d/e 共用一根 0–100% 色标，c 用 RdBu（红 = 模式冰少）。南极立体投影到 50°S，标了 60°S、70°S 和经度。配色 Okabe-Ito：模式 `#D55E00`，观测黑。

数字全在 `seaice_numbers.txt`（脚本写出，别手改）。

## 数据与时段

- **模式 PI = pi_age 模式年 2300–2399**，读 `pi/100years/a_ice_mergetime.nc`（1200 个月）。和论文里其他 PI 场（fh、fw、MLD、WMT 100years）同一段，和 MH/LIG（2300–2399，LIG 缺 2318、2372 共 98 年）、LGM（2400–2499）、MIS3（2500–2599）一致。`{exp}/a_ice_clim.nc` 与这些 100 年均值逐点一致（差 < 1e-6），PI 的对应文件是 `pi/a_ice_clim_beta.nc`。
- a_ice 是 0–1 的分数，units 属性写 "%" 是错的。
- 网格面积用 `fesom.mesh.diag.nc` 的 `nod_area[0]`，单位 m²（mesh_core2 海洋总面积 364.4×10⁶ km²，对得上）；`nodes` 是弧度；`elem` 从 1 起数。冰期网格的维度叫 `nz`/`nod2`，取法一样。
- **观测**：EUMETSAT OSI SAF 海冰密集度气候数据记录 v3.0（OSI-450-a），doi:10.15770/EUM_SAF_OSI_0013，日值，SH EASE2 25 km LAEA 432×432，格点面积 625 km²（由 xc/yc 间距核过，等积投影所以处处相同）。路径 `/pool/data/ICDC/ice_and_snow/osisaf_iceconc/DATA/YYYY/MM/ice_conc_sh_ease2-250_cdr-v3p0_YYYYMMDD1200.nc`。时段 1979–2008。引用：OSI SAF (2022), Global Sea Ice Concentration Climate Data Record v3.0, EUMETSAT OSI SAF, doi:10.15770/EUM_SAF_OSI_0013；算法论文可配 Lavergne et al. 2019 (The Cryosphere 13, 49–78，v2 记录)。

## 步骤

1. `prep_osisaf_clim.py`（SLURM `slurm_prep_osisaf.sh`，shared 16 核，实跑 37 秒）：逐月把当月有的日值逐点平均，写 `osisaf_sh_monthly_1979_2008.nc`（30 年 × 12 月场、每月天数、月内日 extent 均值）。
2. `plot_seaice_obs.py`（前台跑，约 25 秒）：
   - extent = 密集度 ≥ 15% 的面积和，area = 密集度 × 面积之和，模式在原生网格上算，观测在 EASE2 上算。两边都由月平均场算月值；季节循环取「各年月值的平均」，另给「多年平均场的 extent」和观测「日 extent 的月均」（OSI SAF 指数的做法）作对照。
   - 地图：模式用网格三角形线性插值到 EASE2 格点（重心坐标，LAEA 平面里算）。
   - 冰缘纬度：沿每条经线（0.5° 间隔，80–45°S 每 0.05° 采样）找最北的 15% 穿越点并线性插值，再按扇区对经线取平均。有 polynya 也不会被误判。
   - 观测冰区内的缺冰比例：观测插到模式节点（双线性，靠岸缺值用 50 km 内最近格点），按 nod_area 加权；EASE2 上再算一遍作交叉核对。
   - polynya 检查：EASE2 上把 <15% 的格点做 8 邻接连通，不与 50°S 外海相连的才算封闭，再分「贴岸」和「开阔洋」。

## 结论要点（数字见 txt）

- 南半球 9 月 extent：PI 9.89 对观测 19.11（×10⁶ km²，各年月值平均），约为观测的一半；2 月 0.74 对 3.32。季节循环相位对（9 月最大、2 月最小）。
- 缺冰几乎全在 Weddell（60°W–79°E）和 Adélie 扇区西部：9 月 Weddell 扇区 extent 2.67 对 10.36，冰缘平均偏南 9.7°，30°W–30°E 偏南 13–14°；79–120°E 偏南约 4–7°，120°E 以东降到 2° 以内。Ross 扇区（180°–60°W）5.54 对 5.77，冰缘 −63.6° 对 −63.3°，基本吻合。
- 观测 9 月 15% 线以内，PI 有 49%（EASE2 上 50%）面积不足 15%；平均密集度 38% 对 79%。
- Weddell 环流区（30°W–30°E，60–70°S）9 月 PI 平均密集度 7.5%，观测 96%。这不是封闭 polynya：100 年里没有一年出现 > 0.1×10⁶ km² 的封闭开阔洋 polynya，开阔水与外海连通，冰只剩西 Weddell 和沿岸一条带。和模式在 68°S 附近开阔洋对流一致。

## 坑

- **`pi/a_ice_clim.nc` 不是论文的 PI 时段。** history 写的是 2026-03-15 用 `cdo ensmean` 对 `a_ice.fesom.2090…2099` 十个文件求平均，相对路径，来源 run 查不到：和 pi_age、pi_mix0.1、pi_uni100、eh2pi_ssp245/585、ice_cavity_data/PI_CORE_ssp* 的 2090–2099 都对不上。同一天的 `pi/a_ice_reg.nc` 也是这一版（1° 上 9 月 extent 12.50，beta 版 10.89）。旧版改名成 `_beta` 留着，`_beta` 才是 2300–2399。`plot_climate_patterns_jja.py` 的 `load_variable_reg('pi','a_ice')` 读的就是 `pi/a_ice_reg.nc`，所以正文 Fig. 1 的海冰一行很可能以 2090–2099 为 PI 基准。JJA 面积 PI* 6.57 对 PI 6.00，MH−PI 用 PI* 为 −0.01，用 2300–2399 为 +0.56。本文件夹只用 100years 那份。
- 观测前 1987-08 隔天一个文件；1986 年 4、5 月无数据（这两个月气候态只有 29 年）；1986-06 只有 4 天（24–30 日隔天），1987-12 只有 2 天（12 月 1–2 日），1988-01 从 13 日开始，它把 12 月 max 抬到 13.80（去掉后 13.04）。按要求有几天用几天。
- 由月平均场算的 extent 比日 extent 的月均大（9 月 19.11 对 18.83），多年平均场的 extent 更大（20.04）。模式只有月均，所以和模式比要用月平均场那一列。
- 2 月冰缘只在有冰的经线上平均，PI 2 月 Weddell 只有 7/278 条经线有冰，2 月扇区冰缘数字没有意义，别引。
- `rm` 被 hook 拦，本目录的 `__pycache__/` 由 `.gitignore` 挡住，不用管。

## History

- 2026-11-22 新建：OSI SAF v3.0 气候态、图 R21 第一版、`seaice_numbers.txt`；发现 `pi/a_ice_clim.nc` / `pi/a_ice_reg.nc` 时段问题（未改原文件）。
