# fig_seaice_obs：PI 南大洋海冰对比卫星观测（回 R1）

## 画什么

`figR21_seaice_obs.{png,pdf}`（信里图 R3），2026-11-27 起默认两格、只画 JJA（6–8 月）：(a) PI 海冰密集度，按补充材料 A2 海冰行的画法（1°、<1% 设缺测、按掩膜归一的 σ=1.5 高斯平滑、contourf、NCL BlAqGrYeOrRe 色标 10–100%）；(b) PI 减 OSI SAF（RdBu，红 = 模式冰少），在 EASE2 格点上算。两格都叠观测 15% 线（黑实线）和模式 15% 线（朱红虚线），线用未平滑场，陆地设 NaN，免得线沿海岸画一圈。`--six` 出 11-25 的六格版（9 月、2 月、季节循环）为 `figR21_seaice_obs_six.*`，`--six --plain` 出 11-22 的不平滑画法。南极立体投影到 50°S。配色 Okabe-Ito：模式 `#D55E00`，观测黑。

数字全在 `seaice_numbers.txt`（脚本写出，别手改；重画图不改数字）。

## 数据与时段

- **模式 PI = pi_age 模式年 2300–2399**，读 `pi/100years/a_ice_mergetime.nc`（1200 个月）；1° 场读 `pi/a_ice_reg_beta.nc`。和论文里其他 PI 场（fh、fw、MLD、WMT 100years）同一段。MH/LIG 2300–2399（LIG 缺 2318、2372），LGM 2400–2499，MIS3 2500–2599。
- a_ice 是 0–1 的分数，units 属性写 "%" 是错的。
- 网格面积用 `fesom.mesh.diag.nc` 的 `nod_area[0]`，单位 m²；`nodes` 是弧度；`elem` 从 1 起数。冰期网格的维度叫 `nz`/`nod2`。
- **观测**：EUMETSAT OSI SAF 海冰密集度 CDR v3.0（OSI-450-a），doi:10.15770/EUM_SAF_OSI_0013，日值，SH EASE2 25 km LAEA 432×432，格点面积 625 km²。路径 `/pool/data/ICDC/ice_and_snow/osisaf_iceconc/DATA/YYYY/MM/ice_conc_sh_ease2-250_cdr-v3p0_YYYYMMDD1200.nc`，时段 1979–2008。引用 OSI SAF (2022)；算法论文 Lavergne et al. 2019 (The Cryosphere 13, 49–78)。

## 步骤

1. `prep_osisaf_clim.py`（SLURM `slurm_prep_osisaf.sh`，37 秒）：逐月把日值逐点平均，写 `osisaf_sh_monthly_1979_2008.nc`。
2. `plot_seaice_obs.py`（前台跑，约 25 秒）：extent = 密集度 ≥ 15% 的面积和，area = 密集度 × 面积之和，模式在原生网格上算，观测在 EASE2 上算，都由月平均场算月值。地图里模式用网格三角形线性插值到 EASE2。冰缘纬度沿每条经线（0.5° 间隔）找最北的 15% 穿越点，再按扇区平均。polynya 检查：<15% 的格点 8 邻接连通，不与 50°S 外海相连才算封闭。

## 结论要点（数字见 txt）

- JJA（各年 6–8 月月值的平均）：extent PI 7.87 对观测 16.47（×10⁶ km²），约一半；冰缘 Weddell −67.9° 对 −58.6°（偏南约 9°），Ross 扇区（180°–60°W）−65.2° 对 −64.0°，环极 −65.9° 对 −61.2°。
- 9 月 extent 9.89 对 19.11，2 月 0.74 对 3.32；季节相位对。9 月 Weddell 扇区冰缘偏南 9.7°，30°W–30°E 偏南 13–14°，Ross 扇区差 0.4°。
- 观测 9 月 15% 线以内 PI 有 49% 面积不足 15%。Weddell 环流区（30°W–30°E，60–70°S）9 月 PI 7.5% 对观测 96%，不是封闭 polynya，开阔水与外海连通，和模式在 68°S 附近开阔洋对流一致。

## 坑

- **`pi/a_ice_clim.nc`、`pi/a_ice_reg.nc` 不是论文的 PI 时段**，是 2026-03-15 由来源不明的 2090–2099 求平均覆盖的（11-25 搜遍三个项目目录对不上任何 run）。`_beta` 才是 2300–2399。正文 Fig. 1 和补充材料 A2 的 PI 海冰仍读被覆盖的文件（用户 FS 先不动），所以 A2 的 PI 那格 Weddell 是满圈，本图 (a) 中间有一块没冰。JJA 面积 PI* 6.57 对 PI 6.00。
- 观测前 1987-08 隔天一个文件；1986 年 4、5 月无数据；1986-06 只有 4 天，1987-12 只有 2 天（把 12 月 max 抬到 13.80）。有几天用几天。
- 月平均场算的 extent 比日 extent 的月均大（9 月 19.11 对 18.83），和模式比用月平均场那一列。
- 2 月 Weddell 只有 7/278 条经线有冰，2 月扇区冰缘数字没有意义。
- `rm` 被 hook 拦，`__pycache__/` 由 `.gitignore` 挡住。

## History

- 2026-11-27 用户在信里批注「只画 jja……a 画海冰，b 画差异，其他的不要了……简单对比一段话」：加 `make_figure_jja`，默认出两格 JJA 版（png md5 c8901d29），六格版改由 `--six` 出；旧图存 `versions/figR21_seaice_obs_v20261127pre.*`（ec81f0b2）。信里文字压成一段（6–8 月数字）。TexPage d4d9c79。
- 2026-11-25 用户按钮 FT「用正确 PI 的 SI 画法版」：a/b/d/e 改用 SI 画法（正确 PI），TexPage 9115556。旧版 `versions/figR21_seaice_obs_v20261122b.*`。
- 2026-11-25 `compare_si_weddell.py`：SI 原图 Weddell 满圈来自被覆盖的 PI 文件（框内着色 90%），换正确 PI 只剩 55–67%。不要用 `pi/a_ice_reg.nc` 出信里的图（总管 11-25 定）。
- 2026-11-22 按正文 Fig. 2 画法重画（`versions/figR21_seaice_obs_v20261122a.*`）。
- 2026-11-22 新建：OSI SAF v3.0 气候态、图第一版、`seaice_numbers.txt`；发现 PI 海冰文件时段问题。
