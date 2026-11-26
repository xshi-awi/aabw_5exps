# mld_coastal_oras5：冬季深混合层在沿岸还是开阔洋（PI 对 ORAS5）

2026-11-26 用户批注（审稿人 3 主意见）：「ORAS5 的对流基本在沿岸冰间湖，我们的也有但弱，大部分在 Weddell 开阔洋」。这里核这句话，数字进了回复信审稿人 1、3 两段和 R2/R11 图注。

## 脚本与数据

都用 `module load python3/2022.01-gcc-11.2.0`、`python3 -s`，各约 20 s。
- `coastal_mld.py` → `numbers.txt`：原始网格、60°S 以南、面积加权。模式读 `fig_formation_location/data/regions_pi.nc`（JJA MLD1、coast_km、PI 冰来自 `a_ice_clim_beta.nc`）。ORAS5 读 `/pool/data/ICDC/ocean_syntheses/oras5/ORCA025/` 的 `somxl010`、`ileadfra`（opa0，1990–2009 JJA，60 个月），面积 e1t·e2t 取自 `mesh/mesh_mask.nc`，海岸线 = 55°S 以南最大陆块（`tmaskutil` 连通域）旁的海洋点，离岸距离用 cKDTree 球面弦长。
- `where_deep.py` → `numbers_where.txt`：深 MLD 面积按纬带、扇区分，及 65°S 以南的沿岸份额。
- `lon_check.py` → `numbers_lon.txt`：按 20° 经度分，最深点位置。

## 结果

- 60°S 以南 MLD > 200 m 的面积：模式 2.05、ORAS5 1.50 ×10¹² m²。ORAS5 的 75% 在 60–64°S，67% 在 140–60°W（主要 80–120°W），不在大西洋扇区；模式 54% 在离岸 Weddell（60°W–30°E）。
- 65°S 以南 MLD > 200 m：ORAS5 87% 在离岸 200 km 内、11% 离岸 Weddell；模式 6% 沿岸、87% 离岸 Weddell。> 150 m 时 ORAS5 81% 沿岸，模式 13%。
- 离岸 200 km 内 MLD > 200 m 的面积：模式 0.05、ORAS5 0.29 ×10¹² m²（> 150 m：0.16 对 0.60）。沿岸带 MLD 面积分位 p90 模式 119、ORAS5 173 m，p99 211 对 281 m。
- ORAS5 最深的点 75.3°S、163.8°E（Terra Nova Bay 一带，约 590 m）和 66.9°S、50°E；模式最深在 70°S、43–47°W（约 990 m）。
- 60°S 以南 MLD ≤ 100 m 的面积：模式 70%、ORAS5 78%。

## 坑

- 原 R2/R11 图注和审稿人 1 回复里「两者都向南极边缘加深、最深的都在大西洋扇区」对 ORAS5 不成立：沿岸 200 km 内面积平均 MLD 模式 85、ORAS5 92 m，比离岸（122、95 m）还浅。11-26 已改。
- ORAS5 沿岸深 MLD 里 JJA 冰密集度 < 0.7 的只占 18%（月均冰场看不出冰间湖），所以稿件只写「along the coast, where coastal polynyas form」，没说识别出了冰间湖。
- 两边 MLD 定义不同（ORAS5 0.01 kg m⁻³ 密度阈值，模式 bulk Richardson 数），逐点差不全是模式误差。
