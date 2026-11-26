# SAM 的 push 与 pull

问题来自审稿人 3：风不只影响 Ekman 辐散，也通过海冰输出影响盐泵。2026-09-09 做，正文和回复信都已写入。

## 定义与脚本

- **PUSH**：60°S 纬圈积分的向北 Ekman 输运，由 τx 算得。
- **PULL**：65°S 以南沿岸 brine 输入（来自 `fw_clim.nc` 的 fw，正值 = brine 输入，符号见根 CLAUDE.md）。
- 计算 `calc_sam_push_pull.py` → `sam_push_pull.json`、`sam_push_pull.npy`；出图 `plot_sam_push_pull.py` → `figures/figR4_sam_push_pull.pdf`（4 个 panel：push、pull、两者散点、WMT 响应）。
- 风应力：`sam_wind/{exp}_taux.nc`，用 `ncks -v var180,var181` 从 `{exp}/echam_mergetime.nc` 抽取（var180 = ustr，var181 = vstr，1200 个月）。每个实验约 52 秒、177 MB，必须前台跑。

## 结果（正位相减负位相）

- PUSH 在五个态都增强，8.0–12.8 Sv（PI 11.2 Sv，LGM 8.0 Sv）。
- PULL：PI / LIG 增 51 / 42 mSv，LGM / MIS3 增 24 / 31 mSv。
- 两者跨五个气候态相关 r = 0.77，定量支持审稿人的假说。
- 五个态的 Ekman 响应：PI 11.2、MH 8.9、LIG 12.8、LGM 8.0、MIS3 10.1 Sv；brine 响应 51、12、42、24、31 mSv。**不按间冰期/冰期分组**（2026-11-26 用户指出 MIS3 比 MH 大）。原来写的「冰期 Ekman 响应最弱，因为大范围海冰削弱了风应力向海洋的传递」数据不支持，也从没算过（Ekman 输运用的是 ECHAM 地表风应力 var180，冰区没有扣海冰），稿件和信里已删。随冰盖变的是局地热项与淡水项谁大（`fig_sam_buoyancy/` 的 90--94%）。

## 风应力与海冰合成图

`plot_sam_wind_ice.py` → `figures/figR5_sam_wind_ice.pdf`，2×5：风应力异常和海冰密集度异常。海冰那一行**不是**「近岸减少、往北增加」的偶极子（2026-11-26 量过，`explore/sam_windice_check/numbers.txt`）：冬季密集度 >0.9 的内圈几乎不变，变化都在冰缘带、各扇区有增有减，总冰面积五个态都增 0.07--0.57 百万 km²，冰期偏北且约大一倍。近岸析盐增加只在 A9b 的淡水通量里看得到。风应力异常最大到 +0.23 N m⁻²，色标用 ±0.12；用 ±0.06 会整片饱和。
