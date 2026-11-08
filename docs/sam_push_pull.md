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
- **意外发现**：冰期 Ekman 响应反而最弱，尽管冰期平均风应力更强。原因是大范围海冰削弱了风应力向海洋的传递。冰期于是改走 haline 路径，净调制幅度相当但机制不同，和平均态的 regime shift 相呼应。

## 风应力与海冰合成图

`plot_sam_wind_ice.py` → `figures/figR5_sam_wind_ice.pdf`，2×5：风应力异常和海冰密集度异常。风应力异常最大到 +0.23 N m⁻²，色标用 ±0.12；用 ±0.06 会整片饱和。
