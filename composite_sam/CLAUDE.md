# composite_sam：SAM 合成分析

## 任务定义（原根 CLAUDE.md 的任务说明）

- SAM 指数由 `../{exp}/100years/slp*` 计算，用 JJA 平均；变量也取自 `../{exp}/100years/`。
- 合成 = 高 SAM 年减低 SAM 年。阈值 ±1.2σ：高 SAM 为 SAM > mean + 1.2·std，低 SAM 为 SAM < mean − 1.2·std。
- `calc_sam_wmt_composite.py` 第 32 行 `THRESHOLD_STD = 1.2` 一直是实际生效值。2026-05-19 只修了文档：docstring、函数默认参数 `identify_composite_years(threshold_std=1.0→1.2)`（原来被显式参数覆盖，结果不受影响）、两张图注和任务描述。数据不用重算。
- 版式参照 `../plot_wmt_4regions_5exps.py`。热通量图 3 行 5 列：total heat / LW+SW / SH+LH。淡水图 3 行 5 列：total FW / sea ice only / other（prec+evap+snow+runoff）。
- 论文只用 JJA。`calc_sam_wmt_composite_djf.py` 与 DJF 结果存在，但不进论文，也没放进 Zenodo 包（只保留 5 个 `sam_index_<exp>.nc` 作来源）。
- 放在论文哪里：2026-09-09 用户定 SAM 图全部进补充材料，见 `../nc_paper/CLAUDE.md`。

## 脚本与出图

- 热通量：`calc_sam_heatflux_composite_t63.py`（`slurm_sam_heatflux_t63.sh`，ECHAM T63）→ `plot_sam_heatflux_composite_t63.py` → `sam_heatflux_composite_t63_3rows_5cols.{pdf,png}`。
- 淡水：**现役脚本是 `fw_data/plot_sam_fwflux_composite.py`**，读规则网格 `_reg.nc`，contourf 加 cyclic point，方形填满，和热通量图一致。`composite_sam/plot_sam_fwflux_composite.py`（3 月版）用 tripcolor 画原始 FESOM 网格，出来是圆盘、方框四角空白，**不要用**。
- 现役 fw 脚本用 `Path(__file__).parent`，要 `cd composite_sam/fw_data` 再跑；它写到 `composite_sam/`，之后用**绝对路径**复制 PDF 和 PNG 到 `../figures/`（cwd 已在 fw_data 里）。
- 两个画图脚本都只写到 `composite_sam/`，论文引用的是 `figures/` 下的副本，忘了复制就会引用旧图（2026-05 发现过 3 月的旧副本）。
- fw 图的行标题用单行 `['Total FW Flux', 'Sea Ice FW Flux', 'Other FW Flux']`，带 `\n` 的第二行在 18pt 旋转后会和相邻文字重叠。
- WMT 合成图 `wmt_sam_composite_4regions_5exps.pdf`（来自 `calc_sam_wmt_composite.py` 的结果）。
- push/pull 分解和风应力、海冰合成在根目录脚本里，见 `../docs/sam_push_pull.md`。

## 旧稿里从图上读出的数「待核」

2026-01 写 AGU 初稿时 SAM 一节的数字是看 PNG 读出来的，没有经过计算：高 SAM 使 50–65°S 海洋失热增加 >15 W m⁻²；WMT 调制 15–20%；热贡献间冰期 70–80%、冰期 60–65%；淡水响应里海冰项大 3–5 倍；Ross 海最敏感（>20 W m⁻²，3–4 Sv）。投稿版 `nc-main.tex` 第 248 行仍有「15-20%」，修订稿 `build/revised.tex` 已不含该数，改用「3–8 Sv」。要引用这些数前先从数据重算。
