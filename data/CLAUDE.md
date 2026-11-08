# data：AABW 形成率表

`aabw_formation_rates_final.csv`（各态 × 4 区在阈值密度处的 WMT 及热、海冰、其他淡水三项）和 `aabw_formation_anomalies_final.csv`（相对 PI 的差），由根目录 `plot_wmt_sigma_all_36.88_37.62_37.12.py` 写出，同时出 `figures/plot_wmt_sigma_all_{annual,winter}.pdf`（这两张图没进稿子）。

## 口径（2026-11-08 起，发现 D2）

- 与正文 WMT 主图（`plot_wmt_4regions_5exps_from_100years.py`）同源：读 `{exp}/wmt_results/wmt_{region}_100years_{exp}.nc`，σ2 bin 宽 0.2，逐月 100 年，PI/MH/LIG 2300–2399，LGM 2400–2499，MIS3 2500–2599。LIG 只有 98 年，2318 和 2372 两年模式没有输出，主图也一样。
- 年均是全部月份等权平均（每年都是完整 12 个月，和主图先年均再平均相同）。
- 阈值不变：PI/MH/LIG 36.625，LGM 37.625，MIS3 37.125。它们是旧 0.25 bin 的中心，在 0.2 bin 里落在两个中心之间，线性插值。Total = 热 + 海冰 + 其他淡水，不含 brine 盐通量项（`surface_exchange_flux_nonadvective_salt`），这是管线原有的做法。
- 单位：脚本用 −G/1e9，即按 1000 kg m⁻³ 换体积，比体积转换高 3.5%（climdiag `docs/wmt.md`），各项比例不受影响。

## 以前的版本错在哪

2026-01-08 04:41 写的旧表用的是 12 个月气候态文件、0.25 bin。7 分钟后那条管线改成 0.1 bin 并重算了文件，表却没重算，所以旧表既对不上旁边的文件，也对不上主图。改成同源后 PI 南大洋 Total 8.109 → 8.049，LGM 南大洋 11.341 → 9.345，LGM−PI 南大洋 ΔWMT 3.232 → 1.297。旧值在 git 历史里。

## 交叉核对

`crosscheck_formation_climdiag.py`（`sbatch slurm_crosscheck.sh`，约 9 分钟）用 climdiag 0.5.0（46e6514）`surface_wmt` 从 100 年管线的输入（`{exp}/100years/`）重算全部月份，再用 `formation_rate` 求形成率。结果在 `crosscheck_formation_climdiag.txt`：五个态四个区的 100 年 WMT 文件逐月逐 bin 复现到相对 5e-8，两张表与 climdiag 相差不超过 0.0005 Sv（表里只有 3 位小数）。

## 坑

- climdiag 的 oracle 测试 `test_d2_formation_csvs` 记了这两张表旧版的 md5，重算后那条测试会按设计失败，需要 climdiag 那边更新 oracle，这边不改 climdiag。
- 这两张表的数字目前没有出现在修订稿、回复信或任何稿件图里（2026-11-08 查过 `build/revised.tex`、`RESPONSE_LETTER.md`、`nc-main.tex`、`main.tex`）。
