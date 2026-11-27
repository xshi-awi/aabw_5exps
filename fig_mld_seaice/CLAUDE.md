# fig_mld_seaice：补充材料 Fig. A2（label `mld`），6–9 月 MLD 与海冰

`plot_mld_seaice_a2.py`（在项目根目录前台跑，约 1 分钟，PYTHONPATH 要带 gsw）→ `figures/fig02_climate_mld_seaice_jja_mld1000.{pdf,png}`，稿件引 pdf，信里用 png（`letter_figs/fig02_mld1000.png`）。数据、平滑、海冰行和 3 月的原图（根目录 `plot_climate_patterns_jja.py` 的 `plot_figure2()`，`figures/fig02_climate_mld_seaice_jja.pdf`）完全一样，只把 a–c 的 MLD 色标放到 1000 m，d–e 仍是 0–400 m，两段各自一条色标。

坑：月份是 `jja_months=[5,6,7,8]`，即 6–9 月，图注写 June-September。PI 海冰仍读被覆盖的 `pi/a_ice_reg.nc`（用户 FS 先不动），换 `_beta` 要等用户定。

## History

- 2026-11-27 10:52 用户要信里 R20 只放 a–e 并加经纬度：加 `--mld-only`，出 `figures/fig02_climate_mld_seaice_jja_mld1000_mldonly.{png,pdf}`，PNG 拷成 `letter_figs/fig02_mld1000_mldonly.png`。版式是 a–c 一行、d–e 居中一行，各带自己的色标，地图印出约 1.64 in，信里印出高约 5.06 in；没有「MLD」行标（标题和色标已写明）。经纬度 6 pt：经线标在图框外；纬度 80–50°S 沿 135°W 横写，带细白边；40°S 只在角上露一段，标在框下沿左侧。d–e 色标够长，「0」标签恢复。10:45 的单行版存在 `versions/*_mldonly_v20261127_onerow.*`。默认两行输出不变（PNG md5 相同，PDF 只差 CreationDate）。
- 2026-11-27 为 A4 打印放大文字：图宽 20→8 in，字号用 `pt()` 按信里印刷磅数给（刻度 8，色标题 8.5 粗，列标题、行标、面板字母 9 粗），面板字母挪到图框左上角外，免得盖住 (i)(j) 冰缘；LGM/MIS3 色标的「0」标签和「50」相撞，省掉标签留刻度。旧版 `figures/versions/fig02_climate_mld_seaice_jja_mld1000_v20261127pre.*`。
- 2026-11-26 新建，审稿人 3 L134（用户「用1000吧，only in abc」）。
