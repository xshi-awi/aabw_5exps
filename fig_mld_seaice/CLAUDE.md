# fig_mld_seaice：补充材料 Fig. A2（label `mld`），6–9 月 MLD 与海冰

`plot_mld_seaice_a2.py`（在项目根目录前台跑，约 1 分钟，PYTHONPATH 要带 gsw）→ `figures/fig02_climate_mld_seaice_jja_mld1000.{pdf,png}`，稿件引 pdf，信里用 png（`letter_figs/fig02_mld1000.png`）。数据、平滑、海冰行和 3 月的原图（根目录 `plot_climate_patterns_jja.py` 的 `plot_figure2()`，`figures/fig02_climate_mld_seaice_jja.pdf`）完全一样，只把 a–c 的 MLD 色标放到 1000 m，d–e 仍是 0–400 m，两段各自一条色标。

坑：月份是 `jja_months=[5,6,7,8]`，即 6–9 月，图注写 June-September。PI 海冰仍读被覆盖的 `pi/a_ice_reg.nc`（用户 FS 先不动），换 `_beta` 要等用户定。

## History

- 2026-11-26 新建，审稿人 3 L134（用户「用1000吧，only in abc」）。
