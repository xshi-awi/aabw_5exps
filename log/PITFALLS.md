# PITFALLS（跨图、跨文件的坑）

只属于一张图或一个文件夹的坑记在那里：回复信与 docx 的坑见 `nc_paper/CLAUDE.md`，SAM 图的坑见 `composite_sam/CLAUDE.md`。静默出错的一行版在根 CLAUDE.md。

### 极值的空间范围和文字不一致
错在哪：MIS3 最大 MLD 写成 768 m，那是全球最大值；正文说的是 55°S 以南，实为 645 m。
怎么发现：2026-09-21 画对流位置图（figR6）时按区域重算。
怎么处理：`build/revised.tex`、`RESPONSE_LETTER.md`、`NEW_ANALYSIS_RESULTS.md` 三处同改。算极值前先确认范围和要写的文字一致。

### 顺着审稿人的猜测附和
错在哪：R1#13 先写了「冰期 Ross 海偏暖所以增冰受限」，没算。
怎么发现：2026-09-21 分扇区实算，冰期没有 Ross 偏暖，信号在 LIG。
怎么处理：稿子和回复信按数据改写（`docs/ross_sector_seaice.md`）。

### 漂移判据用错量，又差点写成「全部满足」
错在哪：曾把 SST 趋势（−0.03 到 +0.06 K/century）写进稿子当漂移判据。Shi et al. 2022 JCLI 的判据是含陆地的全球地表温度。
怎么发现：回原文核判据措辞。
怎么处理：改用 var167，五个数全列，如实承认 LIG、MIS3 超标（`docs/model_equilibrium.md`）。

### 「致密类」的积分下界没有依据
错在哪：「限定到观测域后 PI 致密类热贡献 69%→61%」来自 σ2 36.6–36.8 以上的积分，不是致密类，信里的数和同一封信里的图相互矛盾。
怎么发现：2026-09-22 做分密度类对比时（git 58fb985）。
怎么处理：改为分密度类结果。`nc_paper/NEW_ANALYSIS_RESULTS.md` §1 仍留着旧数，那个文件未改。

### 冰期密度类对齐方式
错在哪：试过按盐度对齐、按峰值偏移、按面积百分位三种方法对齐冰期密度类，都把 MIS3 热贡献抬到约 70%，因为它们让密度和成分互相抵消着对齐。
怎么处理：按实验设计的盐度增量平移（+0.80 / +0.50 kg m⁻³），不拿自己的输出去拟合（git 58fb985）。

### 旋转网格上的模式自带诊断按旋转纬度分带
错在哪：`awicm2_final384/nod2d.out` 存的是旋转坐标。gmoc_cal12 调 fpost2 时 `rotated_grid=.false.`，PI/MH/LIG 的 `diag_gmoc.nc` 于是按旋转纬度装箱（差到 ±15°，Weddell 陆架被放到 59°S），而冰期两个网格存的是地理坐标，没出错，冰期与间冰期的比较混了两种坐标。fpost2 的行标签还比所跨纬度偏北一格。另外 `mesh_core2` 与 run 实际用的 `awicm2_final384` 垂向层数不同，三维积分不能混用。
怎么发现：climdiag v0.3.0 复现 fpost2 时（总管发现 D3/D6），2026-11-08 在本项目独立核实。
怎么处理：MOC 从原始 w 在地理网格上重算（`data_moc/`）。用任何 fpost2 产物或 ASCII 网格坐标前，先确认网格存的是哪种坐标。

### 后台任务
- 43 GB `echam_mergetime.nc` 用 `run_in_background` 跑 ncks 会被中途杀掉，留下 .tmp 残文件。前台每个实验约 52 秒。
- matplotlib/cartopy 画图脚本放后台跑过一次，输出为空、文件还是旧时间戳，看起来像跑完了。前台正常。
- 读 2300 多个逐年 3D 文件的 `calc_deep_ocean_trend.py`，第一版因深度轴名 `nz`/`zbar` 在第一个实验就崩了，跑了很久才发现。后台任务要早看输出。

### PI 海冰气候态文件被别的 run 覆盖
错在哪：`pi/a_ice_clim.nc` 与 `pi/a_ice_reg.nc` 在 2026-03-15 被换成一份来历不明的 2090–2099 十年平均（对不上 pi_age 及任何已知 PI run），论文 PI 时段 2300–2399 的原件留在 `a_ice_clim_beta.nc`、`a_ice_reg_beta.nc`。其他四个态和 PI 的 sst/sss 等文件都没动。之后读这两个文件的都受影响：修订稿 Fig. 1（4 月）、Fig. 2（3 月）的 PI 海冰，Fig. 4 的 PI 海冰掩膜，`calc_wmt_siz_pellichero.py` 的 PI 海冰区（9 月冰区比正确的大 13%），`plot_ross_sector.py`（已改读 _beta）。
怎么发现：2026-11-22 做卫星海冰对比时，PI 9 月冰范围 12.2 与 100 年逐月算的 9.9 对不上，查时间轴才发现。
怎么处理：未统一重算，列在 `log/OPEN.md` 等用户定。用 PI 海冰前先看 `time:units`，应是 seconds since 2300。

### 本机 python 读时间报错
错在哪：`~/.local` 装了 pandas 2.3，盖住模块自带的 1.4，xarray 2022.3 解 `days since 1900-01-01` 时报 "unable to decode time units"。
怎么处理：`python3 -s` 屏蔽 ~/.local，再把只含 gsw 软链的目录放进 PYTHONPATH（gsw 只装在 ~/.local）。

### 把别人论文的全球数当成南大洋数
错在哪：稿子和信里写 Sidorenko et al. 2019「南大洋暖偏差 1.43 K」。原文 1.43 K 是 LR 历史模拟全球 SST 的 RMSE（对 PHC）；0.25 m 冰厚也没和观测比，只说冰量偏低并归因于积分中南大洋增暖约 1 °C。
怎么发现：2026-11-22 从 Zenodo 的 CC-BY 稿全文核对（`nc_paper/LITERATURE_NOTES.md`）。
