# drift_surface_amoc：表层与翻转环流的漂移，以及南极冰盖淡水怎么进海洋

为审稿人 3 的「是否平衡」做的两项核查（2026-11-26）。作者只想讨论表层和 AMOC 的漂移，并想用 SST 代替 2 m 气温。第二项核查回复信里那句「冰架底融化缺失，像多数模式一样以河流径流代替」是否属实。这里的东西还没进稿件，也没 commit。

## 数据与窗口

分析窗口按各态时间轴核过：PI、MH、LIG 为 2300–2399，LGM 为 2400–2499，MIS3 为 2500–2599。LIG 缺 2318 和 2372 两年，密度倾向文件和 `production/lig_age/outdata/fesom/` 都没有，所以 LIG 只有 98 年，回归用真实年份。SST 和淡水通量取自 `{exp}/surface_density_tendency_100years_{exp}.nc`，这是全球文件（节点数与网格一致，面积和 PI 为 3.64e14 m²），逐月 `tos` 加 `areacello`。年均按月长加权。翻转环流取自 `production/{exp}_age/outdata/fesom/w.fesom.YYYY01.01.nc`，先 12 个月平均成年均 w，再算流函数。

## 方法

流函数用 climdiag 0.5.1（911423d）的 `moc_depth`，设置与 `data_moc/calc_moc_5exps.py` 完全相同（run 自己的网格、从北积分、2° 所跨纬度）。我把它从最后 20 年扩到整整 100 年。全球指数最后 20 年与 `data_moc/moc_indices_yearly.csv` 逐年一致，差别只在 CSV 的第三位小数舍入（≤5e-4 Sv）。

climdiag 支持流域掩膜，所以另算了大西洋（含北冰洋）流函数。掩膜几何取用户 `~/ab0246/fpost2/create_mask.ncl` 的经纬度方框，作用在 `fesom.mesh.diag.nc` 的地理坐标上。然后只保留在方框内经网格单元与 30°N、40°W 连通的节点，这样巴拿马湾、加利福尼亚湾、墨西哥和南美的太平洋沿岸以及南非印度洋一侧的零星节点就去掉了（PI 64 个，LGM 18 个，MIS3 21 个）。LGM 和 MIS3 的方框结果与用户的 `mask_NA_lgm.dat`、`mask_NA_38k.dat` 逐点相同。PI 的 `mask_NA_pi.dat` 比方框多 31 个节点，来源不明，没用它。详见 `atlantic_masks_check.txt`。

指数有四个。大西洋上层环流取 500–4000 m、20–70°N 的最大值，窗口与 Methods 相同，只是换成大西洋流函数，作为主指标。`atl_amoc_ncl` 是用户标准 AMOC（大西洋 20–90°N、800–2000 m），与前者几乎一样。全球上层环流和深层（abyssal）环流照 Methods 定义：前者 500–4000 m、20–70°N 最大值，后者 2000 m 以下、30°S 以南最小值（负值，越负越强）。趋势用最小二乘斜率。误差按残差的一阶自相关 r1 放大，有效样本 n_eff = n(1−r1)/(1+r1)（Santer et al. 2000），95% 区间用 n_eff−2 自由度的 t 分布。

## 结果（每世纪趋势 ± 调整后标准误；SD 为年均值标准差）

全球平均 SST 的趋势是 PI +0.017±0.035，MH −0.030±0.039，LIG −0.025±0.039，LGM +0.034±0.042，MIS3 +0.060±0.033 K，都不显著（MIS3 p=0.07），年际 SD 在 0.06–0.10 K。只有 MIS3 略超 Shi et al. 2022 的 ±0.05 K 判据。用 2 m 气温时 LIG −0.063、MIS3 +0.101，两个都超；换成 SST 后 LIG 回到判据内。冰期与 PI 的全球 SST 差 1.5–1.8 K，漂移最多是它的 4%（MIS3）。

50°S 以南 SST 的趋势是 PI −0.048±0.058，MH −0.008±0.055，LIG −0.099±0.079，LGM −0.014±0.039，MIS3 +0.086±0.040 K（MIS3 p=0.04），SD 0.08–0.11 K。冰期比 PI 冷 3.0–3.3 K，漂移不到 3%。

大西洋上层环流的趋势是 PI +1.27±0.61（p=0.04），MH −0.41±0.54，LIG −0.62±1.62，LGM +0.85±0.45（p=0.06），MIS3 +0.83±0.46 Sv（p=0.08），SD 1.0–1.5 Sv。100 年平均为 17.9、18.4、19.5、22.5、23.1 Sv。LGM 和 MIS3 比 PI 强 4.6 和 5.3 Sv，一个世纪的漂移只有这个差的 16–18%。但 MH 与 PI 只差 0.6 Sv，LIG 差 1.6 Sv，与漂移同量级，这两个差不该解读。全球上层环流的趋势在 −0.66 到 +0.60 之间，SE 0.75–1.03，都不显著，SD 2.0–2.7 Sv。

深层环流的趋势是 PI −1.04±0.65，MH +0.62±0.54，LIG −0.21±0.59，LGM −0.27±0.17，MIS3 −0.30±0.22 Sv，都不显著（p 在 0.11–0.73），SD 在间冰期 1.5–1.9、冰期 0.5–0.6 Sv。间冰期约 −11.5 Sv，冰期 −2.3 和 −1.8 Sv，差约 9.2–9.7 Sv，漂移最多约为它的 11%（PI）。三个间冰期之间只差 0.1 Sv 以内，远小于漂移，不能比。完整表见 `numbers.txt`、`trends.csv`、`state_means.csv`。

## 南极冰盖淡水怎么进海洋

代码和配置给出的答案是这样的。JSBACH 在冰川地块上把降水减蒸发整个当作径流（`mo_soil_hydrology.f90` 第 385 行，`pme_glacier = rain + new_snow + evapotrans`），冰盖不存雪。HD 模式把它加进地表径流（`mo_hydrology.f90` 第 1429–1435 行，Mikolajewicz 2009 的注释写明旧的冰川 calving 已被关掉），沿流向送到沿岸入海点，作为 discharge。discharge 经 `collect.f90` 第 12 个耦合场 `hydr_atm` 传到 FESOM（`gen_forcing_couple.F90` 里赋给 `runoff`，注释写 runoff + calving），作为表层液态淡水加入。FESOM 的 `use_landice_water = .false.`，`use_iceberg_water` 没设，默认 `.false.`，没有冰山模块，也没有冰架空腔。JSBACH 的 `glacier_coupling` 没设，默认 `.false.`。yaml 里删掉的 `lgfw` 这个版本根本没有。五个 run 的这些设置相同，yaml 指向同一份代码 `model_code_beta/awiesm-2.1-wiso`，namcouple 都有 `hydr_atm hydr_oce`。

数据对得上。60°S 以南 `friver` 的 100 年平均在 PI、MH、LIG、LGM、MIS3 分别为 0.085、0.083、0.087、0.070、0.069 Sv，约合 2,690 到 2,180 Gt/yr。ECHAM 气候态里南极冰川陆地上的 P−E 估算为 0.077、0.075、0.079、0.060、0.060 Sv，这是粗算（格点平均 P、E 乘陆地比例和冰川比例），比 friver 低 10–15%。`runoff_clim.nc` 复算出同样的 friver。冰山项 `mass_rhs_…_icebergs` 处处为零，但这是 `calc_surface_density_tendency_100years.py` 把 `ficeberg` 直接设成零造成的，不是模式诊断量，不能当证据。PI 在 60°S 以南的净淡水 `wfo` 为 0.32 Sv（降水 0.47，蒸发 −0.23，海冰 −0.01），径流约占四分之一。LGM 的 `wfo` 只有 0.047 Sv，因为海冰输出强（−0.24），径流 0.070 Sv 比净值还大。

径流是表层液态淡水，经耦合器从 T63 沿岸格点插值到 FESOM 节点，分布在沿岸一条带里。PI 有 69% 落在离岸 100 km 以内，96% 在 200 km 以内（冰期 53–56% 和 88–89%），只有 9% 正好落在岸边节点上（`antarctic_runoff_check.txt`）。它有季节变化，PI 在 1 月最小（0.053 Sv），4–7 月最大（0.10–0.105 Sv）。它不带融冰潜热，也不进入深层。

所以作者那句话的事实部分成立，冰盖的淡水确实以径流形式回到海洋，而且总量闭合，不是丢掉了。但它代替的不只是冰架底融化，而是整个冰盖物质平衡（降水减蒸发），也替代了冰山崩解。「多数模式都这样」是文献判断，这里没有核实。建议写法：「Like in many climate models without ice-shelf cavities, the freshwater that leaves the Antarctic ice sheet through basal melt and calving is returned to the ocean as liquid runoff, because precipitation minus evaporation over the ice sheet is routed by the hydrological discharge model to the coast (about 0.085 Sv south of 60°S in PI and 0.07 Sv in the glacial states).」

## 文件

脚本有 `make_atlantic_masks.py`（掩膜）、`calc_moc_100yr.py` 加 `slurm_moc_100yr.sh`（逐年流函数，作业 29281058，每个态约 2 分钟）、`calc_surface_yearly.py` 加 `slurm_surface.sh`（SST、淡水和 ECHAM P−E，作业 29281091），以及在登录节点跑的 `trend_stats.py` 和 `check_antarctic_runoff.py`。逐年序列在 `moc_indices_yearly_{exp}.csv`、`surface_yearly_{exp}.csv`，逐年流函数场在 `moc_yearly_{exp}.nc`（gitignored）。

## 坑

全球上层环流指数在 PI、MH、LIG 有 87–96% 的年份落在窗口边上（21°N、580 m），量到的是和副热带浅层环流连在一起的部分，所以它的 SD 是大西洋指数的 1.3–2.3 倍。讨论 AMOC 漂移用大西洋指数。深层环流指数在间冰期也落在窗口边上（31°S、3650 m）。南大洋 SST 在海冰下被钉在冰点附近，趋势会被压小，所以要和全球 SST 一起报。PI 的大西洋上层环流有 +1.3 Sv/century 的弱正趋势（p=0.04），写稿时别说「所有态都没有趋势」。climdiag 规定的环境是 `/home/a/a270064/sklearn-env/bin/python`（见 climdiag 的 CLAUDE.md），两个 MOC 相关脚本和 `check_antarctic_runoff.py` 用它跑，其余用 module 的 `python3 -s`。
