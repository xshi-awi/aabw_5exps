# data_moc：五个态的全球 MOC（地理纬度分带）

正文 Fig. 6、回复信 Fig R2.5 / R3.4、Source Data 的 `Fig6_MOC_cell_strength` 都用这里的产物。画图脚本是根目录 `plot_moc_5exps.py`（`nc_paper/SUBMISSION/` 里有一份拷贝，改了要同步）。

## 产物

- `moc_5exps_geo.nc`（gitignored，约 170 kB，重跑 `sbatch slurm_moc.sh` 两分钟）：`moc(exp, depth, lat)`，Sv，lat 是**所跨的地理纬度**（−79…89，2° 间隔），depth 是界面深度（m，向下为正）。
- `moc_indices.csv`：两个指数及其极值所在位置、20 年逐年标准差，另附 fpost2 行标签口径（`*_fpost2_rows`）供与 climdiag `docs/moc.md` 对照。`moc_indices_yearly.csv` 逐年值。
- `verify_moc_binning.txt`：独立核查结果（见下）。

## 口径

- 原始数据：`production/{exp}_age/outdata/fesom/w.fesom.YYYY01.01.nc`（每年 12 个月），先年均再 20 年平均。
- 计算：climdiag 0.5.0（commit 46e6514）`moc_depth`，global、无掩膜、从北积分，单元纬度取三个节点地理纬度的平均。
- 网格：每个 run 实际用的网格（namelist 的 meshpath）。PI/MH/LIG 是 `awicm2_final384`，LGM `mesh_glac1d`，MIS3 `mesh_glac1d_38k`。
- 窗口规则（五个态统一）：WMT 主图 100 年分析期的最后 20 年，PI/MH/LIG 2380–2399，LGM 2480–2499，MIS3 2580–2599。
- 指数（用户 D1 确认要全球 MOC）：AABW = 2000 m 以下、30°S 以南的最小值；NADW = 500–4000 m、20–70°N 的最大值（D8，2026-11-08 起；之前代码里是 0–70°N，任何稿件文字都没写过窗口，用户说「好像是 20–70°N」）。Methods 4.4 写明了这两个窗口。参照值 `NADW_below1000` = 1000–4000 m、20–70°N 的最大值。

## 为什么不用 diag_gmoc.nc（发现 D6，2026-11-08 核实）

- `pi/mh/lig_age/diag_gmoc.nc` 按**旋转纬度**分带。`awicm2_final384/nod2d.out` 存的是旋转坐标（与地理纬度差 ±15°），gmoc_cal12 调 fpost2 时 `rotated_grid=.false.`，所以直接按旋转纬度装箱。`verify_moc_binning.py` 不用 climdiag，从 ASCII 网格自己反旋转、自己算面积，用旋转纬度装箱能把三个文件复现到 0.0000 Sv，用地理纬度装箱则逐点差 30–33 Sv。
- 地理上 78.6°S 以南没有海，跨 79°S 的输运必为零。PI/MH/LIG 文件在这里有 2.3–2.5 Sv，LGM/MIS3 文件只有 0.05 Sv。
- 旋转坐标下 Weddell 扇区最南只到 63.4°S，陆架节点（地理 74.1°S）落在 59.2°S；Weddell 下沉在地理坐标里位于 72–73°S，旋转坐标里被放到 57–58°S。
- fpost2 行标签比所跨纬度偏北一格（标签 L = 跨 L−2 的输运），五个文件都有。
- 旧窗口各不相同：PI/MH 2380–2399，LIG 2352–2371（文件在缺失的 2372 年停了），LGM 2350–2369（文件是 2320–2369），MIS3 2590–2609（文件是 2560–2609）。LGM 和 MIS3 的年份是用单年 MOC 对文件逐条匹配找出来的。

## 网格到底是哪个

run 用的是 `awicm2_final384`。它和 `mesh_core2` 水平网格相同：节点顺序一致，`nod2d.out` 按 Euler 角 (50, 15, −90) 反旋转后与 `mesh_core2` 差 5e-5°，单元连接一致，表层节点面积相差不到 4e-5。但垂向层数不同：244659 个单元中有 119976 个的 `elvls` 不一样，最多差 21 层。所以只用表层的分析（WMT、表层通量图）用 `mesh_core2` 没问题；三维积分（MOC、体积平均）要用 `awicm2_final384/fesom.mesh.diag.nc`。

## 新旧数字（20 年平均，Sv）

| 态 | AABW 旧 → 新 | NADW 旧文件 0–70°N → 地理 0–70°N → 地理 20–70°N | 20–70°N、1000 m 以下（参照） |
|---|---|---|---|
| PI | −10.0 → −11.7 | 19.1 → 19.5 → 18.5 | 16.3（37°N） |
| MH | −10.1 → −11.3 | 19.1 → 20.4 → 19.8 | 16.6（21°N） |
| LIG | −9.2 → −11.4 | 21.3 → 22.1 → 21.6 | 17.4（21°N） |
| LGM | −2.4 → −2.3 | 20.0 → 20.9 → 20.1 | 19.9（21°N） |
| MIS3 | −1.6 → −1.7 | 21.7 → 21.5 → 21.5 | 21.5（37°N） |

按 fpost2 行标签口径读，PI/MH/LIG 的 AABW 是 −10.93/−10.48/−10.69，与 climdiag `docs/moc.md` 的 job 28820630 一致。

## 两个指数的毛病（没改，口径由用户定）

- 间冰期深层环流最强处在 19°S、3400 m（−13.6 Sv）。「30°S 以南」的 AABW 指数在三个间冰期都落在窗口边上（31°S，3650 m），数值取决于窗口边界放在哪。
- 「NADW」指数改成 20–70°N 后，PI、MH、LIG 仍落在 500 m 顶边（580 m）和 20°N 南边（21°N）。21°N 处流函数从 280 m 往下单调减小，量到的是与副热带浅层环流连在一起的上层，不是北大西洋深层水本身。LGM 落在 21°N、910 m（深度在窗口内部），MIS3 在 37°N、1040 m（两边都在内部）。1000 m 以下的参照值五个态全在 1040 m 顶边，所以加深窗口也不能让间冰期出现独立的 NADW 极大。

## 坑

- 作业脚本里的 `sbatch slurm_moc.sh verify` 只重跑核查，不重算场；核查依赖 `/scratch/a/a270064/aabw_moc/wmean_*.npy`，scratch 会被清，清了就先跑完整版。
