# explore/pmip_westerlies：PMIP4 末次盛冰期南半球西风变化（回复审稿人 3）

审稿人 3 引 Gray et al. 2023，说代用资料显示冰期南半球西风偏赤道且减弱，而模式没有这个特征。这里用 Levante 上现有的 PMIP4 输出，算 LGM 相对 piControl 和相对 midHolocene 的西风位置与强度变化，并和我们自己的 AWI-ESM 放在一起比。2026-11-26 建，探索性，结论未进稿件。

## 文件

脚本是 `pmip_westerlies.py`，分三步在前台跑：`compute <模式>` 逐模式读 CMIP6 原始文件，写 `profiles/<模式>_<实验>.nc`（纬向平均廓线）和 `profiles/<模式>_metrics.json`；`ours` 从本项目的 ECHAM 输出算我们自己的 AWI-ESM；`summary` 汇总成 `results.csv`（每模式一行）和 `summary.txt`（逐模式差值和集合统计）。四个 PMIP4 模式加我们的模式全部跑完不到 3 分钟。环境是 `module load python3/2022.01-gcc-11.2.0`，用 `python3 -s` 运行。

## 数据

PMIP4 的 lgm 和 midHolocene 在 `/work/ik1017/CMIP6/data/CMIP6/PMIP/<机构>/<模式>/<实验>/<成员>/Amon/{uas,tauu,ua}/<网格>/<版本>/`，piControl 在同一棵树的 `CMIP/` 下，海陆比例取各实验自己的 `fx/sftlf`。版本目录取字典序最新。Levante 上有 lgm 的只有 AWI-ESM-1-1-LR、INM-CM4-8、MIROC-ES2L（成员 r1i1p1f2）、MPI-ESM1-2-LR 四个，其余成员都是 r1i1p1f1，这四个都同时有 midHolocene 和 piControl。`/pool/data/CMIP6/data` 就是同一棵树。年份取法是 piControl 取最后 100 年，lgm 和 midHolocene 取全部但最多最后 100 年，结果四个模式三个实验都正好是完整的 100 年（具体年份见 `results.csv`），没有发现填充值。

我们自己的 AWI-ESM 用本项目的 `{pi,mh,lgm}/echam_clim.nc`（var165 是 10 m 纬向风，var172 是海陆掩膜）和 `sam_wind/{pi,mh,lgm}_taux.nc`（var180，1200 个月）。

## 方法和符号

指标一就是论文的口径。取 6、7、8 月平均的 uas，沿全纬圈平均（含陆地），找 30–70°S 内的最大值。纬度给两种，一种是格点最大值所在的那一行，一种是在最大值和左右两点上做抛物线拟合得到的顶点纬度。强度一律用格点最大值。指标二是年平均 tauu，只在 sftlf < 50% 的海洋格点上做纬向平均，再找 30–70°S 最大值。指标三是年平均 850 hPa 的 ua，全纬圈平均，地形以下的缺测格点不参与平均。拟合算法照搬 `historical_zone/src/common/latfind.py` 的 `peak`。

纬度一律带符号，南半球为负。位移定义为 |LGM 纬度| − |参照纬度|，正值表示向极、负值表示向赤道。主值用抛物线拟合纬度，CSV 里 `shiftgrid` 列是格点行的差。强度变化给绝对值和百分比，百分比以参照态为分母。另外加了一个辅助的质心纬度（30–70°S 内正值部分按 cos 纬度加权），列名带 `cen`，用来检验最大值位置在平顶廓线上是否稳定。

## 结果

按本口径重算我们的 AWI-ESM，完全复现了论文的数：JJA 10 m 风 PI 45.70°S 5.828 m/s，MH 45.70°S 5.845，LGM 49.43°S 6.303；海洋 tauu MH 47.56°S 0.1718 N/m²，LGM 49.43°S 0.1951。PI 的 tauu 峰值是 49.43°S 0.1718（拟合 48.64°S），之前没有算过。

四个 PMIP4 模式的 LGM 相对 PI，指标一的位移在 −0.56° 到 +1.12° 之间，平均 +0.13°，两个向极两个向赤道，格点行一个都没有动；强度变化在 −6.2% 到 +0.4% 之间，平均 −2.3%，三个减弱一个增强。指标二的位移在 −1.21° 到 −0.36° 之间，平均 −0.62°，四个都略向赤道；强度在 −3.1% 到 +4.4% 之间，平均 −0.7%。指标三（INM-CM4-8 的 lgm 没有 ua，只有三个模式）位移 −1.12° 到 −0.53°，三个都略向赤道，强度 −1.1% 到 +3.3%。

相对 MH 的结果差不多。指标一位移平均 +1.11°，范围 −0.09° 到 +3.70°，那个 +3.70° 来自 MPI-ESM1-2-LR，原因见下面的坑；强度 −7.8% 到 +3.3%。指标二位移 −0.39° 到 +0.08°，强度 −3.4% 到 +5.9%。指标三三个模式都略向赤道，在 −0.77° 到 −0.49° 之间。

总的看，PMIP4 的冰期西风基本不动，位置变化都在一度上下，强度变化在 ±8% 以内，没有一个接近 Gray et al. 的偏赤道 4.8°（2.9–7.1°）和减弱约 25%。我们的 AWI-ESM 在集合之外而且方向相反：指标一 LGM 相对 PI 向极 4.90°（格点 +3.73°）、增强 8.1%，相对 MH 向极 5.12°、增强 7.8%；指标二相对 PI 向极 0.79°、增强 13.5%，相对 MH 向极 1.08°、增强 13.5%。CMIP6 里的 AWI-ESM-1-1-LR 则几乎不变（指标一相对 PI −0.56°、−0.9%），两者为什么不同没有分析。

## 坑

PMIP3 在 Levante 上找不到可用的风。`/work/kd0956/CMIP5_waterpark/data/cmip5/output1/` 下有 8 个 lgm 模式（CNRM-CM5、IPSL-CM5A-LR、FGOALS-g2、MIROC-ESM、MPI-ESM-P、MRI-CGCM3、GISS-E2-R、CCSM4），但月和日数据都只有 pr 和 tas，`/work/kd0956/PMIP3` 只有 KCM 的 piControl。所以集合只有 4 个 PMIP4 模式，Gray 用的 PMIP3/4 全集没法在本地复现，又不能下载。

MIROC-ES2L 冬季的地面西风很怪。它 6–8 月的纬向平均 uas 在 35°S 达到最大（约 3.6 m/s），50°S 附近只有 2 m/s 左右，而年平均 uas、tauu 和 850 hPa 风的峰值都在 43–46°S。逐月看过，时间轴解码正确，季节变化连续，只海洋格点也一样，所以这是模式本身的特征，不是处理错误。它的指标一测的其实是副热带那支风，引用时要说明。

MPI-ESM1-2-LR 的 JJA uas 廓线在 40–46°S 是平顶（PI 在这几行里只差 0.01 m/s），最大值位置会在相邻几行之间跳。它 LGM 相对 MH 的 +3.70° 就是这样跳出来的，质心纬度只差 +0.21°，这个数不能当真。

本机 python 要用 `python3 -s`，并用 `use_cftime=True`，因为 MIROC-ES2L 的 MH 年份是 8000–8099，INM 用 365_day 日历。`echam_clim.nc` 的时间单位是 `day as %Y%m%d.%f`，要 `decode_times=False`，从 0 数的下标 5、6、7 是 6–8 月（脚本里断言过月份）。`results.csv` 里我们自己那一行的年份写的是 clim，因为 echam_clim 是现成的月气候态，取自哪些年这里没有核。
