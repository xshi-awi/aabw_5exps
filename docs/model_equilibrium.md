# 模式是否达到平衡：spin-up、全球地表温度漂移、深海趋势

问题来自审稿人 3：西风增强会不会是没平衡的产物。结果对我们不全是好消息，已全部如实写进 Methods。

## spin-up 长度

由 GRL 的 Text S1 确认（用户上传的 15 页完整 supplementary，应为 `nc_paper/SUBMISSION/grl_supplementary.pdf`）：PI 1,500 年，4 个古气候实验各 1,000 年，取最后 100 年，与投稿版一致。Text S1 还独立验证了 Table 1 的五个 CO₂ 值和 GLAC-1D 21k/38k 的网格分配。先前以为几篇文献互相矛盾是误判：JCLI 的 1000 年只讲它自己那组实验，CP 的 1300 年只讲它自己的 PI 和 LGM。

## 全球地表温度漂移

判据取自 Shi et al. 2022 JCLI 原文：最后 100 年全球平均地表温度趋势不超过 ±0.05 K/century。用 var167（2 m 气温，含陆地）实测：PI +0.019，MH −0.033，**LIG −0.063**，LGM +0.040，**MIS3 +0.101**。LIG 和 MIS3 超标。

Methods 五个数全列、明确承认超标，再给限定论证：冰期与间冰期信号约 4 K，漂移小两个量级；MH 的西风急流与 PI 完全一致。数据 `drift_check/{exp}_t2m.nc`，详表见 `nc_paper/NEW_ANALYSIS_RESULTS.md` §8。不能用 SST 趋势代替，SST 只覆盖海洋。

## 深海（2000 m 以下）趋势

`calc_deep_ocean_trend.py` 读 `production/{exp}_age/outdata/fesom/` 逐年 3D 温盐（每个实验 396–616 年，共 2300 多个文件），体积加权，输出 `deep_ocean_trend.npz`；出图 `plot_deep_trend.py` → `figures/figR10_deep_trend.pdf`。

| exp | 分析期趋势 K/century | 后半段/前半段趋势比 | 整段记录深海升温 |
|-----|------|------|------|
| PI | 0.009 | 0.41 | - |
| MH | 0.016 | 0.18 | - |
| LIG | 0.007 | 0.13 | - |
| LGM | 0.113 | 1.08 | +0.58 K |
| MIS3 | 0.160 | 1.37 | +0.89 K |

间冰期深海已经平了而且还在减速；两个冰期没有，MIS3 甚至在加速。原因是冰期实验从更早的冰期态初始化，而不是从 PI 分叉。Methods 写明影响范围：ideal age 本来就声明是下界；表面 WMT 诊断是对表面强迫的年际响应，不受慢速深海调整影响。
