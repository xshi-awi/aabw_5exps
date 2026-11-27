#!/usr/bin/env python
"""
从100年月数据计算WMT多年平均并绘制对比图
- 读取wmt_results_bin0p1/wmt_*_100years_*.nc (1200个月, sigma2 bin 0.1 kg m-3;
  2026-11-27 用户定全文改 0.1, 数据由 calc_wmt_100years_bin0p1.py 生成; 旧 0.2 在 wmt_results/)
- 在脚本内计算多年平均
- 生成两张图：全年平均 + 冬季平均（6-8月）
- 所有数据乘以-1
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ============================================================================
# Nature-style publication aesthetics
# ============================================================================
# ---- sizes are set for the printed manuscript (2026-11-27) ---------------------
# Both figures are included at \textwidth = 5.15 in. All font sizes, line widths
# and panel positions below are points on that printed page; pt() converts them
# to the native figure size. Ticks 7.5 pt, labels and legend 8.5 pt, column and
# row titles 9.5 pt bold, panel letters 9 pt bold, all text black.
PRINT_W = 5.15                    # printed width (in)
FIG_W = 7.0                       # native width (in); PNG at 600 dpi = 4200 px
S = PRINT_W / FIG_W               # printed size = native size * S
WP = PRINT_W * 72                 # printed width (pt)


def pt(v):
    return v / S


F_TICK, F_LAB, F_LEG, F_TITLE, F_LET = pt(7.5), pt(8.5), pt(8.5), pt(9.5), pt(9)
LW_AX = pt(0.6)

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': F_LAB,
    'axes.linewidth': LW_AX,
    'axes.edgecolor': 'k',
    'axes.labelcolor': 'k',
    'text.color': 'k',
    'xtick.color': 'k',
    'ytick.color': 'k',
    'xtick.labelsize': F_TICK,
    'ytick.labelsize': F_TICK,
    'xtick.major.width': LW_AX,
    'ytick.major.width': LW_AX,
    'xtick.major.size': pt(2),
    'ytick.major.size': pt(2),
    'xtick.major.pad': pt(1.5),
    'ytick.major.pad': pt(1.5),
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# panel grid in printed points (x from the left, y from the bottom)
NR, NC = 4, 5
L, R, GAP_X = 60, 8, 23          # left margin (row titles, y label), right, gap
PW = (WP - L - R - (NC - 1) * GAP_X) / NC      # panel width (~43 pt)
PH, GAP_Y = 48, 13               # panel height, gap between rows (holds the letters)
BOT = 50                         # below the bottom row: ticks, x label, legend
HP = BOT + NR * PH + (NR - 1) * GAP_Y + 23     # printed height (pt)


def box(x, y, w, h):
    return [x / WP, y / HP, w / WP, h / HP]

# Refined Nature-style color palette
COL_TOTAL   = '#111111'   # near-black
COL_HEAT    = '#D7263D'   # crimson
COL_SEAICE  = '#1B998B'   # teal
COL_OTHERFW = '#2E86AB'   # ocean blue

print('='*80)
print('从100年月数据计算WMT并绘制4个区域 x 5个实验对比图')
print('='*80)
print()

# 定义区域和实验
REGIONS = [
    ('southern_ocean', 'Southern Ocean'),
    ('ross_sea', 'Ross Sea'),
    ('weddell_sea', 'Weddell Sea'),
    ('adelie', 'Adélie Land')
]
# row titles are broken into two lines: one line is longer than a panel is tall

EXPERIMENTS = [
    ('pi', 'PI'),
    ('mh', 'MH'),
    ('lig', 'LIG'),
    ('lgm', 'LGM'),
    ('mis', 'MIS3')
]

# 基础路径
BASE_PATH = Path('/work/ba1066/a270064/cc_projects/aabw_5exps')

# 冬季月份（南半球冬季，6-8月）
WINTER_MONTHS = [6, 7, 8]

# ============================================================================
# 定义绘图函数
# ============================================================================
def plot_wmt_comparison(season_name, month_selector=None):
    """
    绘制WMT对比图

    Parameters:
    -----------
    season_name : str
        季节名称，用于文件名和标题
    month_selector : list or None
        月份列表（1-12），None表示全年平均
    """

    fig = plt.figure(figsize=(FIG_W, FIG_W * HP / WP))
    axes = np.empty((NR, NC), dtype=object)
    for r_ in range(NR):
        for c_ in range(NC):
            axes[r_, c_] = fig.add_axes(box(L + c_ * (PW + GAP_X),
                                            BOT + (NR - 1 - r_) * (PH + GAP_Y),
                                            PW, PH))

    # 子图标签 (a)(b)...
    panel_labels = [chr(ord('a') + i) for i in range(20)]

    # Storage for legend handles (collected from one panel)
    legend_handles = None

    print(f'\n{"="*80}')
    print(f'绘制 {season_name} 数据')
    print(f'{"="*80}')

    for row_idx, (region_key, region_title) in enumerate(REGIONS):
        for col_idx, (exp_key, exp_title) in enumerate(EXPERIMENTS):
            ax = axes[row_idx, col_idx]

            print(f'\n处理: {region_title} - {exp_title}')

            # 构建文件路径（100年数据）
            file_path = BASE_PATH / exp_key / 'wmt_results_bin0p1' / f'wmt_{region_key}_100years_{exp_key}.nc'

            if not file_path.exists():
                print(f'  ⚠ 文件不存在: {file_path}')
                ax.text(0.5, 0.5, 'Data Not Available',
                       ha='center', va='center', fontsize=F_LAB,
                       color='k', transform=ax.transAxes)
                if row_idx == 0:
                    ax.set_title(f'{exp_title}', fontsize=F_TITLE, fontweight='bold',
                                 pad=pt(12), color='k')
                ax.grid(True, alpha=0.25, linestyle='--', linewidth=pt(0.4))
                continue

            # 加载数据
            ds = xr.open_dataset(file_path)
            print(f'  ✓ 已加载: {file_path.name}')
            print(f'  时间维度: {len(ds.time)} 个月')

            # 选择月份
            if month_selector is not None:
                # 选择特定月份（冬季）：先选月份，再resample到年求平均，得到100个年值
                ds_sel = ds.isel(time=ds.time.dt.month.isin(month_selector))
                print(f'  选择月份: {month_selector}, 剩余 {len(ds_sel.time)} 个月')
                print(f'  计算冬季年平均: 先选冬季月份，再resample到年，再计算统计量')

                # 对选中月份按年求平均（每年JJA的3个月平均 → 100个年值）
                total_heat_annual = -ds_sel['total_heat_surface_exchange_flux_nonadvective_heat'].resample(time='1Y').mean() / 1e9
                total_heat_mean = total_heat_annual.mean(dim='time')
                total_heat_std = total_heat_annual.std(dim='time')

                evap_salt_annual = -ds_sel['surface_ocean_flux_advective_negative_rhs_evaporation_salt'].resample(time='1Y').mean() / 1e9
                snow_salt_annual = -ds_sel['surface_ocean_flux_advective_negative_rhs_snow_salt'].resample(time='1Y').mean() / 1e9
                seaice_salt_annual = -ds_sel['surface_ocean_flux_advective_negative_rhs_sea_ice_melt_salt'].resample(time='1Y').mean() / 1e9
                rain_salt_annual = -ds_sel['surface_ocean_flux_advective_negative_rhs_rain_and_ice_salt'].resample(time='1Y').mean() / 1e9
                river_salt_annual = -ds_sel['surface_ocean_flux_advective_negative_rhs_rivers_salt'].resample(time='1Y').mean() / 1e9

                evap_salt_mean = evap_salt_annual.mean(dim='time')
                snow_salt_mean = snow_salt_annual.mean(dim='time')
                seaice_salt_mean = seaice_salt_annual.mean(dim='time')
                rain_salt_mean = rain_salt_annual.mean(dim='time')
                river_salt_mean = river_salt_annual.mean(dim='time')

                evap_salt_std = evap_salt_annual.std(dim='time')
                snow_salt_std = snow_salt_annual.std(dim='time')
                seaice_salt_std = seaice_salt_annual.std(dim='time')
                rain_salt_std = rain_salt_annual.std(dim='time')
                river_salt_std = river_salt_annual.std(dim='time')
            else:
                # 年平均：先对每年12个月求平均（得到100个年值），再计算这100个年值的均值和标准差
                print(f'  计算年平均: 先resample到年，再计算统计量')
                total_heat_annual = -ds['total_heat_surface_exchange_flux_nonadvective_heat'].resample(time='1Y').mean() / 1e9
                total_heat_mean = total_heat_annual.mean(dim='time')
                total_heat_std = total_heat_annual.std(dim='time')

                evap_salt_annual = -ds['surface_ocean_flux_advective_negative_rhs_evaporation_salt'].resample(time='1Y').mean() / 1e9
                snow_salt_annual = -ds['surface_ocean_flux_advective_negative_rhs_snow_salt'].resample(time='1Y').mean() / 1e9
                seaice_salt_annual = -ds['surface_ocean_flux_advective_negative_rhs_sea_ice_melt_salt'].resample(time='1Y').mean() / 1e9
                rain_salt_annual = -ds['surface_ocean_flux_advective_negative_rhs_rain_and_ice_salt'].resample(time='1Y').mean() / 1e9
                river_salt_annual = -ds['surface_ocean_flux_advective_negative_rhs_rivers_salt'].resample(time='1Y').mean() / 1e9

                evap_salt_mean = evap_salt_annual.mean(dim='time')
                snow_salt_mean = snow_salt_annual.mean(dim='time')
                seaice_salt_mean = seaice_salt_annual.mean(dim='time')
                rain_salt_mean = rain_salt_annual.mean(dim='time')
                river_salt_mean = river_salt_annual.mean(dim='time')

                evap_salt_std = evap_salt_annual.std(dim='time')
                snow_salt_std = snow_salt_annual.std(dim='time')
                seaice_salt_std = seaice_salt_annual.std(dim='time')
                rain_salt_std = rain_salt_annual.std(dim='time')
                river_salt_std = river_salt_annual.std(dim='time')

            sigma2 = ds['sigma2_l_target'].values

            # 计算分组淡水通量的均值和标准差
            seaice_fw_mean = seaice_salt_mean  # 海冰
            seaice_fw_std = seaice_salt_std

            other_fw_mean = evap_salt_mean + snow_salt_mean + rain_salt_mean + river_salt_mean  # 其他淡水
            # 标准差的传播（独立变量）：σ_total = sqrt(σ1² + σ2² + ...)
            other_fw_std = np.sqrt(evap_salt_std**2 + snow_salt_std**2 + rain_salt_std**2 + river_salt_std**2)

            total_freshwater_mean = seaice_fw_mean + other_fw_mean
            total_freshwater_std = np.sqrt(seaice_fw_std**2 + other_fw_std**2)

            total_wmt_mean = total_heat_mean + total_freshwater_mean
            total_wmt_std = np.sqrt(total_heat_std**2 + total_freshwater_std**2)

            # ====================================================================
            # 绘图（先画阴影，再画线）
            # ====================================================================

            # 1. 总WMT（黑色粗实线 + 灰色阴影）
            ax.fill_between(sigma2,
                           (total_wmt_mean - total_wmt_std).values,
                           (total_wmt_mean + total_wmt_std).values,
                           color='#888888', alpha=0.30, linewidth=0, zorder=5)
            l_total, = ax.plot(sigma2, total_wmt_mean.values,
                   color=COL_TOTAL, linestyle='-', linewidth=pt(1.25),
                   label='Total WMT', solid_capstyle='round',
                   zorder=10)

            # 2. 总热通量（红色实线 + 浅红色阴影）
            ax.fill_between(sigma2,
                           (total_heat_mean - total_heat_std).values,
                           (total_heat_mean + total_heat_std).values,
                           color=COL_HEAT, alpha=0.20, linewidth=0, zorder=4)
            l_heat, = ax.plot(sigma2, total_heat_mean.values,
                   color=COL_HEAT, linestyle='-', linewidth=pt(1.1),
                   label='Heat', solid_capstyle='round',
                   zorder=8)

            # 3. 海冰淡水贡献（蓝绿色实线 + 浅蓝绿色阴影）
            ax.fill_between(sigma2,
                           (seaice_fw_mean - seaice_fw_std).values,
                           (seaice_fw_mean + seaice_fw_std).values,
                           color=COL_SEAICE, alpha=0.20, linewidth=0, zorder=3)
            l_si, = ax.plot(sigma2, seaice_fw_mean.values,
                   color=COL_SEAICE, linestyle='-', linewidth=pt(1.0),
                   label='Sea-ice FW', solid_capstyle='round',
                   zorder=6)

            # 4. 其他淡水贡献（蓝色实线 + 浅蓝色阴影）
            ax.fill_between(sigma2,
                           (other_fw_mean - other_fw_std).values,
                           (other_fw_mean + other_fw_std).values,
                           color=COL_OTHERFW, alpha=0.20, linewidth=0, zorder=2)
            l_oth, = ax.plot(sigma2, other_fw_mean.values,
                   color=COL_OTHERFW, linestyle='-', linewidth=pt(1.0),
                   label='Other FW', solid_capstyle='round',
                   zorder=6)

            # Capture legend handles once (top-left panel)
            if legend_handles is None:
                legend_handles = [l_total, l_heat, l_si, l_oth]

            # 打印统计
            print(f'  Total WMT:        {total_wmt_mean.sum().values:8.2f} ± {total_wmt_std.sum().values:6.2f} Sv')
            print(f'  Total Heat:       {total_heat_mean.sum().values:8.2f} ± {total_heat_std.sum().values:6.2f} Sv')
            print(f'  Sea Ice FW:       {seaice_fw_mean.sum().values:8.2f} ± {seaice_fw_std.sum().values:6.2f} Sv')
            print(f'  Other FW:         {other_fw_mean.sum().values:8.2f} ± {other_fw_std.sum().values:6.2f} Sv')

            # ====================================================================
            # 设置坐标轴 (Nature-style: edge labels only)
            # ====================================================================
            # X-axis label & tick labels only on bottom row
            if row_idx == len(REGIONS) - 1:
                ax.set_xlabel(r'$\sigma_2$  (kg m$^{-3}$)', fontsize=F_LAB,
                              fontweight='bold', labelpad=pt(2), color='k')
            else:
                ax.set_xlabel('')
                ax.tick_params(axis='x', labelbottom=False)

            # Y-axis label only on leftmost column
            if col_idx == 0:
                ax.set_ylabel('WMT  (Sv)', fontsize=F_LAB, fontweight='bold',
                              labelpad=pt(2), color='k')
            else:
                ax.set_ylabel('')

            # 第一行显示实验名称
            if row_idx == 0:
                ax.set_title(f'{exp_title}', fontsize=F_TITLE, fontweight='bold',
                             pad=pt(12), color='k')

            # 第一列显示区域名称（左侧 row label, outside the y-axis label）
            if col_idx == 0:
                y_mid = (BOT + (NR - 1 - row_idx) * (PH + GAP_Y) + PH / 2) / HP
                fig.text(13 / WP, y_mid, region_title.replace(' ', '\n'),
                         rotation=90, ha='center', va='center',
                         multialignment='center', linespacing=1.05,
                         fontsize=F_TITLE, fontweight='bold', color='k')

            # 子图标签放在面板左上角外侧（面板内放不下 9 pt 字母而不压曲线）
            panel_idx = row_idx * 5 + col_idx
            ax.annotate(f'({panel_labels[panel_idx]})', xy=(0, 1),
                        xycoords='axes fraction', xytext=(0, pt(2.5)),
                        textcoords='offset points', fontsize=F_LET,
                        fontweight='bold', ha='left', va='baseline', color='k',
                        annotation_clip=False)

            # 网格和零线 (subtle)
            ax.grid(True, alpha=0.25, linestyle='--', linewidth=pt(0.4),
                    color='#999999')
            ax.set_axisbelow(True)
            ax.axhline(y=0, color='#444444', linestyle='-',
                       linewidth=pt(0.5), alpha=0.6, zorder=1)

            # 设置x轴范围
            if exp_key in ['pi', 'mh', 'lig']:
                ax.set_xlim(35.5, 37.2)
                # tick marks every 0.5 as before; labels only at 36.0 and 37.0
                # (four 7.5 pt labels do not fit in a 0.6 in panel)
                ax.set_xticks([35.5, 36.0, 36.5, 37.0])
                ax.set_xticklabels(['', '36.0', '', '37.0'])
            else:  # lgm, mis
                ax.set_xlim(36.0, 38.5)
                ax.set_xticks([36, 37, 38])

            # Tick polish
            ax.tick_params(axis='both', which='major', labelsize=F_TICK,
                           color='k')

            ds.close()


    # ========================================================================
    # 单一共享图例 (top, outside) + 保存
    # ========================================================================
    if month_selector is None:
        filename = 'figures/plot_wmt_4regions_5exps_annual_from_100years.pdf'
    else:
        filename = 'figures/plot_wmt_4regions_5exps_winter_from_100years.pdf'

    # Single, framed legend at the bottom of the figure
    if legend_handles is not None:
        leg = fig.legend(handles=legend_handles,
                   loc='lower center',
                   bbox_to_anchor=((L + WP - R) / 2 / WP, 2 / HP),
                   ncol=4,
                   frameon=True,
                   fancybox=False,
                   edgecolor='k',
                   facecolor='white',
                   framealpha=1.0,
                   fontsize=F_LEG,
                   handlelength=2.2,
                   handletextpad=0.5,
                   columnspacing=1.5,
                   borderpad=0.5,
                   labelcolor='k')
        leg.get_frame().set_linewidth(LW_AX)

    # Save as PDF (vector) and high-res PNG for previews
    # fixed page size (no bbox_inches='tight') so the printed scale is exact
    plt.savefig(filename, dpi=400)
    png_filename = filename.replace('.pdf', '.png')
    plt.savefig(png_filename, dpi=600)
    plt.close()

    print()
    print(f'✓ 保存到: {filename}')
    print()

    return filename


# ============================================================================
# 主程序：生成两张图
# ============================================================================
print('='*80)
print('开始绘图')
print('='*80)

# 1. 全年平均
annual_file = plot_wmt_comparison('Annual Mean', month_selector=None)

# 2. 冬季平均（6-8月）
winter_file = plot_wmt_comparison('Winter Mean', month_selector=WINTER_MONTHS)

print()
print('='*80)
print('全部完成！')
print('='*80)
print(f'✓ 全年平均: {annual_file}')
print(f'✓ 冬季平均: {winter_file}')
print()
