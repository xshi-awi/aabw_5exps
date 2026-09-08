#!/usr/bin/env python
"""
从100年月数据计算WMT多年平均并绘制对比图
- 读取wmt_results/wmt_*_100years_*.nc (1200个月)
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
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 16,
    'axes.linewidth': 1.2,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#222222',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

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

EXPERIMENTS = [
    ('pi', 'PI'),
    ('mh', 'MH'),
    ('lig', 'LIG'),
    ('lgm', 'LGM'),
    ('mis', 'MIS')
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

    fig, axes = plt.subplots(4, 5, figsize=(20, 15.5),
                              sharey=False)
    # Generous spacing for a clean, breathable journal layout
    fig.subplots_adjust(left=0.10, right=0.985, top=0.955, bottom=0.13,
                        hspace=0.32, wspace=0.28)

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
            file_path = BASE_PATH / exp_key / 'wmt_results' / f'wmt_{region_key}_100years_{exp_key}.nc'

            if not file_path.exists():
                print(f'  ⚠ 文件不存在: {file_path}')
                ax.text(0.5, 0.5, 'Data Not Available',
                       ha='center', va='center', fontsize=14,
                       color='#888888', transform=ax.transAxes)
                if row_idx == 0:
                    ax.set_title(f'{exp_title}', fontsize=22, fontweight='bold',
                                 pad=12, color='#111111')
                ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)
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
                   color=COL_TOTAL, linestyle='-', linewidth=3.0,
                   label='Total WMT', solid_capstyle='round',
                   zorder=10)

            # 2. 总热通量（红色实线 + 浅红色阴影）
            ax.fill_between(sigma2,
                           (total_heat_mean - total_heat_std).values,
                           (total_heat_mean + total_heat_std).values,
                           color=COL_HEAT, alpha=0.20, linewidth=0, zorder=4)
            l_heat, = ax.plot(sigma2, total_heat_mean.values,
                   color=COL_HEAT, linestyle='-', linewidth=2.6,
                   label='Heat', solid_capstyle='round',
                   zorder=8)

            # 3. 海冰淡水贡献（蓝绿色实线 + 浅蓝绿色阴影）
            ax.fill_between(sigma2,
                           (seaice_fw_mean - seaice_fw_std).values,
                           (seaice_fw_mean + seaice_fw_std).values,
                           color=COL_SEAICE, alpha=0.20, linewidth=0, zorder=3)
            l_si, = ax.plot(sigma2, seaice_fw_mean.values,
                   color=COL_SEAICE, linestyle='-', linewidth=2.4,
                   label='Sea-ice FW', solid_capstyle='round',
                   zorder=6)

            # 4. 其他淡水贡献（蓝色实线 + 浅蓝色阴影）
            ax.fill_between(sigma2,
                           (other_fw_mean - other_fw_std).values,
                           (other_fw_mean + other_fw_std).values,
                           color=COL_OTHERFW, alpha=0.20, linewidth=0, zorder=2)
            l_oth, = ax.plot(sigma2, other_fw_mean.values,
                   color=COL_OTHERFW, linestyle='-', linewidth=2.4,
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
                ax.set_xlabel(r'$\sigma_2$  (kg m$^{-3}$)', fontsize=18,
                              labelpad=8, color='#111111')
            else:
                ax.set_xlabel('')
                ax.tick_params(axis='x', labelbottom=False)

            # Y-axis label only on leftmost column
            if col_idx == 0:
                ax.set_ylabel('WMT  (Sv)', fontsize=17,
                              labelpad=6, color='#111111')
            else:
                ax.set_ylabel('')

            # 第一行显示实验名称
            if row_idx == 0:
                ax.set_title(f'{exp_title}', fontsize=22, fontweight='bold',
                             pad=12, color='#111111')

            # 第一列显示区域名称（左侧 row label, outside the y-axis label）
            if col_idx == 0:
                ax.annotate(region_title,
                            xy=(0, 0.5), xycoords='axes fraction',
                            xytext=(-95, 0), textcoords='offset points',
                            rotation=90, ha='center', va='center',
                            fontsize=22, fontweight='bold', color='#111111',
                            annotation_clip=False)

            # 子图标签放左上角内部 (Nature style, parenthesized)
            panel_idx = row_idx * 5 + col_idx
            ax.text(0.04, 0.94, f'({panel_labels[panel_idx]})',
                   transform=ax.transAxes, fontsize=18, fontweight='bold',
                   ha='left', va='top', color='#111111')

            # 网格和零线 (subtle)
            ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6,
                    color='#999999')
            ax.set_axisbelow(True)
            ax.axhline(y=0, color='#444444', linestyle='-',
                       linewidth=0.8, alpha=0.6, zorder=1)

            # 设置x轴范围
            ax.set_xlim(35.5, 38.5)

            # common y-axis across ALL panels (supplementary variant)
            ax.set_ylim(-45, 90)

            # Tick polish
            ax.tick_params(axis='both', which='major', labelsize=15,
                           pad=4, color='#333333')

            ds.close()


    # ========================================================================
    # 单一共享图例 (top, outside) + 保存
    # ========================================================================
    if month_selector is None:
        filename = 'figures/figS_wmt_common_axis_annual.pdf'
    else:
        filename = 'figures/figS_wmt_common_axis_winter.pdf'

    # Single, framed legend at the bottom of the figure
    if legend_handles is not None:
        leg = fig.legend(handles=legend_handles,
                   loc='lower center',
                   bbox_to_anchor=(0.5, 0.025),
                   ncol=4,
                   frameon=True,
                   fancybox=False,
                   edgecolor='#333333',
                   facecolor='white',
                   framealpha=1.0,
                   fontsize=18,
                   handlelength=2.6,
                   handletextpad=0.8,
                   columnspacing=2.5,
                   borderpad=0.8,
                   labelcolor='#111111')
        leg.get_frame().set_linewidth(1.0)

    # Save as PDF (vector) and high-res PNG for previews
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    png_filename = filename.replace('.pdf', '.png')
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
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
