#!/usr/bin/env python
"""
绘制4个南大洋区域x5个实验的WMT对比图
- 生成两张图：全年平均 + 冬季平均（6-9月，AABW形成期）
- 所有数据乘以-1
- 黑色粗实线：总WMT（热+淡水）
- 红色实线：总热通量
- 暖色调虚线：热通量分量 (lw, sw, lh, sh)
- 蓝绿色虚线：海冰淡水贡献
- 蓝色虚线：其他淡水贡献（蒸发+降水+径流+雪）
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

print('='*80)
print('绘制4个区域 x 5个实验的WMT对比图（全年 + 冬季）')
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
BASE_PATH = Path('/home/a/a270064/bb1029/aabw_5exps')

# 冬季月份（南半球冬季，AABW形成主要时期：6-9月）
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

    fig, axes = plt.subplots(4, 5, figsize=(18, 14))

    print(f'\n{"="*80}')
    print(f'绘制 {season_name} 数据')
    print(f'{"="*80}')

    for row_idx, (region_key, region_title) in enumerate(REGIONS):
        for col_idx, (exp_key, exp_title) in enumerate(EXPERIMENTS):
            ax = axes[row_idx, col_idx]

            print(f'\n处理: {region_title} - {exp_title}')

            # 构建文件路径
            file_path = BASE_PATH / exp_key / f'wmt_echam_heat_fesom_freshwater_{region_key}_{exp_key}.nc'

            if not file_path.exists():
                print(f'  ⚠ 文件不存在: {file_path}')
                ax.text(0.5, 0.5, 'Data Not Available',
                       ha='center', va='center', fontsize=12, transform=ax.transAxes)
                ax.set_xlabel(r'$\sigma_2$ [kg/m$^3$]', fontsize=10)
                ax.set_ylabel('WMT [Sv]', fontsize=10)
                if row_idx == 0:
                    ax.set_title(f'{exp_title}', fontsize=12, fontweight='bold')
                if col_idx == 0:
                    ax.text(-0.15, 0.5, region_title, rotation=90, va='center',
                           fontsize=13, fontweight='bold', transform=ax.transAxes)
                ax.grid(True, alpha=0.3)
                continue

            # 加载数据
            ds = xr.open_dataset(file_path)
            print(f'  ✓ 已加载: {file_path.name}')

            # 选择月份
            if month_selector is not None:
                # 选择特定月份
                ds_sel = ds.isel(time=ds.time.dt.month.isin(month_selector))
            else:
                ds_sel = ds

            sigma2 = ds['sigma2_l_target'].values

            # 提取并反转数据（乘以-1）
            # 热通量总量
            total_heat = -ds_sel['total_heat_surface_exchange_flux_nonadvective_heat'].mean(dim='time') / 1e9

            # 热通量分量
            lw_heat = -ds_sel['lw_only_surface_exchange_flux_nonadvective_heat'].mean(dim='time') / 1e9
            sw_heat = -ds_sel['sw_only_surface_exchange_flux_nonadvective_heat'].mean(dim='time') / 1e9
            latent_heat = -ds_sel['latent_only_surface_exchange_flux_nonadvective_heat'].mean(dim='time') / 1e9
            sensible_heat = -ds_sel['sensible_only_surface_exchange_flux_nonadvective_heat'].mean(dim='time') / 1e9

            # 盐通量（淡水贡献）
            evap_salt = -ds_sel['surface_ocean_flux_advective_negative_rhs_evaporation_salt'].mean(dim='time') / 1e9
            snow_salt = -ds_sel['surface_ocean_flux_advective_negative_rhs_snow_salt'].mean(dim='time') / 1e9
            seaice_salt = -ds_sel['surface_ocean_flux_advective_negative_rhs_sea_ice_melt_salt'].mean(dim='time') / 1e9
            rain_salt = -ds_sel['surface_ocean_flux_advective_negative_rhs_rain_and_ice_salt'].mean(dim='time') / 1e9
            river_salt = -ds_sel['surface_ocean_flux_advective_negative_rhs_rivers_salt'].mean(dim='time') / 1e9

            # 计算分组淡水通量
            seaice_fw = seaice_salt  # 海冰
            other_fw = evap_salt + snow_salt + rain_salt + river_salt  # 其他淡水
            total_freshwater = seaice_fw + other_fw
            total_wmt = total_heat + total_freshwater

            # ====================================================================
            # 绘图
            # ====================================================================

            # 1. 总WMT（黑色粗实线）
            ax.plot(sigma2, total_wmt.values,
                   color='k', linestyle='-', linewidth=2.5,
                   label='Total WMT',
                   zorder=10)

            # 2. 总热通量（红色实线）
            ax.plot(sigma2, total_heat.values,
                   color='#e74c3c', linestyle='-', linewidth=2,
                   label='Heat',
                   zorder=8)

            # 3. 热通量分量（暖色调虚线）
            ax.plot(sigma2, lw_heat.values,
                   color='#ff7f50', linestyle='--', linewidth=1.3,
                   label='LW',
                   alpha=0.85, zorder=7)

            ax.plot(sigma2, sw_heat.values,
                   color='#ffa500', linestyle='--', linewidth=1.3,
                   label='SW',
                   alpha=0.85, zorder=7)

            ax.plot(sigma2, latent_heat.values,
                   color='#ff6347', linestyle='--', linewidth=1.3,
                   label='LH',
                   alpha=0.85, zorder=7)

            ax.plot(sigma2, sensible_heat.values,
                   color='#ff4500', linestyle='--', linewidth=1.3,
                   label='SH',
                   alpha=0.85, zorder=7)

            # 4. 海冰淡水贡献（蓝绿色虚线）
            ax.plot(sigma2, seaice_fw.values,
                   color='#16a085', linestyle='--', linewidth=1.5,
                   label='Sea Ice FW',
                   alpha=0.9, zorder=6)

            # 5. 其他淡水贡献（蓝色虚线）
            ax.plot(sigma2, other_fw.values,
                   color='#2980b9', linestyle='--', linewidth=1.5,
                   label='Other FW',
                   alpha=0.9, zorder=6)

            # 打印统计
            print(f'  Total WMT:        {total_wmt.sum().values:8.2f} Sv')
            print(f'  Total Heat:       {total_heat.sum().values:8.2f} Sv')
            print(f'  Sea Ice FW:       {seaice_fw.sum().values:8.2f} Sv')
            print(f'  Other FW:         {other_fw.sum().values:8.2f} Sv')

            # ====================================================================
            # 设置坐标轴
            # ====================================================================
            ax.set_xlabel(r'$\sigma_2$ [kg/m$^3$]', fontsize=10)
            ax.set_ylabel('WMT [Sv]', fontsize=10)

            # 第一行显示实验名称
            if row_idx == 0:
                ax.set_title(f'{exp_title}', fontsize=12, fontweight='bold')

            # 第一列显示区域名称（左侧）
            if col_idx == 0:
                ax.text(-0.15, 0.5, region_title, rotation=90, va='center',
                       fontsize=13, fontweight='bold', transform=ax.transAxes)

            # 网格和零线
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)

            # 设置x轴范围
            ax.set_xlim(34, 39)

            # 图例（紧凑布局，两列）
            ax.legend(loc='best', fontsize=6, framealpha=0.9, ncol=2)

            ds.close()


    # ========================================================================
    # 调整布局并保存
    # ========================================================================
    if month_selector is None:
        title_season = 'Annual Mean'
        filename = 'plot_wmt_4regions_5exps_annual.pdf'
    else:
        title_season = f'Winter Mean (Jun-Sep, AABW Formation Period)'
        filename = 'plot_wmt_4regions_5exps_winter.pdf'

    plt.suptitle(f'Water Mass Transformation: 4 Regions × 5 Experiments\n'
                 f'ECHAM Heat Flux + FESOM Freshwater ({title_season}, Sign Reversed)',
                 fontsize=18, fontweight='bold', y=0.998)

    plt.tight_layout(rect=[0, 0, 1, 0.995])

    plt.savefig(filename, dpi=300, bbox_inches='tight')
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

# 2. 冬季平均（6-9月）
winter_file = plot_wmt_comparison('Winter Mean', month_selector=WINTER_MONTHS)

print()
print('='*80)
print('全部完成！')
print('='*80)
print(f'✓ 全年平均: {annual_file}')
print(f'✓ 冬季平均: {winter_file}')
print()
print('图例说明：')
print('  黑色粗实线：Total WMT (Heat + Freshwater)')
print('  红色实线：  Total Heat')
print('  暖色虚线：  Heat Components (LW, SW, LH, SH)')
print('  蓝绿色虚线：Sea Ice Freshwater')
print('  蓝色虚线：  Other Freshwater (Evap+Prec+Runoff+Snow)')
print()
print('注：冬季月份为6-9月（南半球冬季，AABW主要形成期）')
print()
