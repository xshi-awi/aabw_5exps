#!/usr/bin/env python
"""
绘制4个南大洋区域的WMT对比图（改进版）
- 所有数据乘以-1
- 蓝色实线：总淡水通量贡献
- 冷色调虚线：海冰贡献、其他淡水贡献
- 红色实线：总热通量
- 暖色调虚线：热通量各分量
- 黑色粗实线：总WMT（热+淡水）
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

print('='*80)
print('绘制4个区域的WMT对比图（改进版）')
print('='*80)
print()

# 定义区域
REGIONS = {
    'Southern_Ocean': {
        'title': 'Southern Ocean',
        'file': 'wmt_echam_heat_fesom_freshwater_southern_ocean_mh.nc'
    },
    'Ross_Sea': {
        'title': 'Ross Sea',
        'file': 'wmt_echam_heat_fesom_freshwater_ross_sea_mh.nc'
    },
    'Weddell_Sea': {
        'title': 'Weddell Sea',
        'file': 'wmt_echam_heat_fesom_freshwater_weddell_sea_mh.nc'
    },
    'Adelie': {
        'title': 'Adélie Land',
        'file': 'wmt_echam_heat_fesom_freshwater_adelie_mh.nc'
    }
}

# ============================================================================
# 创建图形
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes_flat = axes.flatten()

print('加载数据并绘图...')
print('-'*80)

for idx, (region_key, region_info) in enumerate(REGIONS.items()):
    ax = axes_flat[idx]

    print(f'\n处理区域 {idx+1}/4: {region_info["title"]}')

    # 加载数据
    file_path = region_info['file']
    if not Path(file_path).exists():
        print(f'  ⚠ 文件不存在: {file_path}')
        continue

    ds = xr.open_dataset(file_path)
    print(f'  ✓ 已加载: {file_path}')

    sigma2 = ds['sigma2_l_target'].values

    # ========================================================================
    # 提取并反转数据（乘以-1）
    # ========================================================================

    # 热通量
    total_heat = -ds['total_heat_surface_exchange_flux_nonadvective_heat'].mean(dim='time') / 1e9
    sw_heat = -ds['sw_only_surface_exchange_flux_nonadvective_heat'].mean(dim='time') / 1e9
    lw_heat = -ds['lw_only_surface_exchange_flux_nonadvective_heat'].mean(dim='time') / 1e9
    latent_heat = -ds['latent_only_surface_exchange_flux_nonadvective_heat'].mean(dim='time') / 1e9
    sensible_heat = -ds['sensible_only_surface_exchange_flux_nonadvective_heat'].mean(dim='time') / 1e9

    # 盐通量（淡水贡献）
    evap_salt = -ds['surface_ocean_flux_advective_negative_rhs_evaporation_salt'].mean(dim='time') / 1e9
    snow_salt = -ds['surface_ocean_flux_advective_negative_rhs_snow_salt'].mean(dim='time') / 1e9
    seaice_salt = -ds['surface_ocean_flux_advective_negative_rhs_sea_ice_melt_salt'].mean(dim='time') / 1e9
    rain_salt = -ds['surface_ocean_flux_advective_negative_rhs_rain_and_ice_salt'].mean(dim='time') / 1e9
    river_salt = -ds['surface_ocean_flux_advective_negative_rhs_rivers_salt'].mean(dim='time') / 1e9

    # 计算组合
    total_freshwater = evap_salt + snow_salt + seaice_salt + rain_salt + river_salt
    other_freshwater = evap_salt + snow_salt + rain_salt + river_salt  # 除了海冰的其他淡水
    total_wmt = total_heat + total_freshwater

    # ========================================================================
    # 绘图
    # ========================================================================

    # 1. 总WMT（黑色粗实线）
    ax.plot(sigma2, total_wmt.values,
           color='k', linestyle='-', linewidth=3.5,
           label=f'Total WMT ({total_wmt.sum().values:+.1f} Sv)',
           zorder=10)

    # 2. 总热通量（红色实线）
    ax.plot(sigma2, total_heat.values,
           color='#e74c3c', linestyle='-', linewidth=2.5,
           label=f'Total Heat ({total_heat.sum().values:+.1f} Sv)',
           zorder=8)

    # 3. 总淡水通量（蓝色实线）
    ax.plot(sigma2, total_freshwater.values,
           color='#2980b9', linestyle='-', linewidth=2.5,
           label=f'Total Freshwater ({total_freshwater.sum().values:+.1f} Sv)',
           zorder=7)

    # 4. 热通量分量（暖色调虚线）
    ax.plot(sigma2, sw_heat.values,
           color='#f39c12', linestyle='--', linewidth=1.8,
           label=f'SW ({sw_heat.sum().values:+.1f} Sv)', alpha=0.8)

    ax.plot(sigma2, lw_heat.values,
           color='#e67e22', linestyle='--', linewidth=1.8,
           label=f'LW ({lw_heat.sum().values:+.1f} Sv)', alpha=0.8)

    ax.plot(sigma2, latent_heat.values,
           color='#d35400', linestyle='--', linewidth=1.8,
           label=f'Latent ({latent_heat.sum().values:+.1f} Sv)', alpha=0.8)

    ax.plot(sigma2, sensible_heat.values,
           color='#c0392b', linestyle='--', linewidth=1.8,
           label=f'Sensible ({sensible_heat.sum().values:+.1f} Sv)', alpha=0.8)

    # 5. 淡水分量（冷色调虚线）
    ax.plot(sigma2, seaice_salt.values,
           color='#16a085', linestyle='--', linewidth=2,
           label=f'Sea Ice ({seaice_salt.sum().values:+.1f} Sv)', alpha=0.8)

    ax.plot(sigma2, other_freshwater.values,
           color='#3498db', linestyle='--', linewidth=2,
           label=f'Other FW ({other_freshwater.sum().values:+.1f} Sv)', alpha=0.8)

    # 打印统计
    print(f'  Total WMT:        {total_wmt.sum().values:8.2f} Sv')
    print(f'  Total Heat:       {total_heat.sum().values:8.2f} Sv')
    print(f'  Total Freshwater: {total_freshwater.sum().values:8.2f} Sv')

    # ========================================================================
    # 设置坐标轴
    # ========================================================================
    ax.set_xlabel(r'Potential Density $\sigma_2$ [kg/m$^3$ - 1000]',
                  fontsize=11, fontweight='bold')
    ax.set_ylabel(r'WMT Rate [Sv]', fontsize=11, fontweight='bold')
    ax.set_title(f'{region_info["title"]}',
                fontsize=13, fontweight='bold')

    # 网格和零线
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)

    # 设置x轴范围
    ax.set_xlim(26, 38)

    # 图例（两列）
    ax.legend(loc='best', fontsize=8.5, framealpha=0.9, ncol=2)

    ds.close()

# ============================================================================
# 调整布局并保存
# ============================================================================
plt.suptitle('Water Mass Transformation in Southern Ocean Regions\n'
             'ECHAM Heat Flux + FESOM Freshwater (Annual Mean, Sign Reversed)',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])

# 保存图像
output_pdf = 'wmt_4regions_comparison.pdf'

plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
plt.close()

print()
print('='*80)
print('完成！')
print('='*80)
print(f'✓ 保存到: {output_pdf}')
print()

print('图例说明：')
print('  黑色粗实线：Total WMT (Heat + Freshwater)')
print('  红色实线：  Total Heat')
print('  蓝色实线：  Total Freshwater (Evap+Snow+Rain+River+SeaIce)')
print('  暖色虚线：  Heat components (SW, LW, Latent, Sensible)')
print('  冷色虚线：  Freshwater components (Sea Ice, Other FW)')
print()
