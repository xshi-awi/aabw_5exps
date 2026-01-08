#!/usr/bin/env python
"""
绘制穿过 σ₂=36.62 密度层的WMT各分量的季节变化
4个区域，每个区域一个子图
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

print('='*80)
print('绘制 σ₂=36.62 密度层的季节变化')
print('='*80)
print()

# 定义区域
REGIONS = {
    'Southern_Ocean': {
        'title': 'Southern Ocean',
        'file': 'wmt_echam_heat_fesom_freshwater_southern_ocean_pi.nc'
    },
    'Ross_Sea': {
        'title': 'Ross Sea',
        'file': 'wmt_echam_heat_fesom_freshwater_ross_sea_pi.nc'
    },
    'Weddell_Sea': {
        'title': 'Weddell Sea',
        'file': 'wmt_echam_heat_fesom_freshwater_weddell_sea_pi.nc'
    },
    'Adelie': {
        'title': 'Adélie Land',
        'file': 'wmt_echam_heat_fesom_freshwater_adelie_pi.nc'
    }
}

# 目标密度
TARGET_SIGMA2 = 36.62

# ============================================================================
# 创建图形
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes_flat = axes.flatten()

print('加载数据并绘图...')
print('-'*80)

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
x_months = np.arange(1, 13)

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

    # 找到最接近目标密度的索引
    sigma2 = ds['sigma2_l_target'].values
    sigma2_idx = np.argmin(np.abs(sigma2 - TARGET_SIGMA2))
    actual_sigma2 = sigma2[sigma2_idx]
    print(f'  目标密度: {TARGET_SIGMA2}, 实际: {actual_sigma2:.2f}')

    # ========================================================================
    # 提取时间序列（在目标密度层）并反转符号
    # ========================================================================

    # 热通量
    total_heat = -ds['total_heat_surface_exchange_flux_nonadvective_heat'].isel(sigma2_l_target=sigma2_idx) / 1e9
    sw_heat = -ds['sw_only_surface_exchange_flux_nonadvective_heat'].isel(sigma2_l_target=sigma2_idx) / 1e9
    lw_heat = -ds['lw_only_surface_exchange_flux_nonadvective_heat'].isel(sigma2_l_target=sigma2_idx) / 1e9
    latent_heat = -ds['latent_only_surface_exchange_flux_nonadvective_heat'].isel(sigma2_l_target=sigma2_idx) / 1e9
    sensible_heat = -ds['sensible_only_surface_exchange_flux_nonadvective_heat'].isel(sigma2_l_target=sigma2_idx) / 1e9

    # 盐通量（淡水贡献）
    evap_salt = -ds['surface_ocean_flux_advective_negative_rhs_evaporation_salt'].isel(sigma2_l_target=sigma2_idx) / 1e9
    snow_salt = -ds['surface_ocean_flux_advective_negative_rhs_snow_salt'].isel(sigma2_l_target=sigma2_idx) / 1e9
    seaice_salt = -ds['surface_ocean_flux_advective_negative_rhs_sea_ice_melt_salt'].isel(sigma2_l_target=sigma2_idx) / 1e9
    rain_salt = -ds['surface_ocean_flux_advective_negative_rhs_rain_and_ice_salt'].isel(sigma2_l_target=sigma2_idx) / 1e9
    river_salt = -ds['surface_ocean_flux_advective_negative_rhs_rivers_salt'].isel(sigma2_l_target=sigma2_idx) / 1e9

    # 计算组合
    total_freshwater = evap_salt + snow_salt + seaice_salt + rain_salt + river_salt
    other_freshwater = evap_salt + snow_salt + rain_salt + river_salt
    total_wmt = total_heat + total_freshwater

    # ========================================================================
    # 绘图
    # ========================================================================

    # 1. 总WMT（黑色粗实线）
    ax.plot(x_months, total_wmt.values,
           color='k', linestyle='-', linewidth=3,
           label='Total WMT', marker='o', markersize=6, zorder=10)

    # 2. 总热通量（红色实线）
    ax.plot(x_months, total_heat.values,
           color='#e74c3c', linestyle='-', linewidth=2.5,
           label='Total Heat', marker='s', markersize=5, zorder=8)

    # 3. 总淡水通量（蓝色实线）
    ax.plot(x_months, total_freshwater.values,
           color='#2980b9', linestyle='-', linewidth=2.5,
           label='Total Freshwater', marker='^', markersize=5, zorder=7)

    # 4. 热通量分量（暖色调虚线）
    ax.plot(x_months, sw_heat.values,
           color='#f39c12', linestyle='--', linewidth=1.5,
           label='SW', marker='v', markersize=4, alpha=0.8)

    ax.plot(x_months, lw_heat.values,
           color='#e67e22', linestyle='--', linewidth=1.5,
           label='LW', marker='<', markersize=4, alpha=0.8)

    ax.plot(x_months, latent_heat.values,
           color='#d35400', linestyle='--', linewidth=1.5,
           label='Latent', marker='>', markersize=4, alpha=0.8)

    ax.plot(x_months, sensible_heat.values,
           color='#c0392b', linestyle='--', linewidth=1.5,
           label='Sensible', marker='d', markersize=4, alpha=0.8)

    # 5. 淡水分量（冷色调虚线）
    ax.plot(x_months, seaice_salt.values,
           color='#16a085', linestyle='--', linewidth=1.8,
           label='Sea Ice', marker='p', markersize=4, alpha=0.8)

    ax.plot(x_months, other_freshwater.values,
           color='#3498db', linestyle='--', linewidth=1.8,
           label='Other FW', marker='*', markersize=5, alpha=0.8)

    # 打印统计
    print(f'  Annual mean Total WMT: {total_wmt.mean().values:8.2f} Sv')
    print(f'  Annual mean Heat:      {total_heat.mean().values:8.2f} Sv')
    print(f'  Annual mean Freshwater:{total_freshwater.mean().values:8.2f} Sv')
    print(f'  Range Total WMT:       [{total_wmt.min().values:6.2f}, {total_wmt.max().values:6.2f}] Sv')

    # ========================================================================
    # 设置坐标轴
    # ========================================================================
    ax.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax.set_ylabel(r'WMT Rate [Sv]', fontsize=11, fontweight='bold')
    ax.set_title(f'{region_info["title"]}\n' + r'$\sigma_2$ = ' + f'{actual_sigma2:.2f}',
                fontsize=13, fontweight='bold')

    # 网格和零线
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)

    # 设置x轴
    ax.set_xticks(x_months)
    ax.set_xticklabels(months, rotation=0, ha='center')
    ax.set_xlim(0.5, 12.5)

    # 图例
    ax.legend(loc='best', fontsize=8, framealpha=0.9, ncol=2)

    ds.close()

# ============================================================================
# 调整布局并保存
# ============================================================================
plt.suptitle(r'Seasonal Cycle of WMT at $\sigma_2$ = ' + f'{TARGET_SIGMA2}\n'
             'ECHAM Heat Flux + FESOM Freshwater (Sign Reversed)',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])

# 保存图像
output_pdf = f'wmt_sigma{TARGET_SIGMA2}_seasonal_4regions.pdf'
output_png = f'wmt_sigma{TARGET_SIGMA2}_seasonal_4regions.png'

plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.close()

print()
print('='*80)
print('完成！')
print('='*80)
print(f'✓ 保存到: {output_pdf}')
print(f'✓ 保存到: {output_png}')
print()

print('图例说明：')
print('  黑色粗实线：Total WMT (Heat + Freshwater)')
print('  红色实线：  Total Heat')
print('  蓝色实线：  Total Freshwater')
print('  暖色虚线：  Heat components (SW, LW, Latent, Sensible)')
print('  冷色虚线：  Freshwater components (Sea Ice, Other FW)')
print()
