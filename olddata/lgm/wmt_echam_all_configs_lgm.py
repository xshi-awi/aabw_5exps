#!/usr/bin/env python
"""
使用 ECHAM 热通量数据计算 Southern Ocean 的 WMT
计算7个不同的热通量配置，并保存到同一个文件中
"""

import warnings
import numpy as np
from dask.diagnostics import ProgressBar
import xarray as xr
import xgcm
import xbudget
from pathlib import Path
from scipy.interpolate import griddata

# 导入 xwmt
import xwmt

print('='*80)
print('ECHAM 热通量 → FESOM → xWMT (所有7个配置)')
print('='*80)
print()

# ============================================================================
# 1. 加载 ECHAM 数据
# ============================================================================
print('步骤 1: 加载 ECHAM T63 气候态数据...')
print('-'*80)

echam_file = Path('lgm_echam_clim.nc')
echam_ds = xr.open_dataset(echam_file)

echam_lon = echam_ds['lon'].values
echam_lat = echam_ds['lat'].values
echam_time = echam_ds['time'].values

# 提取4个热通量分量 (time, lat, lon)
var92 = echam_ds['var92'].values  # LW flux
var95 = echam_ds['var95'].values  # SW flux
var111 = echam_ds['var111'].values  # latent heat
var120 = echam_ds['var120'].values  # sensible heat

print(f'  ECHAM 网格: lon({len(echam_lon)}) × lat({len(echam_lat)})')
print(f'  时间步数: {len(echam_time)}')
print()

# ============================================================================
# 2. 加载 FESOM 网格信息
# ============================================================================
print('步骤 2: 加载 FESOM 三角网格信息...')
print('-'*80)

DATA_DIR = Path(".")

mesh = xr.open_dataset("/home/a/a270064/bb1029/inputs/mesh_glac1d/fesom.mesh.diag.nc")
areacello_1d = mesh['nod_area'].isel(nz=0).values
fesom_lon_1d = np.degrees(mesh['nodes'].isel(n2=0).values)
fesom_lat_1d = np.degrees(mesh['nodes'].isel(n2=1).values)

nnodes = len(fesom_lon_1d)
print(f'  FESOM 节点数: {nnodes}')
print()

# 加载表面示踪剂
sst_data = xr.open_dataset(DATA_DIR / "sst_clim.nc")
sss_data = xr.open_dataset(DATA_DIR / "sss_clim.nc")
time_coord = sst_data['time']
ntime = len(time_coord)
print(f'  时间步数: {ntime}')
print()

# ============================================================================
# 3. 插值 ECHAM → FESOM
# ============================================================================
print('步骤 3: 插值 ECHAM → FESOM...')
print('-'*80)

echam_lon_2d, echam_lat_2d = np.meshgrid(echam_lon, echam_lat)

var92_fesom = np.zeros((ntime, nnodes), dtype=np.float32)
var95_fesom = np.zeros((ntime, nnodes), dtype=np.float32)
var111_fesom = np.zeros((ntime, nnodes), dtype=np.float32)
var120_fesom = np.zeros((ntime, nnodes), dtype=np.float32)

print('开始逐月插值...')
for t in range(ntime):
    if (t + 1) % 3 == 0:
        print(f'  时间步 {t+1}/{ntime}...')

    var92_flat = var92[t, :, :].flatten()
    var95_flat = var95[t, :, :].flatten()
    var111_flat = var111[t, :, :].flatten()
    var120_flat = var120[t, :, :].flatten()

    points = np.column_stack([echam_lon_2d.flatten(), echam_lat_2d.flatten()])
    xi = np.column_stack([fesom_lon_1d, fesom_lat_1d])

    var92_fesom[t, :] = griddata(points, var92_flat, xi, method='nearest')
    var95_fesom[t, :] = griddata(points, var95_flat, xi, method='nearest')
    var111_fesom[t, :] = griddata(points, var111_flat, xi, method='nearest')
    var120_fesom[t, :] = griddata(points, var120_flat, xi, method='nearest')

print('✓ 插值完成')
print()

# ============================================================================
# 4. 准备 Southern Ocean 区域mask
# ============================================================================
print('步骤 4: 准备 Southern Ocean 区域...')
print('-'*80)

lat_min, lat_max = -90, -60
lon_min, lon_max = -180, 180

lat_mask = (fesom_lat_1d >= lat_min) & (fesom_lat_1d <= lat_max)
lon_mask = (fesom_lon_1d >= lon_min) & (fesom_lon_1d <= lon_max)
region_mask = lat_mask & lon_mask

print(f'区域内节点数: {region_mask.sum()} ({region_mask.sum()/nnodes*100:.1f}%)')
print()

# ============================================================================
# 5. 定义7个热通量配置
# ============================================================================
wmt_configs = {
    'total_heat': {
        'description': '总热通量效应 (SW + LW + 潜热 + 感热)',
        'rsntds': var95_fesom,
        'rlntds': var92_fesom,
        'hflso': var111_fesom,
        'hfsso': var120_fesom
    },
    'net_radiation': {
        'description': '净辐射效应 (SW + LW)',
        'rsntds': var95_fesom,
        'rlntds': var92_fesom,
        'hflso': np.zeros_like(var111_fesom),
        'hfsso': np.zeros_like(var120_fesom)
    },
    'turbulent': {
        'description': '湍流通量效应 (潜热 + 感热)',
        'rsntds': np.zeros_like(var95_fesom),
        'rlntds': np.zeros_like(var92_fesom),
        'hflso': var111_fesom,
        'hfsso': var120_fesom
    },
    'sw_only': {
        'description': '短波辐射效应 (var95)',
        'rsntds': var95_fesom,
        'rlntds': np.zeros_like(var92_fesom),
        'hflso': np.zeros_like(var111_fesom),
        'hfsso': np.zeros_like(var120_fesom)
    },
    'lw_only': {
        'description': '长波辐射效应 (var92)',
        'rsntds': np.zeros_like(var95_fesom),
        'rlntds': var92_fesom,
        'hflso': np.zeros_like(var111_fesom),
        'hfsso': np.zeros_like(var120_fesom)
    },
    'latent_only': {
        'description': '潜热效应 (var111)',
        'rsntds': np.zeros_like(var95_fesom),
        'rlntds': np.zeros_like(var92_fesom),
        'hflso': var111_fesom,
        'hfsso': np.zeros_like(var120_fesom)
    },
    'sensible_only': {
        'description': '感热效应 (var120)',
        'rsntds': np.zeros_like(var95_fesom),
        'rlntds': np.zeros_like(var92_fesom),
        'hflso': np.zeros_like(var111_fesom),
        'hfsso': var120_fesom
    }
}

# ============================================================================
# 6. 循环计算每个配置的WMT
# ============================================================================
print('步骤 5: 计算7个配置的WMT...')
print('='*80)

# 用于存储所有配置的结果
all_results = {}

for config_name, config in wmt_configs.items():
    print(f'\n配置: {config_name}')
    print(f'描述: {config["description"]}')
    print('-'*80)

    # ====================================================================
    # 准备数据集
    # ====================================================================
    ds = xr.Dataset()

    # 坐标
    ds.coords['time'] = time_coord
    ds.coords['xh'] = np.arange(nnodes)
    ds.coords['xq'] = np.arange(nnodes + 1)
    ds.coords['yh'] = [0]
    ds.coords['yq'] = [0, 1]

    # 示踪剂
    tos_2d = sst_data['sst'].values[:, np.newaxis, :]
    ds['tos'] = (['time', 'yh', 'xh'], tos_2d)
    ds['tos'].attrs['units'] = 'degC'

    sos_2d = sss_data['sss'].values[:, np.newaxis, :]
    ds['sos'] = (['time', 'yh', 'xh'], sos_2d)
    ds['sos'].attrs['units'] = 'psu'

    # 网格面积 - 应用区域mask
    areacello_masked = np.where(region_mask, areacello_1d, 0.0)
    areacello_2d = areacello_masked[np.newaxis, :]
    ds['areacello'] = (['yh', 'xh'], areacello_2d)
    ds['areacello'].attrs['units'] = 'm2'

    # 经纬度
    ds['lon'] = (['yh', 'xh'], fesom_lon_1d[np.newaxis, :])
    ds['lon'].attrs['units'] = 'degrees_east'
    ds['lat'] = (['yh', 'xh'], fesom_lat_1d[np.newaxis, :])
    ds['lat'].attrs['units'] = 'degrees_north'

    # 热通量 (取负号: ECHAM约定 → xwmt约定)
    ds['rsntds'] = (['time', 'yh', 'xh'], -config['rsntds'][:, np.newaxis, :])
    ds['rsntds'].attrs['units'] = 'W m-2'

    ds['rlntds'] = (['time', 'yh', 'xh'], -config['rlntds'][:, np.newaxis, :])
    ds['rlntds'].attrs['units'] = 'W m-2'

    ds['hflso'] = (['time', 'yh', 'xh'], -config['hflso'][:, np.newaxis, :])
    ds['hflso'].attrs['units'] = 'W m-2'

    ds['hfsso'] = (['time', 'yh', 'xh'], -config['hfsso'][:, np.newaxis, :])
    ds['hfsso'].attrs['units'] = 'W m-2'

    # 淡水通量全部设为 0
    for var in ['prlq', 'prsn', 'evs', 'friver', 'fsitherm']:
        ds[var] = (['time', 'yh', 'xh'], np.zeros((ntime, 1, nnodes), dtype=np.float32))
        ds[var].attrs['units'] = 'kg m-2 s-1'

    # ====================================================================
    # 创建 Grid
    # ====================================================================
    coords = {
        'X': {'center': 'xh', 'outer': 'xq'},
        'Y': {'center': 'yh', 'outer': 'yq'}
    }
    boundary = {'X': 'extend', 'Y': 'extend'}
    metrics = {('X', 'Y'): 'areacello'}

    grid = xgcm.Grid(ds, coords=coords, metrics=metrics, boundary=boundary,
                     autoparse_metadata=False)

    # ====================================================================
    # 使用 xbudget 构建预算
    # ====================================================================
    xbudget_dict = xbudget.load_preset_budget(model="MOM6_surface")
    xbudget.collect_budgets(ds, xbudget_dict)

    decompose_list = [
        "surface_exchange_flux",
        "advective",
        "surface_ocean_flux_advective_negative_rhs"
    ]
    decomposed_budgets = xbudget.aggregate(xbudget_dict, decompose=decompose_list)

    # 移除 3D lambda 坐标
    for budget_name in ['heat', 'salt', 'mass']:
        if budget_name in decomposed_budgets and 'lambda' in decomposed_budgets[budget_name]:
            del decomposed_budgets[budget_name]['lambda']

    # ====================================================================
    # 计算 WMT
    # ====================================================================
    print('计算 WMT...')

    wmt_obj = xwmt.WaterMassTransformations(grid, decomposed_budgets)

    sigma2_bins = np.arange(0, 40, 0.25)

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        G = wmt_obj.integrate_transformations("sigma2", bins=sigma2_bins, sum_components=True)
        with ProgressBar():
            G.load()

    print('✓ WMT 计算完成')

    # ====================================================================
    # 只保留 surface_exchange_flux_nonadvective_heat 变量
    # ====================================================================
    # 只选择需要的变量
    target_var = 'surface_exchange_flux_nonadvective_heat'

    if target_var in G.data_vars:
        new_name = f'{config_name}_{target_var}'
        all_results[new_name] = G[target_var]
        print(f'  保存变量: {new_name}')
    else:
        print(f'  警告: 未找到变量 {target_var}')

    print()

# ============================================================================
# 7. 合并所有配置到一个数据集
# ============================================================================
print('\n步骤 6: 合并所有配置到一个数据集...')
print('='*80)

# 创建最终输出数据集
output_ds = xr.Dataset()

# 添加坐标（从第一个变量中获取）
if all_results:
    first_var_name = list(all_results.keys())[0]
    first_var_data = all_results[first_var_name]

    # 复制坐标
    for coord_name in first_var_data.coords:
        output_ds.coords[coord_name] = first_var_data.coords[coord_name]

    # 添加所有变量
    for var_name, var_data in all_results.items():
        output_ds[var_name] = var_data

print(f'总共 {len(all_results)} 个变量（7个配置）')
print()

# 添加全局属性
output_ds.attrs['title'] = 'ECHAM heat flux WMT analysis - Southern Ocean'
output_ds.attrs['description'] = '7 different heat flux configurations'
output_ds.attrs['configurations'] = ', '.join(wmt_configs.keys())
output_ds.attrs['region'] = 'Southern Ocean (50°S-90°S)'
output_ds.attrs['experiment'] = 'LGM'

# ============================================================================
# 8. 保存最终结果
# ============================================================================
print('步骤 7: 保存结果...')
print('-'*80)

output_file = 'wmt_echam_all_configs_southern_ocean_lgm.nc'
print(f'保存到: {output_file}')
output_ds.to_netcdf(output_file)

print()
print('='*80)
print('所有计算完成!')
print('='*80)
print()
print('输出文件包含7个配置:')
for config_name, config in wmt_configs.items():
    print(f'  - {config_name}: {config["description"]}')
