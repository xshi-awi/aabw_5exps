#!/usr/bin/env python
"""
使用 ECHAM 热通量数据 + FESOM 淡水通量数据计算4个区域的 WMT
- 计算7个不同的热通量配置
- 添加完整的淡水通量计算（来自FESOM）
- 每个区域保存到一个文件中，包含感兴趣的变量

与原始 wmt_echam_all_configs_pi.py 的主要改进：
1. 添加淡水通量计算（evap, prec, snow, runoff, fsitherm）
2. 预处理时反转ECHAM热通量符号（ECHAM约定 → xwmt约定）
3. 选择性输出感兴趣的变量：
   热通量: surface_exchange_flux_nonadvective_heat
   盐通量: surface_ocean_flux_advective_negative_rhs_*_salt,
           surface_exchange_flux_nonadvective_salt
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

# ============================================================================
# 定义区域
# ============================================================================
REGIONS = {
    'Southern_Ocean': {
        'lat_range': (-90, -60),
        'lon_range': (-180, 180),
        'description': 'Southern Ocean (south of 60°S)'
    },
    'Ross_Sea': {
        'lat_range': (-90, -60),
        'lon_range': (-180, -60),
        'description': 'Ross Sea AABW source (50°S-90°S, 180°W-60°W)'
    },
    'Weddell_Sea': {
        'lat_range': (-90, -60),
        'lon_range': (-60, 79),
        'description': 'Weddell Sea AABW source (50°S-90°S, 60°W-79°E)'
    },
    'Adelie': {
        'lat_range': (-90, -60),
        'lon_range': (79, 180),
        'description': 'Adélie Land AABW source (50°S-90°S, 79°E-180°E)'
    }
}

# 定义感兴趣的输出变量
VARIABLES_TO_SAVE = {
    'heat': [
        'surface_exchange_flux_nonadvective_heat',
    ],
    'salt': [
        'surface_ocean_flux_advective_negative_rhs_evaporation_salt',
        'surface_ocean_flux_advective_negative_rhs_snow_salt',
        'surface_ocean_flux_advective_negative_rhs_sea_ice_melt_salt',
        'surface_ocean_flux_advective_negative_rhs_rain_and_ice_salt',
        'surface_ocean_flux_advective_negative_rhs_rivers_salt',
        'surface_exchange_flux_nonadvective_salt',
    ]
}

print('='*80)
print('ECHAM 热通量 + FESOM 淡水通量 → xWMT (4个区域 × 7个热通量配置)')
print('='*80)
print()

# ============================================================================
# 1. 加载 ECHAM 数据并预处理（反转符号）
# ============================================================================
print('步骤 1: 加载 ECHAM T63 气候态数据...')
print('-'*80)

echam_file = Path('lgm_echam_clim.nc')
echam_ds = xr.open_dataset(echam_file)

echam_lon = echam_ds['lon'].values
echam_lat = echam_ds['lat'].values
echam_time = echam_ds['time'].values

# 提取4个热通量分量 (time, lat, lon)
print('  提取ECHAM热通量分量（不反转符号）...')
var92 = echam_ds['var92'].values  # LW flux
var95 = echam_ds['var95'].values  # SW flux
var111 = echam_ds['var111'].values  # latent heat
var120 = echam_ds['var120'].values  # sensible heat

print(f'  ECHAM 网格: lon({len(echam_lon)}) × lat({len(echam_lat)})')
print(f'  时间步数: {len(echam_time)}')
print(f'  使用ECHAM原始符号约定')
print()

# ============================================================================
# 2. 加载 FESOM 网格信息
# ============================================================================
print('步骤 2: 加载 FESOM 三角网格信息...')
print('-'*80)

DATA_DIR = Path(".")

mesh = xr.open_dataset("/home/a/a270064/bb1029/inputs/mesh_glac1d/fesom.mesh.diag.nc")
nod_area_dims = mesh['nod_area'].dims
if 'nl' in nod_area_dims:
    areacello_1d = mesh['nod_area'].isel(nl=0).values
elif 'nz' in nod_area_dims:
    areacello_1d = mesh['nod_area'].isel(nz=0).values
else:
    raise ValueError(f"Unexpected dimensions in nod_area: {nod_area_dims}")
fesom_lon_1d = np.degrees(mesh['nodes'].isel(n2=0).values)
fesom_lat_1d = np.degrees(mesh['nodes'].isel(n2=1).values)

nnodes = len(fesom_lon_1d)
print(f'  FESOM 节点数: {nnodes}')
print()

# 加载表面示踪剂
print('  加载 FESOM 表面示踪剂和通量数据...')
sst_data = xr.open_dataset(DATA_DIR / "sst_clim.nc")
sss_data = xr.open_dataset(DATA_DIR / "sss_clim.nc")
time_coord = sst_data['time']
ntime = len(time_coord)
print(f'  时间步数: {ntime}')

# 加载淡水通量数据
prec_data = xr.open_dataset(DATA_DIR / "prec_clim.nc")
snow_data = xr.open_dataset(DATA_DIR / "snow_clim.nc")
evap_data = xr.open_dataset(DATA_DIR / "evap_clim.nc")
runoff_data = xr.open_dataset(DATA_DIR / "runoff_clim.nc")
fw_data = xr.open_dataset(DATA_DIR / "fw_clim.nc")
print('  ✓ 淡水通量数据加载完成')

# 加载海冰密集度数据 (用于开放水域修正)
aice_data = xr.open_dataset(DATA_DIR / "a_ice_clim.nc")
print('  ✓ 海冰密集度数据加载完成')
print()

# ============================================================================
# 3. 插值 ECHAM → FESOM
# ============================================================================
print('步骤 3: 插值 ECHAM → FESOM...')
print('-'*80)

# 转换ECHAM经度从0-360到-180-180以匹配FESOM
echam_lon_converted = np.where(echam_lon > 180, echam_lon - 360, echam_lon)

echam_lon_2d, echam_lat_2d = np.meshgrid(echam_lon_converted, echam_lat)

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
# 4. 定义7个热通量配置
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
# 5. 循环处理每个区域
# ============================================================================
print('步骤 5: 循环处理4个区域...')
print('='*80)

for region_name, region_info in REGIONS.items():
    print('\n')
    print('='*80)
    print(f'处理区域: {region_name}')
    print('='*80)
    print(f'描述: {region_info["description"]}')
    print(f'纬度范围: {region_info["lat_range"]}')
    print(f'经度范围: {region_info["lon_range"]}')
    print()

    lat_min, lat_max = region_info['lat_range']
    lon_min, lon_max = region_info['lon_range']

    # 创建区域mask
    lat_mask = (fesom_lat_1d >= lat_min) & (fesom_lat_1d <= lat_max)
    lon_mask = (fesom_lon_1d >= lon_min) & (fesom_lon_1d <= lon_max)
    region_mask = lat_mask & lon_mask

    print(f'区域内节点数: {region_mask.sum()} ({region_mask.sum()/nnodes*100:.1f}%)')
    print()

    # 用于存储当前区域所有配置的结果
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

        # ====================================================================
        # 热通量（ECHAM原始符号）
        # ====================================================================
        ds['rsntds'] = (['time', 'yh', 'xh'], config['rsntds'][:, np.newaxis, :])
        ds['rsntds'].attrs['units'] = 'W m-2'

        ds['rlntds'] = (['time', 'yh', 'xh'], config['rlntds'][:, np.newaxis, :])
        ds['rlntds'].attrs['units'] = 'W m-2'

        ds['hflso'] = (['time', 'yh', 'xh'], config['hflso'][:, np.newaxis, :])
        ds['hflso'].attrs['units'] = 'W m-2'

        ds['hfsso'] = (['time', 'yh', 'xh'], config['hfsso'][:, np.newaxis, :])
        ds['hfsso'].attrs['units'] = 'W m-2'

        # ====================================================================
        # 淡水通量（来自FESOM）
        # 注意：evap 和 snow 需要乘以开放水域覆盖率 (potential fluxes)
        # prec, runoff 是 effective fluxes，不需要修正
        # ====================================================================
        RHO_WATER = 1000.0  # kg/m³

        # 获取海冰密集度 (范围 0-1)
        aice_2d = aice_data['a_ice'].values[:, np.newaxis, :]
        open_water_2d = 1.0 - aice_2d  # 开放水域覆盖率

        # prlq: 降雨 (正值 = 进入海洋) - effective flux
        prlq_2d = prec_data['prec'].values[:, np.newaxis, :] * RHO_WATER
        ds['prlq'] = (['time', 'yh', 'xh'], prlq_2d)
        ds['prlq'].attrs['units'] = 'kg m-2 s-1'
        ds['prlq'].attrs['long_name'] = 'Rainfall Flux'

        # prsn: 降雪 (正值 = 进入海洋) × 开放水域覆盖率 - potential flux
        prsn_2d = snow_data['snow'].values[:, np.newaxis, :] * RHO_WATER #* open_water_2d
        ds['prsn'] = (['time', 'yh', 'xh'], prsn_2d)
        ds['prsn'].attrs['units'] = 'kg m-2 s-1'
        ds['prsn'].attrs['long_name'] = 'Snowfall Flux (open water corrected)'

        # evs: 蒸发 (FESOM: 负值 = 蒸发) × 开放水域覆盖率 - potential flux
        evs_2d = evap_data['evap'].values[:, np.newaxis, :] * RHO_WATER #* open_water_2d
        ds['evs'] = (['time', 'yh', 'xh'], evs_2d)
        ds['evs'].attrs['units'] = 'kg m-2 s-1'
        ds['evs'].attrs['long_name'] = 'Water Evaporation Flux (open water corrected)'

        # friver: 径流 (正值 = 进入海洋)
        friver_2d = runoff_data['runoff'].values[:, np.newaxis, :] * RHO_WATER
        ds['friver'] = (['time', 'yh', 'xh'], friver_2d)
        ds['friver'].attrs['units'] = 'kg m-2 s-1'
        ds['friver'].attrs['long_name'] = 'River Runoff Flux'

        # fsitherm: 海冰热力学通量 = fw - evap - (prec + snow) - runoff
        fw_2d = - fw_data['fw'].values[:, np.newaxis, :] * RHO_WATER  # correct fw
        fsitherm_2d = fw_2d - evs_2d - prlq_2d - prsn_2d - friver_2d
        ds['fsitherm'] = (['time', 'yh', 'xh'], fsitherm_2d)
        ds['fsitherm'].attrs['units'] = 'kg m-2 s-1'
        ds['fsitherm'].attrs['long_name'] = 'Sea Ice Thermodynamic Flux'

        # wfo: 总淡水通量
        ds['wfo'] = (['time', 'yh', 'xh'], fw_2d)
        ds['wfo'].attrs['units'] = 'kg m-2 s-1'
        ds['wfo'].attrs['long_name'] = 'Water Flux into Ocean'

        # 其他淡水通量分量设为 0
        for var in ['ficeberg', 'vprec']:
            ds[var] = (['time', 'yh', 'xh'], np.zeros((ntime, 1, nnodes), dtype=np.float32))
            ds[var].attrs['units'] = 'kg m-2 s-1'

        # ====================================================================
        # 盐通量
        # ====================================================================
        S_ice = 5.0  # psu, 典型海冰盐度
        sfdsi_2d = fsitherm_2d * S_ice * 0.001  # kg m-2 s-1
        ds['sfdsi'] = (['time', 'yh', 'xh'], sfdsi_2d)
        ds['sfdsi'].attrs['units'] = 'kg m-2 s-1'
        ds['sfdsi'].attrs['long_name'] = 'Downward Sea Ice Basal Salt Flux'
        ds['sfdsi'].attrs['note'] = f'Estimated from fsitherm assuming sea ice salinity = {S_ice} psu'

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
        # 选择性保存感兴趣的变量
        # ====================================================================
        print('选择感兴趣的变量...')

        # 热通量变量
        for var_name in VARIABLES_TO_SAVE['heat']:
            if var_name in G.data_vars:
                new_name = f'{config_name}_{var_name}'
                all_results[new_name] = G[var_name]
                print(f'  ✓ 热通量: {new_name}')
            else:
                print(f'  ⚠ 未找到: {var_name}')

        # 盐通量变量（只在第一个配置时保存，因为所有配置的淡水通量相同）
        if config_name == 'total_heat':
            for var_name in VARIABLES_TO_SAVE['salt']:
                if var_name in G.data_vars:
                    all_results[var_name] = G[var_name]
                    print(f'  ✓ 盐通量: {var_name}')
                else:
                    print(f'  ⚠ 未找到: {var_name}')

        print()

    # ========================================================================
    # 合并当前区域的所有配置到一个数据集
    # ========================================================================
    print('\n合并当前区域的结果...')
    print('-'*80)

    # 创建输出数据集
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

    heat_vars = [k for k in all_results.keys() if 'heat' in k]
    salt_vars = [k for k in all_results.keys() if 'salt' in k]

    print(f'总共 {len(all_results)} 个变量')
    print(f'  - 热通量变量: {len(heat_vars)} (7个配置)')
    print(f'  - 盐通量变量: {len(salt_vars)}')
    print()

    # 添加全局属性
    output_ds.attrs['title'] = f'ECHAM heat flux + FESOM freshwater WMT analysis - {region_name}'
    output_ds.attrs['description'] = '7 heat flux configurations + freshwater salt fluxes'
    output_ds.attrs['heat_configurations'] = ', '.join(wmt_configs.keys())
    output_ds.attrs['region'] = region_info['description']
    output_ds.attrs['lat_range'] = str(region_info['lat_range'])
    output_ds.attrs['lon_range'] = str(region_info['lon_range'])
    output_ds.attrs['experiment'] = 'LGM'

    # ========================================================================
    # 保存当前区域的结果
    # ========================================================================
    output_file = f'wmt_echam_heat_fesom_freshwater_{region_name.lower()}_lgm.nc'
    print(f'保存到: {output_file}')
    output_ds.to_netcdf(output_file)
    print(f'文件大小: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB')
    print()

print('\n')
print('='*80)
print('所有区域计算完成!')
print('='*80)
print()
print('输出文件列表:')
for region_name in REGIONS.keys():
    output_file = f'wmt_echam_heat_fesom_freshwater_{region_name.lower()}_lgm.nc'
    print(f'  - {output_file}')
print()
print('每个文件包含:')
print('  - 7个热通量配置 (每个配置1个heat变量)')
print('  - 6个盐通量变量 (来自淡水通量)')
print()
print('热通量配置:')
for config_name, config in wmt_configs.items():
    print(f'  - {config_name}: {config["description"]}')
print()
print('盐通量变量:')
for var_name in VARIABLES_TO_SAVE['salt']:
    print(f'  - {var_name}')
