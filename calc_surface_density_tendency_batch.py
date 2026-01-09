#!/usr/bin/env python
"""
批量计算所有5个试验的海表面各过程引起的密度趋势

输出格式: (time, nod2) - 与FESOM数据格式一致

试验列表:
- pi: Pre-Industrial
- mh: Mid-Holocene
- lig: Last Interglacial
- lgm: Last Glacial Maximum
- mis: MIS3
"""

import warnings
import numpy as np
from dask.diagnostics import ProgressBar
import xarray as xr
import xgcm
import xbudget
from pathlib import Path
from scipy.interpolate import griddata
import sys

# ============================================================================
# 配置
# ============================================================================
EXPERIMENTS = {
    'pi': {
        'name': 'Pre-Industrial',
        'mesh_path': '/home/a/a270064/bb1029/inputs/mesh_core2'
    },
    'mh': {
        'name': 'Mid-Holocene',
        'mesh_path': '/home/a/a270064/bb1029/inputs/mesh_core2'
    },
    'lig': {
        'name': 'Last Interglacial',
        'mesh_path': '/home/a/a270064/bb1029/inputs/mesh_core2'
    },
    'lgm': {
        'name': 'Last Glacial Maximum',
        'mesh_path': '/home/a/a270064/bb1029/inputs/mesh_glac1d'
    },
    'mis': {
        'name': 'MIS3',
        'mesh_path': '/home/a/a270064/bb1029/inputs/mesh_glac1d_38k'
    }
}

S_ICE = 5.0  # psu, 海冰盐度
RHO_WATER = 1000.0  # kg/m³

def process_experiment(exp_name, exp_config):
    """处理单个试验"""

    print('='*80)
    print(f'处理试验: {exp_name.upper()} - {exp_config["name"]}')
    print('='*80)
    print()

    DATA_DIR = Path(exp_name)

    # ========================================================================
    # 1. 加载 ECHAM 数据
    # ========================================================================
    print('步骤 1: 加载 ECHAM 数据...')
    print('-'*80)

    echam_file = DATA_DIR / 'echam_clim.nc'
    if not echam_file.exists():
        print(f'错误: 文件不存在 {echam_file}')
        return False

    echam_ds = xr.open_dataset(echam_file)
    echam_lon = echam_ds['lon'].values
    echam_lat = echam_ds['lat'].values
    echam_time = echam_ds['time'].values

    # 提取4个热通量分量 (ECHAM符号约定)
    var92 = echam_ds['var92'].values  # LW (longwave net)
    var95 = echam_ds['var95'].values  # SW (shortwave net)
    var111 = echam_ds['var111'].values  # LH (latent heat)
    var120 = echam_ds['var120'].values  # SH (sensible heat)

    print(f'  ECHAM网格: {len(echam_lon)} × {len(echam_lat)}, 时间步: {len(echam_time)}')
    print(f'  提取4个热通量分量: LW (var92), SW (var95), LH (var111), SH (var120)')
    print()

    # ========================================================================
    # 2. 加载 FESOM 网格
    # ========================================================================
    print('步骤 2: 加载 FESOM 网格...')
    print('-'*80)

    mesh_path = exp_config['mesh_path']
    mesh = xr.open_dataset(f'{mesh_path}/fesom.mesh.diag.nc')

    # 不同网格的垂直维度名称不同: nl (core2) vs nz (glac1d)
    if 'nl' in mesh['nod_area'].dims:
        areacello_1d = mesh['nod_area'].isel(nl=0).values
    elif 'nz' in mesh['nod_area'].dims:
        areacello_1d = mesh['nod_area'].isel(nz=0).values
    else:
        raise ValueError(f"Unknown vertical dimension in nod_area: {mesh['nod_area'].dims}")

    fesom_lon_1d = np.degrees(mesh['nodes'].isel(n2=0).values)
    fesom_lat_1d = np.degrees(mesh['nodes'].isel(n2=1).values)
    nnodes = len(fesom_lon_1d)

    print(f'  FESOM节点数: {nnodes}')
    print(f'  网格路径: {mesh_path}')
    print()

    # ========================================================================
    # 3. 加载 FESOM 表面数据
    # ========================================================================
    print('步骤 3: 加载 FESOM 表面数据...')
    print('-'*80)

    sst_data = xr.open_dataset(DATA_DIR / "sst_clim.nc")
    sss_data = xr.open_dataset(DATA_DIR / "sss_clim.nc")
    time_coord = sst_data['time']
    ntime = len(time_coord)

    prec_data = xr.open_dataset(DATA_DIR / "prec_clim.nc")
    snow_data = xr.open_dataset(DATA_DIR / "snow_clim.nc")
    evap_data = xr.open_dataset(DATA_DIR / "evap_clim.nc")
    runoff_data = xr.open_dataset(DATA_DIR / "runoff_clim.nc")
    fw_data = xr.open_dataset(DATA_DIR / "fw_clim.nc")
    aice_data = xr.open_dataset(DATA_DIR / "a_ice_clim.nc")

    print(f'  时间步数: {ntime}')
    print()

    # ========================================================================
    # 4. 插值 ECHAM → FESOM
    # ========================================================================
    print('步骤 4: 插值 ECHAM → FESOM...')
    print('-'*80)

    echam_lon_converted = np.where(echam_lon > 180, echam_lon - 360, echam_lon)
    echam_lon_2d, echam_lat_2d = np.meshgrid(echam_lon_converted, echam_lat)

    var92_fesom = np.zeros((ntime, nnodes), dtype=np.float32)
    var95_fesom = np.zeros((ntime, nnodes), dtype=np.float32)
    var111_fesom = np.zeros((ntime, nnodes), dtype=np.float32)
    var120_fesom = np.zeros((ntime, nnodes), dtype=np.float32)

    for t in range(ntime):
        if (t + 1) % 3 == 0:
            print(f'  时间步 {t+1}/{ntime}...')

        points = np.column_stack([echam_lon_2d.flatten(), echam_lat_2d.flatten()])
        xi = np.column_stack([fesom_lon_1d, fesom_lat_1d])

        var92_fesom[t, :] = griddata(points, var92[t, :, :].flatten(), xi, method='nearest')
        var95_fesom[t, :] = griddata(points, var95[t, :, :].flatten(), xi, method='nearest')
        var111_fesom[t, :] = griddata(points, var111[t, :, :].flatten(), xi, method='nearest')
        var120_fesom[t, :] = griddata(points, var120[t, :, :].flatten(), xi, method='nearest')

    print('  ✓ 插值完成')
    print()

    # ========================================================================
    # 5. 准备 xbudget 数据集
    # ========================================================================
    print('步骤 5: 准备 xbudget 数据集...')
    print('-'*80)

    ds = xr.Dataset()

    # 坐标 - 使用xbudget需要的虚拟2D结构
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

    # 网格面积
    areacello_2d = areacello_1d[np.newaxis, :]
    ds['areacello'] = (['yh', 'xh'], areacello_2d)
    ds['areacello'].attrs['units'] = 'm2'

    # 经纬度
    ds['lon'] = (['yh', 'xh'], fesom_lon_1d[np.newaxis, :])
    ds['lon'].attrs['units'] = 'degrees_east'
    ds['lat'] = (['yh', 'xh'], fesom_lat_1d[np.newaxis, :])
    ds['lat'].attrs['units'] = 'degrees_north'

    # 热通量 - 使用xbudget变量命名
    # rsntds = SW, rlntds = LW, hflso = LH, hfsso = SH
    ds['rsntds'] = (['time', 'yh', 'xh'], var95_fesom[:, np.newaxis, :])
    ds['rsntds'].attrs['units'] = 'W m-2'
    ds['rsntds'].attrs['long_name'] = 'Surface Net Downward Shortwave Radiation'

    ds['rlntds'] = (['time', 'yh', 'xh'], var92_fesom[:, np.newaxis, :])
    ds['rlntds'].attrs['units'] = 'W m-2'
    ds['rlntds'].attrs['long_name'] = 'Surface Net Downward Longwave Radiation'

    ds['hflso'] = (['time', 'yh', 'xh'], var111_fesom[:, np.newaxis, :])
    ds['hflso'].attrs['units'] = 'W m-2'
    ds['hflso'].attrs['long_name'] = 'Surface Downward Latent Heat Flux'

    ds['hfsso'] = (['time', 'yh', 'xh'], var120_fesom[:, np.newaxis, :])
    ds['hfsso'].attrs['units'] = 'W m-2'
    ds['hfsso'].attrs['long_name'] = 'Surface Downward Sensible Heat Flux'

    # 淡水通量
    aice_2d = aice_data['a_ice'].values[:, np.newaxis, :]

    prlq_2d = prec_data['prec'].values[:, np.newaxis, :] * RHO_WATER
    ds['prlq'] = (['time', 'yh', 'xh'], prlq_2d)
    ds['prlq'].attrs['units'] = 'kg m-2 s-1'

    prsn_2d = snow_data['snow'].values[:, np.newaxis, :] * RHO_WATER
    ds['prsn'] = (['time', 'yh', 'xh'], prsn_2d)
    ds['prsn'].attrs['units'] = 'kg m-2 s-1'

    evs_2d = evap_data['evap'].values[:, np.newaxis, :] * RHO_WATER
    ds['evs'] = (['time', 'yh', 'xh'], evs_2d)
    ds['evs'].attrs['units'] = 'kg m-2 s-1'

    friver_2d = runoff_data['runoff'].values[:, np.newaxis, :] * RHO_WATER
    ds['friver'] = (['time', 'yh', 'xh'], friver_2d)
    ds['friver'].attrs['units'] = 'kg m-2 s-1'

    fw_2d = - fw_data['fw'].values[:, np.newaxis, :] * RHO_WATER
    fsitherm_2d = fw_2d - evs_2d - prlq_2d - prsn_2d - friver_2d
    ds['fsitherm'] = (['time', 'yh', 'xh'], fsitherm_2d)
    ds['fsitherm'].attrs['units'] = 'kg m-2 s-1'

    ds['wfo'] = (['time', 'yh', 'xh'], fw_2d)
    ds['wfo'].attrs['units'] = 'kg m-2 s-1'

    for var in ['ficeberg', 'vprec']:
        ds[var] = (['time', 'yh', 'xh'], np.zeros((ntime, 1, nnodes), dtype=np.float32))
        ds[var].attrs['units'] = 'kg m-2 s-1'

    # 盐通量
    sfdsi_2d = fsitherm_2d * S_ICE * 0.001
    ds['sfdsi'] = (['time', 'yh', 'xh'], sfdsi_2d)
    ds['sfdsi'].attrs['units'] = 'kg m-2 s-1'

    print('  ✓ 数据集准备完成')
    print()

    # ========================================================================
    # 6. 定义热通量配置 (分别计算每个热通量分量的贡献)
    # ========================================================================
    print('步骤 6: 定义热通量配置...')
    print('-'*80)

    heat_flux_configs = {
        'total': {
            'description': '总热通量 (SW + LW + LH + SH)',
            'rsntds': var95_fesom,
            'rlntds': var92_fesom,
            'hflso': var111_fesom,
            'hfsso': var120_fesom
        },
        'sw': {
            'description': '短波辐射 (SW only)',
            'rsntds': var95_fesom,
            'rlntds': np.zeros_like(var92_fesom),
            'hflso': np.zeros_like(var111_fesom),
            'hfsso': np.zeros_like(var120_fesom)
        },
        'lw': {
            'description': '长波辐射 (LW only)',
            'rsntds': np.zeros_like(var95_fesom),
            'rlntds': var92_fesom,
            'hflso': np.zeros_like(var111_fesom),
            'hfsso': np.zeros_like(var120_fesom)
        },
        'lh': {
            'description': '潜热通量 (LH only)',
            'rsntds': np.zeros_like(var95_fesom),
            'rlntds': np.zeros_like(var92_fesom),
            'hflso': var111_fesom,
            'hfsso': np.zeros_like(var120_fesom)
        },
        'sh': {
            'description': '感热通量 (SH only)',
            'rsntds': np.zeros_like(var95_fesom),
            'rlntds': np.zeros_like(var92_fesom),
            'hflso': np.zeros_like(var111_fesom),
            'hfsso': var120_fesom
        }
    }

    print(f'  定义了 {len(heat_flux_configs)} 个热通量配置')
    for config_name, config in heat_flux_configs.items():
        print(f'    - {config_name}: {config["description"]}')
    print()

    # ========================================================================
    # 7. 循环运行 xbudget (对每个热通量配置)
    # ========================================================================
    print('步骤 7: 对每个热通量配置运行 xbudget...')
    print('-'*80)

    coords = {
        'X': {'center': 'xh', 'outer': 'xq'},
        'Y': {'center': 'yh', 'outer': 'yq'}
    }
    boundary = {'X': 'extend', 'Y': 'extend'}
    metrics = {('X', 'Y'): 'areacello'}

    # 存储每个配置的预算结果
    all_heat_budgets = {}

    for config_name, config in heat_flux_configs.items():
        print(f'\n  配置: {config_name} - {config["description"]}')
        print('  ' + '-'*76)

        # 创建临时数据集,使用当前配置的热通量
        ds_temp = ds.copy()
        ds_temp['rsntds'] = (['time', 'yh', 'xh'], config['rsntds'][:, np.newaxis, :])
        ds_temp['rlntds'] = (['time', 'yh', 'xh'], config['rlntds'][:, np.newaxis, :])
        ds_temp['hflso'] = (['time', 'yh', 'xh'], config['hflso'][:, np.newaxis, :])
        ds_temp['hfsso'] = (['time', 'yh', 'xh'], config['hfsso'][:, np.newaxis, :])

        # 创建网格
        grid = xgcm.Grid(ds_temp, coords=coords, metrics=metrics, boundary=boundary,
                         autoparse_metadata=False)

        # 运行 xbudget
        xbudget_dict = xbudget.load_preset_budget(model="MOM6_surface")
        xbudget.collect_budgets(ds_temp, xbudget_dict)

        decompose_list = [
            "surface_exchange_flux",
            "advective",
            "surface_ocean_flux_advective_negative_rhs"
        ]
        decomposed_budgets = xbudget.aggregate(xbudget_dict, decompose=decompose_list)

        # 提取热量预算项 (只保存非平流项,因为这是表面强迫)
        if 'heat' in decomposed_budgets and 'rhs' in decomposed_budgets['heat']:
            all_heat_budgets[config_name] = {}
            for term_name, var_name in decomposed_budgets['heat']['rhs'].items():
                if 'nonadvective' in term_name and var_name in grid._ds.data_vars:
                    all_heat_budgets[config_name][term_name] = grid._ds[var_name].values[:, 0, :]
                    print(f'    ✓ 提取 {term_name}')

        # 只在第一次运行时保存 mass 和 salt 预算 (因为它们不依赖热通量)
        if config_name == 'total':
            # 保存完整的 decomposed_budgets 用于后续提取
            decomposed_budgets_total = decomposed_budgets
            grid_total = grid

    print('\n  ✓ 所有热通量配置的 xbudget 计算完成')
    print()

    # ========================================================================
    # 8. 提取变量并重塑为 (time, nod2) 格式
    # ========================================================================
    print('步骤 8: 提取变量并重塑为 (time, nod2) 格式...')
    print('-'*80)

    output_ds = xr.Dataset()

    # 坐标: 使用 nod2 而不是 xh
    output_ds.coords['time'] = time_coord
    output_ds.coords['nod2'] = np.arange(nnodes)

    # 经纬度 (1D)
    output_ds['lon'] = (['nod2'], fesom_lon_1d)
    output_ds['lon'].attrs['units'] = 'degrees_east'
    output_ds['lat'] = (['nod2'], fesom_lat_1d)
    output_ds['lat'].attrs['units'] = 'degrees_north'

    # 面积 (1D)
    output_ds['areacello'] = (['nod2'], areacello_1d)
    output_ds['areacello'].attrs['units'] = 'm2'

    # 示踪剂: (time, yh, xh) → (time, nod2)
    output_ds['tos'] = (['time', 'nod2'], sst_data['sst'].values)
    output_ds['tos'].attrs['units'] = 'degC'
    output_ds['sos'] = (['time', 'nod2'], sss_data['sss'].values)
    output_ds['sos'].attrs['units'] = 'psu'

    # 热通量分量: (time, yh, xh) → (time, nod2)
    # 保存4个独立的热通量分量
    output_ds['SW_flux'] = (['time', 'nod2'], var95_fesom)
    output_ds['SW_flux'].attrs['units'] = 'W m-2'
    output_ds['SW_flux'].attrs['long_name'] = 'Shortwave radiation flux (var95)'
    output_ds['SW_flux'].attrs['source'] = 'ECHAM var95 (rsntds)'

    output_ds['LW_flux'] = (['time', 'nod2'], var92_fesom)
    output_ds['LW_flux'].attrs['units'] = 'W m-2'
    output_ds['LW_flux'].attrs['long_name'] = 'Longwave radiation flux (var92)'
    output_ds['LW_flux'].attrs['source'] = 'ECHAM var92 (rlntds)'

    output_ds['LH_flux'] = (['time', 'nod2'], var111_fesom)
    output_ds['LH_flux'].attrs['units'] = 'W m-2'
    output_ds['LH_flux'].attrs['long_name'] = 'Latent heat flux (var111)'
    output_ds['LH_flux'].attrs['source'] = 'ECHAM var111 (hflso)'

    output_ds['SH_flux'] = (['time', 'nod2'], var120_fesom)
    output_ds['SH_flux'].attrs['units'] = 'W m-2'
    output_ds['SH_flux'].attrs['long_name'] = 'Sensible heat flux (var120)'
    output_ds['SH_flux'].attrs['source'] = 'ECHAM var120 (hfsso)'

    # 淡水和盐通量: (time, yh, xh) → (time, nod2)
    flux_mapping = {
        'evs': (evap_data['evap'].values * RHO_WATER, 'kg m-2 s-1', 'Water Evaporation Flux'),
        'prlq': (prec_data['prec'].values * RHO_WATER, 'kg m-2 s-1', 'Liquid Precipitation Flux'),
        'prsn': (snow_data['snow'].values * RHO_WATER, 'kg m-2 s-1', 'Snowfall Flux'),
        'friver': (runoff_data['runoff'].values * RHO_WATER, 'kg m-2 s-1', 'River Runoff Flux'),
        'fsitherm': (fsitherm_2d[:, 0, :], 'kg m-2 s-1', 'Sea Ice Thermodynamic Flux'),
        'wfo': (fw_2d[:, 0, :], 'kg m-2 s-1', 'Water Flux into Ocean'),
        'sfdsi': (sfdsi_2d[:, 0, :], 'kg m-2 s-1', 'Downward Sea Ice Basal Salt Flux')
    }

    for var_name, (var_data, units, long_name) in flux_mapping.items():
        output_ds[var_name] = (['time', 'nod2'], var_data)
        output_ds[var_name].attrs['units'] = units
        output_ds[var_name].attrs['long_name'] = long_name

    # xbudget 预算项: (time, yh, xh) → (time, nod2)
    print('  提取质量预算项...')
    if 'mass' in decomposed_budgets_total and 'rhs' in decomposed_budgets_total['mass']:
        for term_name, var_name in decomposed_budgets_total['mass']['rhs'].items():
            if var_name in grid_total._ds.data_vars:
                data_3d = grid_total._ds[var_name].values  # (time, yh, xh)
                data_2d = data_3d[:, 0, :]  # (time, nod2)
                output_ds[var_name] = (['time', 'nod2'], data_2d)
                output_ds[var_name].attrs['units'] = 'kg m-2 s-1'
                print(f'    ✓ {var_name}')

    print('  提取热量预算项 (按热通量分量)...')
    # 为每个热通量配置保存非平流预算项
    for config_name, budget_terms in all_heat_budgets.items():
        for term_name, data_2d in budget_terms.items():
            # 构造变量名: heat_rhs_sum_<config>_<term>
            # 例如: heat_rhs_sum_sw_surface_exchange_flux_sum_nonadvective
            var_name_parts = term_name.split('_')
            # 找到 'sum' 后面的部分
            if 'sum' in var_name_parts:
                sum_idx = var_name_parts.index('sum')
                # 在 'sum' 后插入配置名
                new_var_name = '_'.join(var_name_parts[:sum_idx+1] + [config_name] + var_name_parts[sum_idx+1:])
            else:
                new_var_name = f'{term_name}_{config_name}'

            output_ds[new_var_name] = (['time', 'nod2'], data_2d)
            output_ds[new_var_name].attrs['units'] = 'W m-2'
            output_ds[new_var_name].attrs['long_name'] = f'Heat budget from {config_name} flux'
            print(f'    ✓ {new_var_name}')

    print('  提取盐度预算项...')
    if 'salt' in decomposed_budgets_total and 'rhs' in decomposed_budgets_total['salt']:
        for term_name, var_name in decomposed_budgets_total['salt']['rhs'].items():
            if var_name in grid_total._ds.data_vars:
                data_3d = grid_total._ds[var_name].values  # (time, yh, xh)
                data_2d = data_3d[:, 0, :]  # (time, nod2)
                output_ds[var_name] = (['time', 'nod2'], data_2d)
                output_ds[var_name].attrs['units'] = 'psu s-1'
                print(f'    ✓ {var_name}')

    print()

    # ========================================================================
    # 9. 保存
    # ========================================================================
    print('步骤 9: 保存到 NetCDF...')
    print('-'*80)

    output_ds.attrs['title'] = f'Surface density tendency analysis - {exp_config["name"]}'
    output_ds.attrs['experiment'] = exp_name
    output_ds.attrs['mesh_path'] = mesh_path
    output_ds.attrs['description'] = 'Heat and salt tendencies from surface fluxes using xbudget'
    output_ds.attrs['output_format'] = '(time, nod2) - compatible with FESOM data format'
    output_ds.attrs['heat_flux_components'] = 'SW_flux, LW_flux, LH_flux, SH_flux (from ECHAM var95, var92, var111, var120)'
    output_ds.attrs['heat_flux_configs'] = ', '.join(heat_flux_configs.keys())
    output_ds.attrs['note'] = 'Heat budget terms computed separately for each flux component (total, sw, lw, lh, sh)'

    output_file = DATA_DIR / f'surface_density_tendency_{exp_name}.nc'
    print(f'保存到: {output_file}')

    with ProgressBar():
        output_ds.to_netcdf(output_file)

    file_size = output_file.stat().st_size / 1024 / 1024
    print(f'文件大小: {file_size:.2f} MB')
    print()

    # 验证输出格式
    print('验证输出格式:')
    check_ds = xr.open_dataset(output_file)
    print(f'  维度: {dict(check_ds.dims)}')
    print(f'  变量数: {len(check_ds.data_vars)}')
    sample_var = list(check_ds.data_vars)[0]
    print(f'  示例变量 {sample_var} 形状: {check_ds[sample_var].shape}')
    check_ds.close()
    print()

    return True

# ============================================================================
# 主程序
# ============================================================================
if __name__ == '__main__':
    print('='*80)
    print('批量计算海表面密度趋势 - 5个试验')
    print('='*80)
    print()

    print('试验列表:')
    for exp_name, exp_config in EXPERIMENTS.items():
        print(f'  {exp_name:5s} - {exp_config["name"]:25s} [{exp_config["mesh_path"]}]')
    print()

    success_count = 0
    failed_exps = []

    for exp_name, exp_config in EXPERIMENTS.items():
        try:
            success = process_experiment(exp_name, exp_config)
            if success:
                success_count += 1
                print(f'✓ {exp_name.upper()} 完成\n')
            else:
                failed_exps.append(exp_name)
                print(f'✗ {exp_name.upper()} 失败\n')
        except Exception as e:
            print(f'✗ {exp_name.upper()} 错误: {e}\n')
            failed_exps.append(exp_name)

    # ========================================================================
    # 总结
    # ========================================================================
    print('='*80)
    print('批量处理完成!')
    print('='*80)
    print()
    print(f'成功: {success_count}/{len(EXPERIMENTS)} 个试验')
    if failed_exps:
        print(f'失败: {", ".join(failed_exps)}')
    print()

    print('输出文件:')
    for exp_name in EXPERIMENTS.keys():
        output_file = Path(exp_name) / f'surface_density_tendency_{exp_name}.nc'
        if output_file.exists():
            size_mb = output_file.stat().st_size / 1024 / 1024
            print(f'  {output_file} ({size_mb:.2f} MB)')
    print()

    print('输出格式: (time, nod2) - 与FESOM数据格式一致')
    print()
