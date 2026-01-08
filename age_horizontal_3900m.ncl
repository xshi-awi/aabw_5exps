load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"


begin

; 读取各个实验的数据
exps = (/"pi", "mh", "lig", "lgm", "mis"/)
exp_names = (/"PI", "MH", "LIG", "LGM", "MIS3"/)
nexp = dimsizes(exps)

; 读取第一个文件获取维度信息
f = addfile("pi/age_reg.nc", "r")
age_full = f->age(0,:,:,:)  ; time, depth_coord, lat, lon
depth = f->depth_coord
lat = f->lat
lon = f->lon

; 选择3900m深度（找到最接近的深度层，注意深度是负值）
depth_target = -3900.0
depth_diff = abs(depth - depth_target)
depth_idx = minind(depth_diff)
print("Selected depth: " + depth(depth_idx) + " m (index: " + depth_idx + ")")

; 存储所有实验的数据
age_maps = new((/nexp, dimsizes(lat), dimsizes(lon)/), float)
age_maps@_FillValue = -999.0

do i = 0, nexp-1
  ; 读取age数据
  f = addfile(exps(i) + "/age_reg.nc", "r")
  age_data = f->age(0, depth_idx, :, :)  ; 选择3900m深度

  ; 转换为float并处理缺失值
  age_tmp = tofloat(age_data)
  age_tmp@_FillValue = -999.0
  age_tmp = where(isnan_ieee(age_tmp), -999.0, age_tmp)

  ; 读取mask
  fs = addfile(exps(i) + "/mask.nc", "r")
  salt_mask = fs->salt(0, depth_idx, :, :)
  age_tmp = where(isnan_ieee(salt_mask), -999.0, age_tmp)

  ; 只保留50S以南的数据（设置50S以北为缺失值）
  do ilat = 0, dimsizes(lat)-1
    if (lat(ilat) .ge. -50.0) then
      age_tmp(ilat, :) = -999.0
    end if
  end do

  age_maps(i, :, :) = age_tmp
end do

; 创建图形
wks = gsn_open_wks("pdf", "age_horizontal_3900m")
gsn_define_colormap(wks, "BlAqGrYeOrRe")

map = new(nexp, graphic)

; 设置图形参数
res = True
res@gsnDraw = False
res@gsnFrame = False
res@gsnSpreadColors = True
res@gsnPolar = "SH"  ; 南半球极地投影
res@gsnRightString = ""

; 地图投影设置
res@mpMinLatF = -90  ; 南极
res@mpMaxLatF = -50  ; 50S
res@mpCenterLonF = 0

; 等值线设置
res@cnFillMode = "RasterFill"
res@cnFillDrawOrder = "PreDraw"
res@cnLineDrawOrder = "PreDraw"
res@cnLinesOn = False
res@cnFillOn = True
res@cnInfoLabelOn = False
res@lbLabelBarOn = False
res@lbOrientation = "Vertical"
res@lbBoxLinesOn = False

; 等值线级别
res@cnLevelSelectionMode = "ExplicitLevels"
res@cnLevels = fspan(0, 2500, 21)

; 缺失值设置
res@cnMissingValPerimOn = True
res@cnMissingValFillPattern = 0
res@cnMissingValFillColor = "gray"

; 图形大小
res@vpWidthF = 0.35
res@vpHeightF = 0.35

; 绘制每个实验
do i = 0, nexp-1
  res@gsnLeftString = exp_names(i) + " (3900m)"
  res@gsnLeftStringFontHeightF = 0.02

  ; 准备数据（创建新变量并添加坐标）
  age_plot = new((/dimsizes(lat), dimsizes(lon)/), float)
  age_plot!0 = "lat"
  age_plot!1 = "lon"
  age_plot&lat = lat
  age_plot&lon = lon
  age_plot@_FillValue = -999.0
  age_plot = (/age_maps(i, :, :)/)

  map(i) = gsn_csm_contour_map_polar(wks, age_plot, res)
end do

; Panel设置
pres = True
pres@gsnFrame = False
pres@gsnDraw = True
pres@amJust = "TopLeft"
pres@gsnPanelLabelBar = True
pres@lbBoxLinesOn = False
pres@gsnPanelFigureStrings = exp_names
pres@gsnPanelFigureStringsFontHeightF = 0.012
pres@gsnPanelMainString = "Age at 3900m depth (50~S~o~N~S-90~S~o~N~S)"

; 绘制panel（5个子图排成一行或两行）
gsn_panel(wks, map, (/2, 3/), pres)

frame(wks)

end
