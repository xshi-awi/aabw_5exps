load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"


begin

; 实验列表（paleo实验）
exps = (/"mh", "lig", "lgm", "mis"/)
exp_names = (/"MH-PI", "LIG-PI", "LGM-PI", "MIS3-PI"/)
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

; 读取PI实验数据作为参考
f = addfile("pi/age_reg.nc", "r")
age_data_pi = f->age(0, depth_idx, :, :)
age_pi = tofloat(age_data_pi)
age_pi@_FillValue = -999.0
age_pi = where(isnan_ieee(age_pi), -999.0, age_pi)

; 读取PI的mask
fs = addfile("pi/mask.nc", "r")
salt_mask_pi = fs->salt(0, depth_idx, :, :)
age_pi = where(isnan_ieee(salt_mask_pi), -999.0, age_pi)

; 只保留50S以南的数据
do ilat = 0, dimsizes(lat)-1
  if (lat(ilat) .ge. -50.0) then
    age_pi(ilat, :) = -999.0
  end if
end do

; 存储所有实验的anomaly数据
age_anomaly = new((/nexp, dimsizes(lat), dimsizes(lon)/), float)
age_anomaly@_FillValue = -999.0

; 计算每个paleo实验相对于PI的anomaly
do i = 0, nexp-1
  ; 读取paleo实验的age数据
  f = addfile(exps(i) + "/age_reg.nc", "r")
  age_data = f->age(0, depth_idx, :, :)

  ; 转换为float并处理缺失值
  age_tmp = tofloat(age_data)
  age_tmp@_FillValue = -999.0
  age_tmp = where(isnan_ieee(age_tmp), -999.0, age_tmp)

  ; 读取mask
  fs = addfile(exps(i) + "/mask.nc", "r")
  salt_mask = fs->salt(0, depth_idx, :, :)
  age_tmp = where(isnan_ieee(salt_mask), -999.0, age_tmp)

  ; 只保留50S以南的数据
  do ilat = 0, dimsizes(lat)-1
    if (lat(ilat) .ge. -50.0) then
      age_tmp(ilat, :) = -999.0
    end if
  end do

  ; 计算anomaly (paleo - PI)
  age_anomaly(i, :, :) = age_tmp - age_pi

  ; 如果PI或paleo其中一个是缺失值，则anomaly也设为缺失值
  age_anomaly(i, :, :) = where(age_pi .eq. -999.0 .or. age_tmp .eq. -999.0, -999.0, age_anomaly(i, :, :))

  ; 将绝对值 < 5 的值设为 0（在色标中显示为白色）
  age_anomaly(i, :, :) = where(abs(age_anomaly(i, :, :)) .lt. 5.0 .and. age_anomaly(i, :, :) .ne. -999.0, 0.0, age_anomaly(i, :, :))
end do

; 创建图形
wks = gsn_open_wks("pdf", "age_horizontal_3900m_anomaly")
gsn_define_colormap(wks, "testcmap")  ; 使用与age_vertical_anm相同的色标

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

; 等值线级别（与age_vertical_anm相同，去除-5到5之间的值以显示为白色）
res@cnLevelSelectionMode = "ExplicitLevels"
res@cnLevels = (/-1500, -1000, -800, -600, -400, -300, -200, -150, -100, -75, -50, -25, -5, 5, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1500/)
res@lbLabelStride = 1

; 缺失值设置
res@cnMissingValPerimOn = True
res@cnMissingValFillPattern = 0
res@cnMissingValFillColor = "gray"

; 图形大小
res@vpWidthF = 0.35
res@vpHeightF = 0.35

; 绘制每个实验的anomaly
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
  age_plot = (/age_anomaly(i, :, :)/)

  map(i) = gsn_csm_contour_map_polar(wks, age_plot, res)
end do

; Panel设置
pres = True
pres@gsnFrame = False
pres@gsnDraw = True
pres@amJust = "TopLeft"
pres@gsnPanelLabelBar = True
pres@lbBoxLinesOn = False
; 对称的label显示 - 显示选定的对称值（与age_vertical_anm相同）
; 26 levels: -1500,-1000,-800,-600,-400,-300,-200,-150,-100,-75,-50,-25,-5,5,25,50,75,100,150,200,300,400,600,800,1000,1500
pres@lbLabelStrings = (/"-1500", "", "-800", "", "-400", "", "", "-150", "", "-75", "", "-25", "", "", "25", "", "75", "", "150", "", "", "400", "", "800", "", "1500"/)
pres@lbLabelAutoStride = False
pres@gsnPanelFigureStrings = exp_names
pres@gsnPanelFigureStringsFontHeightF = 0.012
pres@gsnPanelMainString = "Age Anomaly at 3900m depth (50~S~o~N~S-90~S~o~N~S) [year]"

; 绘制panel（4个子图排成2x2）
gsn_panel(wks, map, (/2, 2/), pres)

frame(wks)

end
