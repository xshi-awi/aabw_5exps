load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"


begin


f    = addfile ("pi/age_reg.nc", "r")
a=f->age(0,:,:,:)  ;time,depth,lat,lon
a@_FillValue=-999
a=where(isnan_ieee(a),-999,a)
fs   = addfile ("pi/mask.nc", "r")
salt_mask=fs->salt(0,:,:,:)
a=where(isnan_ieee(salt_mask),-999,a)

a_n=dim_avg_n_Wrap(a(:,:,{-70:-20}),2)
a_i=dim_avg_n_Wrap(a(:,:,{40:110}),2)
a(:,:,{-70:110})=a@_FillValue
a_p=dim_avg_n_Wrap(a(:,:,:),2)


f    = addfile ("lgm/age_reg.nc", "r")
b=f->age(0,:,:,:)  ;time,depth,lat,lon
b@_FillValue=-999
b=where(isnan_ieee(b),-999,b)
fs   = addfile ("lgm/mask.nc", "r")
salt_mask=fs->salt(0,:,:,:)
b=where(isnan_ieee(salt_mask),-999,b)

b_n=dim_avg_n_Wrap(b(:,:,{-70:-20}),2)
b_i=dim_avg_n_Wrap(b(:,:,{40:110}),2)
b(:,:,{-70:110})=b@_FillValue
b_p=dim_avg_n_Wrap(b(:,:,:),2)


f    = addfile ("mh/age_reg.nc", "r")
c=f->age(0,:,:,:)  ;time,depth,lat,lon
c@_FillValue=-999
c=where(isnan_ieee(c),-999,c)
fs   = addfile ("mh/mask.nc", "r")
salt_mask=fs->salt(0,:,:,:)
c=where(isnan_ieee(salt_mask),-999,c)

c_n=dim_avg_n_Wrap(c(:,:,{-70:-20}),2)
c_i=dim_avg_n_Wrap(c(:,:,{40:110}),2)
c(:,:,{-70:110})=c@_FillValue
c_p=dim_avg_n_Wrap(c(:,:,:),2)


f    = addfile ("lig/age_reg.nc", "r")
d=f->age(0,:,:,:)  ;time,depth,lat,lon
d@_FillValue=-999
d=where(isnan_ieee(d),-999,d)
fs   = addfile ("lig/mask.nc", "r")
salt_mask=fs->salt(0,:,:,:)
d=where(isnan_ieee(salt_mask),-999,d)

d_n=dim_avg_n_Wrap(d(:,:,{-70:-20}),2)
d_i=dim_avg_n_Wrap(d(:,:,{40:110}),2)
d(:,:,{-70:110})=d@_FillValue
d_p=dim_avg_n_Wrap(d(:,:,:),2)


f    = addfile ("mis/age_reg.nc", "r")
e=f->age(0,:,:,:)  ;time,depth,lat,lon
e@_FillValue=-999
e=where(isnan_ieee(e),-999,e)
fs   = addfile ("mis/mask.nc", "r")
salt_mask=fs->salt(0,:,:,:)
e=where(isnan_ieee(salt_mask),-999,e)

e_n=dim_avg_n_Wrap(e(:,:,{-70:-20}),2)
e_i=dim_avg_n_Wrap(e(:,:,{40:110}),2)
e(:,:,{-70:110})=e@_FillValue
e_p=dim_avg_n_Wrap(e(:,:,:),2)

  wks = gsn_open_wks("pdf","age_vertical")
gsn_define_colormap(wks, "BlAqGrYeOrRe")
;gsn_define_colormap(wks, "BlWhRe")
  map=new(15,graphic)
res                  = True
res@tiXAxisString=""
res@gsnDraw          = False
res@gsnFrame         = False
res@gsnSpreadColors  = True
res@gsnRightString=""
;res@cnSmoothingOn    = True
res@cnFillMode    = "RasterFill"
res@cnFillDrawOrder  = "PreDraw"
res@cnLineDrawOrder  = "PreDraw"
res@cnLinesOn        = False
res@cnFillOn         = True
res@cnMonoLevelFlag=True
res@cnLevelFlag="NoLine"
res@lbLabelBarOn     =False
res@lbOrientation="Vertical"
res@lbBoxLinesOn=False
res@cnInfoLabelOn   = False
;res@lbLabelStride=25
res@cnLevelSelectionMode = "ExplicitLevels"
  res@vpWidthF               = 0.8
  res@vpHeightF              = 0.4
res@gsnYAxisIrregular2Linear=  True
res@cnLevelSelectionMode = "ExplicitLevels" ; explicit contour levels
res@cnLevels=fspan(0,2500,21)
  res@cnMissingValPerimOn     = True             ; turn on perimeter
  res@cnMissingValFillPattern = 0                ; choose a fill pattern
  res@cnMissingValFillColor   = "gray"          ; choose a pattern color
  res@tmYLMode="Explicit"
  res@tmYLValues =(/0,-1000,-2000,-3000,-4000,-5000/)
  res@tmYLLabels =(/0,-1000,-2000,-3000,-4000,-5000/)


res@gsnRightString=""
res@gsnLeftStringFontHeightF=40
res@gsnLeftString="PI (Atlantic)"
map(0) = gsn_csm_contour(wks,a_n,res)   ; Draw contours over a map.
res@gsnLeftString="MH (Atlantic)"
map(1) = gsn_csm_contour(wks,c_n,res)   ; Draw contours over a map.
res@gsnLeftString="LIG (Atlantic)"
map(2) = gsn_csm_contour(wks,d_n,res)   ; Draw contours over a map.
res@gsnLeftString="MIS3 (Atlantic)"
map(3) = gsn_csm_contour(wks,e_n,res)   ; Draw contours over a map.
res@gsnLeftString="LGM (Atlantic)"
map(4) = gsn_csm_contour(wks,b_n,res)   ; Draw contours over a map.
res@gsnLeftString="PI (Indian)"
map(5) = gsn_csm_contour(wks,a_i,res)   ; Draw contours over a map.
res@gsnLeftString="MH (Indian)"
map(6) = gsn_csm_contour(wks,c_i,res)   ; Draw contours over a map.
res@gsnLeftString="LIG (Indian)"
map(7) = gsn_csm_contour(wks,d_i,res)   ; Draw contours over a map.
res@gsnLeftString="MIS3 (Indian)"
map(8) = gsn_csm_contour(wks,e_i,res)   ; Draw contours over a map.
res@gsnLeftString="LGM (Indian)"
map(9) = gsn_csm_contour(wks,b_i,res)   ; Draw contours over a map.
res@gsnLeftString="PI (Pacific)"
map(10) = gsn_csm_contour(wks,a_p,res)   ; Draw contours over a map.
res@gsnLeftString="MH (Pacific)"
map(11) = gsn_csm_contour(wks,c_p,res)   ; Draw contours over a map.
res@gsnLeftString="LIG (Pacific)"
map(12) = gsn_csm_contour(wks,d_p,res)   ; Draw contours over a map.
res@gsnLeftString="MIS3 (Pacific)"
map(13) = gsn_csm_contour(wks,e_p,res)   ; Draw contours over a map.
res@gsnLeftString="LGM (Pacific)"
map(14) = gsn_csm_contour(wks,b_p,res)   ; Draw contours over a map.

pres=True
pres@gsnFrame=False
pres@gsnDraw=True
;pres@gsnMaximize=True
pres@amJust="Topleft"
pres@gsnPanelLabelBar  = True
;pres@lbLabelStride=5
pres@lbBoxLinesOn=False
gsn_panel(wks,map(0:14),(/3,5/),pres)

frame(wks)

end

