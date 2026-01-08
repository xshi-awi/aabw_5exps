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

; Calculate anomalies (exp - pi)
mh_n_anm = c_n
mh_n_anm = c_n - a_n
mh_i_anm = c_i
mh_i_anm = c_i - a_i
mh_p_anm = c_p
mh_p_anm = c_p - a_p

lig_n_anm = d_n
lig_n_anm = d_n - a_n
lig_i_anm = d_i
lig_i_anm = d_i - a_i
lig_p_anm = d_p
lig_p_anm = d_p - a_p

mis_n_anm = e_n
mis_n_anm = e_n - a_n
mis_i_anm = e_i
mis_i_anm = e_i - a_i
mis_p_anm = e_p
mis_p_anm = e_p - a_p

lgm_n_anm = b_n
lgm_n_anm = b_n - a_n
lgm_i_anm = b_i
lgm_i_anm = b_i - a_i
lgm_p_anm = b_p
lgm_p_anm = b_p - a_p

; Set values with absolute value < 5 to 0 (will appear white with the color scale)
; Keep land mask (-999) unchanged
mh_n_anm = where(abs(mh_n_anm).lt.5.and.mh_n_anm.ne.-999, 0, mh_n_anm)
mh_i_anm = where(abs(mh_i_anm).lt.5.and.mh_i_anm.ne.-999, 0, mh_i_anm)
mh_p_anm = where(abs(mh_p_anm).lt.5.and.mh_p_anm.ne.-999, 0, mh_p_anm)
lig_n_anm = where(abs(lig_n_anm).lt.5.and.lig_n_anm.ne.-999, 0, lig_n_anm)
lig_i_anm = where(abs(lig_i_anm).lt.5.and.lig_i_anm.ne.-999, 0, lig_i_anm)
lig_p_anm = where(abs(lig_p_anm).lt.5.and.lig_p_anm.ne.-999, 0, lig_p_anm)
mis_n_anm = where(abs(mis_n_anm).lt.5.and.mis_n_anm.ne.-999, 0, mis_n_anm)
mis_i_anm = where(abs(mis_i_anm).lt.5.and.mis_i_anm.ne.-999, 0, mis_i_anm)
mis_p_anm = where(abs(mis_p_anm).lt.5.and.mis_p_anm.ne.-999, 0, mis_p_anm)
lgm_n_anm = where(abs(lgm_n_anm).lt.5.and.lgm_n_anm.ne.-999, 0, lgm_n_anm)
lgm_i_anm = where(abs(lgm_i_anm).lt.5.and.lgm_i_anm.ne.-999, 0, lgm_i_anm)
lgm_p_anm = where(abs(lgm_p_anm).lt.5.and.lgm_p_anm.ne.-999, 0, lgm_p_anm)

  wks = gsn_open_wks("pdf","age_vertical_anm")
;gsn_define_colormap(wks, "NCV_jaisnd")
gsn_define_colormap(wks, "testcmap")
  map=new(12,graphic)
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
res@lbLabelStride=1
res@cnLevelSelectionMode = "ExplicitLevels"
  res@vpWidthF               = 0.8
  res@vpHeightF              = 0.4
res@gsnYAxisIrregular2Linear=  True
res@cnLevelSelectionMode = "ExplicitLevels" ; explicit contour levels
; Non-uniform levels to accommodate different magnitudes
; Remove values between -5 and 5 from color scale so they appear white
res@cnLevels=(/-1500,-1000,-800,-600,-400,-300,-200,-150,-100,-75,-50,-25,-5,5,25,50,75,100,150,200,300,400,600,800,1000,1500/)
  res@cnMissingValPerimOn     = True             ; turn on perimeter
  res@cnMissingValFillPattern = 0                ; choose a fill pattern
  res@cnMissingValFillColor   = "gray"          ; choose a pattern color
  res@tmYLMode="Explicit"
  res@tmYLValues =(/0,-1000,-2000,-3000,-4000,-5000/)
  res@tmYLLabels =(/0,-1000,-2000,-3000,-4000,-5000/)


res@gsnRightString=""
res@gsnLeftStringFontHeightF=40
res@gsnLeftString="MH-PI (Atlantic)"
map(0) = gsn_csm_contour(wks,mh_n_anm,res)   ; Draw contours over a map.
res@gsnLeftString="LIG-PI (Atlantic)"
map(1) = gsn_csm_contour(wks,lig_n_anm,res)   ; Draw contours over a map.
res@gsnLeftString="MIS3-PI (Atlantic)"
map(2) = gsn_csm_contour(wks,mis_n_anm,res)   ; Draw contours over a map.
res@gsnLeftString="LGM-PI (Atlantic)"
map(3) = gsn_csm_contour(wks,lgm_n_anm,res)   ; Draw contours over a map.
res@gsnLeftString="MH-PI (Indian)"
map(4) = gsn_csm_contour(wks,mh_i_anm,res)   ; Draw contours over a map.
res@gsnLeftString="LIG-PI (Indian)"
map(5) = gsn_csm_contour(wks,lig_i_anm,res)   ; Draw contours over a map.
res@gsnLeftString="MIS3-PI (Indian)"
map(6) = gsn_csm_contour(wks,mis_i_anm,res)   ; Draw contours over a map.
res@gsnLeftString="LGM-PI (Indian)"
map(7) = gsn_csm_contour(wks,lgm_i_anm,res)   ; Draw contours over a map.
res@gsnLeftString="MH-PI (Pacific)"
map(8) = gsn_csm_contour(wks,mh_p_anm,res)   ; Draw contours over a map.
res@gsnLeftString="LIG-PI (Pacific)"
map(9) = gsn_csm_contour(wks,lig_p_anm,res)   ; Draw contours over a map.
res@gsnLeftString="MIS3-PI (Pacific)"
map(10) = gsn_csm_contour(wks,mis_p_anm,res)   ; Draw contours over a map.
res@gsnLeftString="LGM-PI (Pacific)"
map(11) = gsn_csm_contour(wks,lgm_p_anm,res)   ; Draw contours over a map.

pres=True
pres@gsnFrame=False
pres@gsnDraw=True
;pres@gsnMaximize=True
pres@amJust="Topleft"
pres@gsnPanelLabelBar  = True
; Symmetric label display - show selected symmetric values
; 26 levels: -1500,-1000,-800,-600,-400,-300,-200,-150,-100,-75,-50,-25,-5,5,25,50,75,100,150,200,300,400,600,800,1000,1500
pres@lbLabelStrings=(/"-1500","","-800","","-400","","","-150","","-75","","-25","","","25","","75","","150","","","400","","800","","1500"/)
pres@lbLabelAutoStride=False
pres@lbBoxLinesOn=False
gsn_panel(wks,map(0:11),(/3,4/),pres)

frame(wks)

end

