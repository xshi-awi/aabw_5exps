 for var in sigma_tendency_sw sigma_tendency_lh sigma_tendency_sh sigma_tendency_total_heat sigma_tendency_prec sigma_tendency_evap sigma_tendency_runoff sigma_tendency_snow sigma_tendency_seaice sigma_tendency_total_freshwater sigma_tendency_total
 do
	 echo ${var}
	 ln -s fesom_surface_tendencies.nc ${var}.fesom.2000.nc
	 lgm_interp ${var} 2000 ${var}_reg.nc
	 rm ${var}.fesom.2000.nc
 done
