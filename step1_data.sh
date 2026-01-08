
for exp in pi mh lig lgm mis
do
	for var in a_ice evap fh fw MLD1 MLD2 prec runoff  snow sss sst
	do
		interp ${exp}/${var}_clim.nc ${var} ${exp}/${var}_reg.nc
	done
done

exit


for exp in mis
do
        mkdir -p ${exp}
                dir=/home/a/a270064/bb1029/production/${exp}_age/outdata/fesom
        for var in a_ice evap fh fw MLD1 MLD2 prec runoff  snow sss sst #temp salt
        do
                cdo ensmean ${dir}/${var}.fesom.25*.nc ${exp}/${var}_clim.nc
        done
        cp ${dir}/age.fesom.249901.01.nc ${exp}/age.nc
        interp ${exp}/age.nc age ${exp}/age_reg.nc
        interp ${dir}/salt.fesom.200101.01.nc salt ${exp}/mask.nc
                dir=/home/a/a270064/bb1029/production/${exp}_age/outdata/echam
        cdo mergetime ${dir}/${exp}_age_25*echam ${exp}/echam_mt
        cdo -f nc copy -ymonmean ${exp}/echam_mt ${exp}/echam_clim.nc
        rm ${exp}/echam_mt
done

for exp in pi mh lig #lgm mis
do
	mkdir -p ${exp}
		dir=/home/a/a270064/bb1029/production/${exp}_age/outdata/fesom
	for var in a_ice evap fh fw MLD1 MLD2 prec runoff  snow sss sst temp salt
	do
		cdo ensmean ${dir}/${var}.fesom.23*.nc ${exp}/${var}_clim.nc
	done
	cp ${dir}/age.fesom.239901.01.nc ${exp}/age.nc
	interp ${exp}/age.nc age ${exp}/age_reg.nc
	interp ${dir}/salt.fesom.200101.01.nc salt ${exp}/mask.nc 
		dir=/home/a/a270064/bb1029/production/${exp}_age/outdata/echam
	cdo mergetime ${dir}/${exp}_age_23*echam ${exp}/echam_mt
        cdo -f nc copy -ymonmean ${exp}/echam_mt ${exp}/echam_clim.nc	
	rm ${exp}/echam_mt
done

for exp in lgm 
do
        mkdir -p ${exp}
                dir=/home/a/a270064/bb1029/production/${exp}_age/outdata/fesom
        for var in a_ice evap fh fw MLD1 MLD2 prec runoff  snow sss sst temp salt
        do
                cdo ensmean ${dir}/${var}.fesom.24*.nc ${exp}/${var}_clim.nc
        done
	cp ${dir}/age.fesom.249901.01.nc ${exp}/age.nc
	interp ${exp}/age.nc age ${exp}/age_reg.nc
	interp ${dir}/salt.fesom.200101.01.nc salt ${exp}/mask.nc 
                dir=/home/a/a270064/bb1029/production/${exp}_age/outdata/echam
        cdo mergetime ${dir}/${exp}_age_24*echam ${exp}/echam_mt
        cdo -f nc copy -ymonmean ${exp}/echam_mt ${exp}/echam_clim.nc
        rm ${exp}/echam_mt
done


