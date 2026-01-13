
for exp in pi mh lig lgm mis
do
for code in 92 95 111 120
do
#	cdo selcode,${code} ${exp}/echam_mergetime.nc ${exp}/100years/${code}_mergetime_echam.nc
rm ${exp}/100years/echam_*.nc
done
for var in a_ice prec evap fw runoff sss sst snow
do
	#cdo mergetime ${exp}/100years/${var}.fesom.*.nc ${exp}/100years/${var}_mergetime.nc 
	rm ${exp}/100years/${var}.fesom.*.nc 
done
done

exit


for exp in mis
do
                dir=/home/a/a270064/bb1029/production/${exp}_age/outdata/echam
        cdo mergetime ${dir}/${exp}_age_25*echam ${exp}/echam_mt
        cdo -f nc copy ${exp}/echam_mt ${exp}/echam_mergetime.nc
	rm ${exp}/echam_mt
done

for exp in pi mh lig #lgm mis
do
		dir=/home/a/a270064/bb1029/production/${exp}_age/outdata/echam
	cdo mergetime ${dir}/${exp}_age_23*echam ${exp}/echam_mt
        cdo -f nc copy ${exp}/echam_mt ${exp}/echam_mergetime.nc
	rm ${exp}/echam_mt
done

for exp in lgm 
do
                dir=/home/a/a270064/bb1029/production/${exp}_age/outdata/echam
        cdo mergetime ${dir}/${exp}_age_24*echam ${exp}/echam_mt
        cdo -f nc copy ${exp}/echam_mt ${exp}/echam_mergetime.nc
        rm ${exp}/echam_mt
done


