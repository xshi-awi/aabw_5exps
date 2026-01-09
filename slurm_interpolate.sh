#!/bin/bash
#
# Batch interpolation script for surface density tendency files
# Runs directly without SLURM submission
# Processes all 5 experiments × multiple variables
#

set -e

# Create log directory
mkdir -p logs

echo "========================================"
echo "Batch Interpolation Job"
echo "========================================"
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo ""

# Change to working directory
cd /work/ba0989/a270064/bb1029/aabw_5exps

# Set number of parallel jobs (adjust based on available cores)
MAX_PARALLEL=8
echo "Max parallel jobs: $MAX_PARALLEL"
echo ""

# Load required modules (if needed)
# module load python/3.10  # Uncomment if needed

# Define experiments
experiments=("pi" "mh" "lig" "lgm" "mis")

# Variables to interpolate (34 total variables from the NetCDF file)
# To get this list: ncdump -h pi/surface_density_tendency_pi.nc | grep "(time, nod2)" | awk '{print $2}' | cut -d'(' -f1
variables=(
    "tos"
    "sos"
    "SW_flux"
    "LW_flux"
    "LH_flux"
    "SH_flux"
    "evs"
    "prlq"
    "prsn"
    "friver"
    "fsitherm"
    "wfo"
    "sfdsi"
    "mass_rhs_sum_surface_exchange_flux_sum_rain_and_ice"
    "mass_rhs_sum_surface_exchange_flux_sum_snow"
    "mass_rhs_sum_surface_exchange_flux_sum_evaporation"
    "mass_rhs_sum_surface_exchange_flux_sum_rivers"
    "mass_rhs_sum_surface_exchange_flux_sum_icebergs"
    "mass_rhs_sum_surface_exchange_flux_sum_sea_ice_melt"
    "mass_rhs_sum_surface_exchange_flux_sum_virtual_precip_restoring"
    "surface_exchange_flux_nonadvective_total"
    "surface_exchange_flux_nonadvective_sw"
    "surface_exchange_flux_nonadvective_lw"
    "surface_exchange_flux_nonadvective_lh"
    "surface_exchange_flux_nonadvective_sh"
    "salt_rhs_sum_surface_exchange_flux_sum_nonadvective"
    "salt_rhs_sum_surface_exchange_flux_sum_advective"
    "salt_rhs_sum_surface_ocean_flux_advective_negative_rhs_sum_rain_and_ice"
    "salt_rhs_sum_surface_ocean_flux_advective_negative_rhs_sum_snow"
    "salt_rhs_sum_surface_ocean_flux_advective_negative_rhs_sum_evaporation"
    "salt_rhs_sum_surface_ocean_flux_advective_negative_rhs_sum_rivers"
    "salt_rhs_sum_surface_ocean_flux_advective_negative_rhs_sum_icebergs"
    "salt_rhs_sum_surface_ocean_flux_advective_negative_rhs_sum_sea_ice_melt"
    "salt_rhs_sum_surface_ocean_flux_advective_negative_rhs_sum_virtual_precip_restoring"
)

total_jobs=$((${#experiments[@]} * ${#variables[@]}))
echo "Total interpolation jobs: $total_jobs"
echo ""

# Function to interpolate one variable
interpolate_var() {
    local exp=$1
    local var=$2
    local input_file="${exp}/surface_density_tendency_${exp}.nc"
    local output_file="${exp}/${var}_reg.nc"

    if [ ! -f "$input_file" ]; then
        echo "ERROR: $input_file not found"
        return 1
    fi

    # Skip if output already exists
    if [ -f "$output_file" ]; then
        echo "SKIP: $exp/$var (already exists)"
        return 0
    fi

    echo "START: $exp/$var ($(date '+%H:%M:%S'))"

    if interp "$input_file" "$var" "$output_file" > /dev/null 2>&1; then
        size=$(ls -lh "$output_file" | awk '{print $5}')
        echo "DONE: $exp/$var ($size) ($(date '+%H:%M:%S'))"
        return 0
    else
        echo "FAIL: $exp/$var"
        return 1
    fi
}

export -f interpolate_var

# Process experiments sequentially, variables in parallel within each experiment
for exp in "${experiments[@]}"; do
    echo "========================================"
    echo "Processing experiment: $exp"
    echo "========================================"

    input_file="${exp}/surface_density_tendency_${exp}.nc"
    if [ ! -f "$input_file" ]; then
        echo "WARNING: $input_file not found, skipping..."
        continue
    fi

    # Process all variables for this experiment
    for var in "${variables[@]}"; do
        interpolate_var "$exp" "$var" &

        # Limit concurrent jobs to MAX_PARALLEL
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
            sleep 1
        done
    done

    # Wait for all background jobs of this experiment to finish
    wait

    # Summary for this experiment
    count=$(ls ${exp}/*_reg.nc 2>/dev/null | wc -l)
    echo "Completed interpolation: $exp ($count files)"
    echo ""

    # Merge all variables into one file
    echo "========================================"
    echo "Merging variables for: $exp"
    echo "========================================"

    merged_file="${exp}/surface_density_tendency_${exp}_reg.nc"

    # Check if all variables are interpolated
    if [ $count -eq ${#variables[@]} ]; then
        echo "Merging $count variables into $merged_file"
        echo "Start time: $(date '+%H:%M:%S')"

        # Use cdo merge to combine all variables
        cdo -O merge ${exp}/*_reg.nc "$merged_file" 2>&1

        if [ $? -eq 0 ]; then
            merged_size=$(ls -lh "$merged_file" | awk '{print $5}')
            echo "✓ Merge successful: $merged_file ($merged_size)"
            echo "End time: $(date '+%H:%M:%S')"

            # Remove individual variable files (but keep the merged file)
            echo "Cleaning up individual variable files..."
            deleted_count=0
            for var in "${variables[@]}"; do
                var_file="${exp}/${var}_reg.nc"
                if [ -f "$var_file" ]; then
                    rm -f "$var_file"
                    ((deleted_count++))
                fi
            done

            echo "✓ Cleanup complete: deleted $deleted_count individual files"
            echo "✓ Final merged file: $merged_file ($merged_size)"
            echo ""
        else
            echo "✗ Merge failed for $exp"
            echo "  Keeping individual variable files"
            echo ""
        fi
    else
        echo "WARNING: Only $count/${#variables[@]} variables found, skipping merge"
        echo ""
    fi
done

echo "========================================"
echo "All interpolations and merging completed!"
echo "========================================"
echo "End time: $(date)"
echo ""

# Final summary
echo "Final Summary:"
echo "----------------------------------------"
total_size_bytes=0
for exp in "${experiments[@]}"; do
    if [ -d "$exp" ]; then
        merged_file="${exp}/surface_density_tendency_${exp}_reg.nc"
        if [ -f "$merged_file" ]; then
            size=$(ls -lh "$merged_file" | awk '{print $5}')
            size_bytes=$(ls -l "$merged_file" | awk '{print $5}')
            total_size_bytes=$((total_size_bytes + size_bytes))
            echo "  ✓ $exp: $merged_file ($size)"
        else
            # Check for individual files
            count=$(ls ${exp}/*_reg.nc 2>/dev/null | wc -l)
            if [ $count -gt 0 ]; then
                echo "  ⚠ $exp: $count individual files (merge failed or incomplete)"
            else
                echo "  ✗ $exp: No output files found"
            fi
        fi
    fi
done

echo "----------------------------------------"
# Convert total size to human readable
total_size_gb=$(echo "scale=2; $total_size_bytes / 1024 / 1024 / 1024" | bc)
total_size_mb=$(echo "scale=1; $total_size_bytes / 1024 / 1024" | bc)
echo "Total merged output: ${total_size_mb} MB (${total_size_gb} GB)"
echo ""

# List final output files
echo "Final output files:"
ls -lh */surface_density_tendency_*_reg.nc 2>/dev/null || echo "  No merged files found"
