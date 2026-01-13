#!/bin/bash
#
# Merge 4-segment WMT 100-year timeseries files using CDO
# Each experiment has 4 regions, each region has 4 time segments (0-25, 25-50, 50-75, 75-100 years)
#

set -e  # Exit on error

# Check if CDO is available
if ! command -v cdo &> /dev/null; then
    echo "ERROR: CDO not found. Please load CDO module first."
    echo "Try: module load cdo"
    exit 1
fi

# Define experiments and their regions
EXPERIMENTS=("pi" "mh" "lig" "lgm" "mis")
REGIONS=("southern_ocean" "ross_sea" "weddell_sea" "adelie")

# Output summary
echo "=========================================="
echo "WMT Timeseries Merge Script"
echo "=========================================="
echo "Experiments: ${EXPERIMENTS[@]}"
echo "Regions: ${REGIONS[@]}"
echo "Tool: CDO mergetime"
echo "=========================================="
echo ""

# Loop through each experiment
for EXP in "${EXPERIMENTS[@]}"; do
    echo "Processing experiment: ${EXP}"

    WMT_DIR="${EXP}/wmt_results"

    # Check if directory exists
    if [ ! -d "${WMT_DIR}" ]; then
        echo "  WARNING: Directory ${WMT_DIR} not found, skipping..."
        continue
    fi

    # Loop through each region
    for REGION in "${REGIONS[@]}"; do
        echo "  Merging region: ${REGION}"

        # Define input files (4 segments)
        INPUT_FILES=(
            "${WMT_DIR}/wmt_${REGION}_y000_025_${EXP}.nc"
            "${WMT_DIR}/wmt_${REGION}_y025_050_${EXP}.nc"
            "${WMT_DIR}/wmt_${REGION}_y050_075_${EXP}.nc"
            "${WMT_DIR}/wmt_${REGION}_y075_100_${EXP}.nc"
        )

        # Define output file
        OUTPUT_FILE="${WMT_DIR}/wmt_${REGION}_100years_${EXP}.nc"

        # Check if all input files exist
        MISSING=0
        for FILE in "${INPUT_FILES[@]}"; do
            if [ ! -f "${FILE}" ]; then
                echo "    WARNING: Missing file ${FILE}"
                MISSING=1
            fi
        done

        if [ ${MISSING} -eq 1 ]; then
            echo "    SKIPPED: Not all 4 segments found for ${REGION}"
            continue
        fi

        # Check if output already exists
        if [ -f "${OUTPUT_FILE}" ]; then
            echo "    EXISTS: ${OUTPUT_FILE} already exists, skipping..."
            continue
        fi

        # Merge using CDO
        echo "    Running: cdo mergetime ${INPUT_FILES[@]} ${OUTPUT_FILE}"
        cdo -O mergetime "${INPUT_FILES[@]}" "${OUTPUT_FILE}"

        # Check success and report file size
        if [ -f "${OUTPUT_FILE}" ]; then
            SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
            echo "    SUCCESS: Created ${OUTPUT_FILE} (${SIZE})"
        else
            echo "    ERROR: Failed to create ${OUTPUT_FILE}"
        fi

    done

    echo ""
done

echo "=========================================="
echo "Merge Complete!"
echo "=========================================="
echo ""
echo "Summary of merged files:"
for EXP in "${EXPERIMENTS[@]}"; do
    if [ -d "${EXP}/wmt_results" ]; then
        echo "${EXP}:"
        ls -lh ${EXP}/wmt_results/wmt_*_100years_*.nc 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    fi
done
echo ""
