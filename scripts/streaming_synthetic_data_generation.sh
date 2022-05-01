#!/bin/sh

# Script that generates synthetic time series with injected outlier regions

injection_mode="all"
output_dir_base="../streaming_data/synthetic_data/all_outlier_types/variable_inj_region_length/12_48/"
num_datasets=10

# Synthetic time series parameters
window_size=6H
num_points_per_window=6
point_frequency=1H
confidence_interval_width=0.98
white_noise_std=0.1
inj_boost_percent=50
inj_gap_percent=2
inj_num_regions=10
inj_region_length=24
inj_min_length=12
inj_max_length=48
synthetic_num_periods=60
bandwidth=86400
kernel="epa"

mkdir -p $output_dir_base

for i in `seq 1 $num_datasets`
do
    seed=$i
    echo "Seed:" $seed

    output_dir=${output_dir_base}seed_${seed}/

    # Create the synthetic time series with the injected outliers for the specified seed
    python ../synthetic_data_generator.py -o $output_dir \
    --window_size $window_size --num_points_per_window $num_points_per_window --point_frequency $point_frequency \
    --confidence_interval_width $confidence_interval_width --white_noise_std $white_noise_std -im $injection_mode \
    --kernel $kernel --seed $seed --synthetic_num_periods $synthetic_num_periods --inj_boost_percent $inj_boost_percent \
    --bandwidth_selection custom --bandwidth $bandwidth --inj_gap_percent $inj_gap_percent --inj_num_regions $inj_num_regions --inj_region_length $inj_region_length \
    --variable_inj_region_length $inj_min_length $inj_max_length --aggregation_model aggregate_prophet_model    
done