#!/bin/sh

# Generate multipe synthetic datasets and inject them with all the anomaly types
cd ../../

num_datasets=10
input_dir="synthetic_data/mdi_synthetic_data/meanshift5/"
out_dir="synthetic_data/mdi_synthetic_data/meanshift5/"
window_size="16H"
num_points_per_window=16
point_frequency="1H"
confidence_interval_width=0.90
kernel="gaussian"
bandwidth=86400


for dir in $input_dir*/;
do

    python synthetic_data_generator.py -o $dir --window_size $window_size --num_points_per_window  $num_points_per_window \
    --point_frequency $point_frequency --confidence_interval_width $confidence_interval_width --bandwidth $bandwidth \
    --kernel $kernel --input_time_series_df ${dir}df.pickle --input_regions_df ${dir}regions_df.pickle aggregate_ma_model

done