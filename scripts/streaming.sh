#!/bin/sh


# Script that alert detection on a time series that progressively gets longer to simulate a streaming scenario

out_dir_base="../streaming_data/high_residual/consistent/"
input_dir="../streaming_data/high_residual/consistent/timeseries/"


# Alert detection parameters
window_size="6H"
num_points_per_window=6
point_frequency="1H"
confidence_interval_width=0.95
clustering_mode="KDE"
bandwidth=86400
seed=1

for filepath in ${input_dir}*.pickle;
do
    filename=$(basename $filepath .pickle)
    
    out_dir=${out_dir_base}outputs/$filename/
    mkdir -p $out_dir

    #Run alert detection on the given input time series
    python ../synthetic_data_generator.py -o $out_dir --input_time_series_df $filepath \
    --window_size $window_size --num_points_per_window $num_points_per_window \
    --point_frequency $point_frequency --confidence_interval_width $confidence_interval_width \
    --clustering_mode $clustering_mode --seed $seed --bandwidth $bandwidth \
    aggregate_prophet_model

done