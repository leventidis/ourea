#!/bin/sh

# Script that evaluates the synthetic dataset by simulating streaming

output_dir_base="../streaming_data/synthetic_data/variable_inj_region_length/12_48/"
input_time_series_base="../streaming_data/synthetic_data/variable_inj_region_length/12_48/"

# Simulate streaming parameters
num_points_in_history=2016
future_window_size=1008
streaming_num_points_step=6
confidence_interval_width=0.98
aggregation_model="aggregate_prophet_model"
window_size=6H
num_points_per_window=6
point_frequency=1H
kernel="epa"
bandwidth=86400
seed=1
bandwidth_selection="custom"

# Evaluation Parameters


for sub_dir_path in $input_time_series_base*
do
    input_time_series_df=$sub_dir_path/df.pickle
    output_dir=$sub_dir_path/evaluation/future_prediction_fixed_bandwidth/
    regions_df=$sub_dir_path/regions_df.pickle

    # Simulate streaming and perform the evaluation
    python ../simulate_streaming.py --output_dir $output_dir \
    --input_time_series_df $input_time_series_df --num_points_in_history $num_points_in_history \
    --streaming_num_points_step $streaming_num_points_step --confidence_interval_width $confidence_interval_width \
    --aggregation_model $aggregation_model --window_size $window_size --num_points_per_window $num_points_per_window \
    --point_frequency $point_frequency --kernel $kernel --bandwidth $bandwidth --seed $seed --bandwidth_selection $bandwidth_selection \
    --future_window_size $future_window_size --do_not_save_streaming_dataframes --evaluate_raised_alerts --regions_df $regions_df

    break
done