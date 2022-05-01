#!/bin/sh

# Script that evaluates the synthetic dataset by simulating streaming
cd ..

output_dir_base="streaming_data/synthetic_data/variable_inj_region_length/12_48/"
input_time_series_base="streaming_data/synthetic_data/variable_inj_region_length/12_48/"

# Simulate streaming parameters
num_points_in_history=2016
future_window_size=1008
streaming_num_points_step=6
seed=1

# MDI Parameters
mdi_method='gaussian_cov'
mdi_proposals='dense'
mdi_mode='TS'
mdi_extint_min_len=12
mdi_extint_max_len=48
mdi_td_dim=0
mdi_td_lag=0

for sub_dir_path in $input_time_series_base*
do
    input_time_series_df=$sub_dir_path/df.pickle
    output_dir=$sub_dir_path/evaluation/mdi/
    regions_df=$sub_dir_path/regions_df.pickle

    # Simulate streaming and perform the evaluation
    python libmaxdiv/mdi_simulate_streaming.py --output_dir $output_dir \
    --input_time_series_df $input_time_series_df --regions_df $regions_df \
    --num_points_in_history $num_points_in_history --streaming_num_points_step $streaming_num_points_step \
    --method $mdi_method --num_intervals 10 --proposals $mdi_proposals --mode $mdi_mode \
    --extint_min_len $mdi_extint_min_len --extint_max_len $mdi_extint_max_len --td_dim $mdi_td_dim --td_lag $mdi_td_lag
done