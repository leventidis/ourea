python simulate_streaming.py --output_dir streaming_data/all_outlier_types/ISJ/ \
--input_time_series_df streaming_data/all_outlier_types/df_full.pickle \
--num_points_in_history 2016 --streaming_num_points_step 6 --confidence_interval_width 0.98 \
--aggregation_model aggregate_prophet_model \
--window_size 6H --num_points_per_window 6 --point_frequency 1H --kernel epa --bandwidth 86400 --seed 1 --bandwidth_selection ISJ

python simulate_streaming.py --output_dir streaming_data/all_outlier_types/fixed_bandwidth/ \
--input_time_series_df streaming_data/all_outlier_types/df_full.pickle \
--num_points_in_history 2016 --streaming_num_points_step 6 --confidence_interval_width 0.98 \
--aggregation_model aggregate_prophet_model \
--window_size 6H --num_points_per_window 6 --point_frequency 1H --kernel epa --bandwidth 86400 --seed 1 --bandwidth_selection custom


# With Future Prediction fixed_bandwidth
python simulate_streaming.py --output_dir streaming_data/all_outlier_types/future_prediction_fixed_bandwidth/ \
--input_time_series_df streaming_data/all_outlier_types/df_full.pickle \
--num_points_in_history 2016 --streaming_num_points_step 6 --confidence_interval_width 0.98 \
--aggregation_model aggregate_prophet_model \
--window_size 6H --num_points_per_window 6 --point_frequency 1H \
--kernel epa --bandwidth 86400 --seed 1 --bandwidth_selection custom --future_window_size 168