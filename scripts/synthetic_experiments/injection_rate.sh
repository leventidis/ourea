#!/bin/sh

# Generate multipe synthetic datasets and inject them with all the anomaly types
cd ../../

num_datasets=10
out_dir="synthetic_data/synthetic_experiments/variable_injection_rate/"
window_size="6H"
num_points_per_window=6
point_frequency="1H"
confidence_interval_width=0.98
white_noise_std=0.10
kernel="epa"
synthetic_num_periods=60
bandwidth=86400
inj_num_regions=10
inj_region_length=24


seed_start=1
injection_rate_step_size=0.05
injection_rate_start=0.40
injection_rate_end=1.0

####### ---------- MDI Related Parameters ---------- #######
mdi_method='gaussian_cov'
mdi_proposals='dense'
mdi_mode='TS'
mdi_extint_min_len=12
mdi_extint_max_len=48
mdi_td_dim=0
mdi_td_lag=0

for injection_rate in $(seq $injection_rate_start $injection_rate_step_size $injection_rate_end);
do

    for seed in $(seq $seed_start $num_datasets);
    do

        # # Data Generation
        # python synthetic_data_generator.py -o $out_dir/rate_$injection_rate/seed$seed/ \
        # --window_size $window_size --num_points_per_window $num_points_per_window \
        # --point_frequency $point_frequency --confidence_interval_width $confidence_interval_width \
        # --injection_mode all --kernel $kernel --synthetic_num_periods $synthetic_num_periods --bandwidth $bandwidth \
        # --inj_num_regions $inj_num_regions --outlier_injection_rate $injection_rate \
        # --seed $seed --white_noise_std $white_noise_std --aggregation_model aggregate_prophet_model

        # # Fixed Bandwidth
        # python simulate_streaming.py --output_dir $out_dir/rate_$injection_rate/seed$seed/evaluation/fixed_bandwidth/ \
        # --input_time_series_df $out_dir/rate_$injection_rate/seed$seed/df.pickle \
        # --regions_df $out_dir/rate_$injection_rate/seed$seed/regions_df.pickle \
        # --num_points_in_history 10080 --streaming_num_points_step 4 --confidence_interval_width $confidence_interval_width \
        # --aggregation_model aggregate_prophet_model \
        # --window_size $window_size --num_points_per_window $num_points_per_window --point_frequency $point_frequency \
        # --kernel $kernel --bandwidth $bandwidth --seed 1 --bandwidth_selection custom

        # # ISJ Bandwidth
        # python simulate_streaming.py --output_dir $out_dir/rate_$injection_rate/seed$seed/evaluation/isj_bandwidth/ \
        # --input_time_series_df $out_dir/rate_$injection_rate/seed$seed/df.pickle \
        # --regions_df $out_dir/rate_$injection_rate/seed$seed/regions_df.pickle \
        # --num_points_in_history 10080 --streaming_num_points_step 4 --confidence_interval_width $confidence_interval_width \
        # --aggregation_model aggregate_prophet_model \
        # --window_size $window_size --num_points_per_window $num_points_per_window --point_frequency $point_frequency \
        # --kernel $kernel --bandwidth $bandwidth --seed 1 --bandwidth_selection ISJ

        # # ISJ Bandwidth with Adjusted Local Minima
        # python simulate_streaming.py --output_dir $out_dir/rate_$injection_rate/seed$seed/evaluation/isj_bandwidth_adj_local_min/ \
        # --input_time_series_df $out_dir/rate_$injection_rate/seed$seed/df.pickle \
        # --regions_df $out_dir/rate_$injection_rate/seed$seed/regions_df.pickle \
        # --num_points_in_history 10080 --streaming_num_points_step 4 --confidence_interval_width $confidence_interval_width \
        # --aggregation_model aggregate_prophet_model \
        # --window_size $window_size --num_points_per_window $num_points_per_window --point_frequency $point_frequency \
        # --kernel $kernel --bandwidth $bandwidth --seed 1 --bandwidth_selection ISJ --adjusted_local_minima

        # MDI Anomalous Regions
        python libmaxdiv/get_mdi_regions.py --output_dir $out_dir/rate_$injection_rate/seed$seed/evaluation/mdi/ \
        --input_time_series_df $out_dir/rate_$injection_rate/seed$seed/df.pickle \
        --method $mdi_method --num_intervals $inj_num_regions --proposals $mdi_proposals --mode $mdi_mode \
        --extint_min_len $mdi_extint_min_len --extint_max_len $mdi_extint_max_len --td_dim $mdi_td_dim --td_lag $mdi_td_lag

    done

done