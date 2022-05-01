import pandas as pd
import numpy as np

import math
import argparse
import json
from pathlib import Path
from tqdm import tqdm

import utils
from timeit import default_timer as timer

def raw_model(cur_df, forecast_raw, args):
    '''
    Run the raw model

    Check if forecast_raw covers the timestamps in cur_df, if it doesn't retrain the model and update forecast_raw

    Returns the updated cur_df, forecast_raw and runtime
    '''
    start = timer()

    if len(forecast_raw.index) < len(cur_df.index):
        # Update the raw model to fit further in the future
        model_raw, forecast_raw = utils.outlier_detection.raw_data_fit(
            df=cur_df, confidence_interval_width=args.confidence_interval_width,
            point_frequency=args.point_frequency, future_window_size=args.future_window_size
        )
    
    cur_df = utils.outlier_detection.update_df_with_raw_model_fit(df=cur_df, forecast=forecast_raw)
    raw_fit_time = timer() - start

    return cur_df, forecast_raw, raw_fit_time

def agg_models(cur_df, forecasts_agg, args):
    '''
    Run the aggregate models

    Check if aggregate forecasts in `forecasts_agg` cover the timestamps in df_agg, if not then retrain the model and update `forecasts_agg`

    Returns the updated df_ag, forecasts_agg and runtime
    '''
    start = timer()
    df_agg = utils.aggregation.get_agg_df(cur_df, args.window_size)

    if (len(forecasts_agg[0].index) < len(df_agg.index)) or (len(forecasts_agg[1].index) < len(df_agg.index)):
        df_agg, models_agg, forecasts_agg = utils.aggregation.get_aggregate_model_fit(df_agg=df_agg, args=args)
    
    # Ensure that the forecast_agg dataframes are at least as large as df_agg
    assert len(forecasts_agg[0].index) >= len(df_agg.index), "Aggregate dataframe is not covered by current forecast"
    assert len(forecasts_agg[1].index) >= len(df_agg.index), "Aggregate dataframe is not covered by current forecast"
    
    # Update df_agg from the forecasts accordingly
    df_agg = utils.aggregation.update_agg_df_with_agg_model_fit(
        df_agg=df_agg,
        forecast_num_outliers=forecasts_agg[0].head(len(df_agg.index)),
        forecast_residual_sum=forecasts_agg[1].head(len(df_agg.index))
    )
    agg_fit_time = timer() - start

    return df_agg, forecasts_agg, agg_fit_time

def main(args):
    
    # Read the entire input times series
    df_full = pd.read_pickle(args.input_time_series_df)
    df_full = df_full.sort_values(by='timestamp')

    # In case args.future_window_size is specified store the forecasts from previous runs for re-use
    forecast_raw = pd.DataFrame()
    forecasts_agg = [pd.DataFrame(), pd.DataFrame()]

    # Initialize the AlertEvaluation Class
    alert_evaluation = utils.alerts.AlertEvaluation(df_full, args.num_top_k_alerts, args.regions_df)

    # Simulate streaming by extending the data in the dataframe iteratively (keep track of runtime for each component)
    for cur_size in tqdm(range(args.num_points_in_history, len(df_full.index)+1, args.streaming_num_points_step)):
        cur_df = df_full.head(cur_size).copy()

        # Fit model on raw data
        cur_df, forecast_raw, raw_fit_time = raw_model(cur_df, forecast_raw, args)

        # Perform Outlier Detection in Aggregate views of the raw data
        df_agg, forecasts_agg, agg_fit_time = agg_models(cur_df, forecasts_agg, args)

        # Identify outliers across all views (raw + aggregation) and perform voting 
        # (assign each timestamp to count of views that marked it as an outlier)
        cur_df = cur_df.copy()[:math.floor(len(cur_df.index) / len(args.window_size)) * len(df_agg.index)]
        cur_df = utils.voting.add_voting_to_df(df=cur_df, df_agg=df_agg, period=args.num_points_per_window, freq=args.point_frequency)

        # TODO: Only compute KDE in the frozen time range (i.e., time range where new data does not have an impact to kde)
        start = timer()
        cur_df, density_df, clusters_df, selected_bandwidth = utils.clustering.perform_clustering(
            df=cur_df, kernel=args.kernel, bandwidth_selection=args.bandwidth_selection,
            bandwidth=args.bandwidth, adjusted_local_minima=args.adjusted_local_minima
        )
        kde_time = timer() - start

        print("Elapsed Time:", kde_time)

        if args.evaluate_raised_alerts:
            alert_evaluation.df_list.append(cur_df)
            alert_evaluation.density_df_list.append(density_df)
            alert_evaluation.clusters_df_list.append(clusters_df)

        if not args.do_not_save_streaming_dataframes:
            # Save dataframes
            out_dir = args.output_dir + 'outputs/size_' + str(cur_size) + '/' 
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            cur_df.to_pickle(out_dir + 'df.pickle')
            density_df.to_pickle(out_dir + 'density_df.pickle')
            clusters_df.to_pickle(out_dir + 'clusters_df.pickle')

            # Save the runtimes into a json file
            runtime_dict = {'raw_fit_time': raw_fit_time, 'agg_fit_time': agg_fit_time, 'kde_time': kde_time, 'bandwidth': selected_bandwidth}
            with open(out_dir + 'stats.json', 'w') as fp:
                json.dump(runtime_dict, fp, indent=4)

    # Perform the evaluation over the populated `alert_evaluation` Object and save the alerts_df
    if args.evaluate_raised_alerts:
        alert_evaluation.set_dfs_to_indices(args.num_points_in_history, args.streaming_num_points_step)
        alert_evaluation.alerts_df = alert_evaluation.get_alerts_df()
        alert_evaluation.alerts_df = alert_evaluation.get_true_regions_coverage()
        alert_evaluation.alerts_df = alert_evaluation.update_alerts_df_with_eval_scores()
        print(alert_evaluation.alerts_df)
        alert_evaluation.alerts_df.to_pickle(args.output_dir + 'alerts_df.pickle')



if __name__ == "__main__":
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Evaluating detected outliers by in a synthetic time series by injecting them')

    parser.add_argument('-o', '--output_dir', metavar='output_dir', required=True,
    help='Path to the output directory where output files and figures are stored. Path must terminate with backslash "\\"')

    parser.add_argument('--input_time_series_df', help='The input time series data used to simulate its streaming.', required=True)

    parser.add_argument('--num_points_in_history', default=100, type=int,
        help='The number of the first timestamps in the timeseries that are used as the base (i.e., history) timestamps \
        after the specified history will be processed by simulating a streaming scenario. \
        Ideally this value should be the same as the largest aggregation size specified by `--num_points_per_window`.'
    )

    parser.add_argument('--streaming_num_points_step', default=10, type=int,
        help='The number of data points added between consecutive re-computations of the anomalies \
            (i.e., every how many data points we examine the detectied anomalies with respect to our current the model)'
    )

    parser.add_argument('--future_window_size', type=int,
        help='If specified the prophet model is used to predict into the future. \
        The value is specified using an integer denoting the number of data points we want to predict ahead'
    )

    parser.add_argument('--confidence_interval_width', default=0.95, type=float,
        help='Specifies the confidence interval used by the prophet model (both for the raw and aggregate views)'
    )

    parser.add_argument('--seed', metavar='seed', type=int, default=1,
        help='The seed is used to ensure model fit by Prophet is deterministic'
    )

    # -------------------------- Aggregation Parameters -------------------------- #

    parser.add_argument('-agg_m', '--aggregation_model', choices=['aggregate_ma_model', 'aggregate_prophet_model'],
        default="aggregate_prophet_model", help='Aggregation model used (currently either moving average or prophet)'
    )

    parser.add_argument('-ws', '--window_size', default='1D', 
        help='Specifies the window size used for the aggregation. The window size is specified \
        using a string in the pandas aliases offsets format: \
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases (e.g., "4H", "2D", "1M" etc.) '
    )

    parser.add_argument('--rolling_window_size', default=10, type=int,
        help='Number of datapoints used to compute the moving average'
    )

    parser.add_argument('--num_std_dev_for_outlier', default=2, type=float,
        help='Specifies the number of standard deviations from the moving moving average by which \
        a point in an aggregated view is considered an outlier'
    )

    parser.add_argument('--num_points_per_window', type=int, default=24, 
        help='Specifies the number of points (i.e., timestamps) that are present in each aggregation window. \
        For example if the window size is 1D and we have one timestmap per hour then the number of points per window is 24.'
    )

    parser.add_argument('--point_frequency', default='1H', 
        help='Specifies the frequency of the data points in the time series. The frequency needs to be specified using a string \
        in the pandas aliases offsets format: \
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases (e.g., "4H", "2D", "1M" etc.)'
    )

    # -------------------------- KDE/Clustering Parameters -------------------------- #
    parser.add_argument('--kernel', default='epa',
        choices=['gaussian', 'epa', 'exponential', 'linear', 'cosine'], 
        help='Specifies the kernel used in KDE'
    )

    parser.add_argument('--bandwidth', type=float, default=24*60*60, help='The bandwidth used by KDE specified in seconds. \
        The bandwidth argument is used only if the --bandwidth_selection argument is set to custom.'
    )

    parser.add_argument('--bandwidth_selection', default='custom', choices=['custom', 'ISJ', 'silverman'],
        help='The mode for selecting the KDE bandwidth. If custom the the value specified in the --bandwidth is used'
    )

    parser.add_argument('--adjusted_local_minima', action='store_true',
        help='If specified then the local minima used to pick the boundaries of clusters are adjusted'
    )


    # -------------------------- Evaluation Parameters -------------------------- #
    # These are optional parameters that are relevant only we are also want to evaluate the detected alerts while streaming

    parser.add_argument('--do_not_save_streaming_dataframes', action='store_true',
        help='If specified the dataframes generated by the simulation of streaming are not saved to pickle files but kept in memory'
    )

    parser.add_argument('--evaluate_raised_alerts', action='store_true', 
        help='If specified we run the evaluation pipeline to identify the alerts and their quality'
    )

    parser.add_argument('--regions_df', type=str, help="Path to the regions_df dataframe that specifies \
        the groundtruth regions. This argument must be specified if `evaluate_raised_alerts` is also specified."
    )

    parser.add_argument('--num_top_k_alerts', type=int, default=10, help="Number of top-k alerts to raise. \
        If `regions_df` is specified then this argument is ignored"
    )



    # Parse the arguments
    args = parser.parse_args()

    print('\nOutput directory:', args.output_dir)
    print('Input time series dataframe path:', args.input_time_series_df)
    print('Number of points used as history:', args.num_points_in_history)
    print('Streaming number of points per step:', args.streaming_num_points_step)
    print('Confidence interval width:', args.confidence_interval_width)
    print('Aggregation_model:', args.aggregation_model)
    print('Window size:', args.window_size)
    print('Rolling window size:', args.rolling_window_size)
    print('Number of standard deviations for outliers:', args.num_std_dev_for_outlier)
    print('Number of points per window:', args.num_points_per_window)
    print('Point frequency:', args.point_frequency)
    print('Kernel:', args.kernel)
    print('Bandwidth:', args.bandwidth, 'seconds')
    print('Bandwidth Selection', args.bandwidth_selection)
    if args.adjusted_local_minima:
        print("Clusters are defined between adjusted local minima!")
    print('Seed:', args.seed)
    if args.future_window_size:
        print("Future Window size for prophet:", args.future_window_size)

    # Evaluation Parameters
    print('\n')
    if args.do_not_save_streaming_dataframes:
        print("Streaming dataframes will not be saved!")
    if args.evaluate_raised_alerts:
        print("Raised alerts will be evaluated!")
        print("regions_df path:", args.regions_df)
    print('\n\n')

    # Create output directory if it doesn't exist
    Path(args.output_dir + 'outputs/').mkdir(parents=True, exist_ok=True)

    # Specify the seed
    np.random.seed(args.seed)

    # Save the input arguments in the output_dir
    with open(args.output_dir + 'args.json', 'w') as fp:
        json.dump(vars(args), fp, sort_keys=True, indent=4)

    main(args)