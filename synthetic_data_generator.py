from collections import defaultdict
from utils.outlier_injection import level_shift_injection, perturbation_injection
import pandas as pd
import numpy as np
import pickle
import json

from prophet import Prophet
from sklearn import preprocessing
from pathlib import Path

import math
import argparse
import random

import matplotlib.pyplot as plt
import seaborn as sns
import timesynth as ts
sns.set(rc={'figure.figsize':(20, 11)}, font_scale=2)

import utils


### Aggregation Functions ###

def normalize_vector(vals):
    '''
    Given a list `vals` of numeric values. Return a normalized list that ranges between 0-1
    '''
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_vals = min_max_scaler.fit_transform(vals.reshape(-1, 1) )
    return scaled_vals

def top_k_evaluator(vals, relevant_types, num_true_relevant_outliers):
    '''
    vals: a sorted list of the outlier types in a given ranking 
    '''
    precision, recall, f1_score = [], [], []
    num_relevant = 0

    # Compute top-k precision, recall and f1_scores
    for k in range(len(vals)):
        if vals[k] in relevant_types:
            num_relevant += 1

        precision_at_k = num_relevant / (k+1)
        recall_at_k = num_relevant / num_true_relevant_outliers

        if (precision_at_k + recall_at_k == 0):
            f1_score_at_k = 0
        else:
            f1_score_at_k = (2*precision_at_k*recall_at_k) / (precision_at_k + recall_at_k)
        
        precision.append(precision_at_k)
        recall.append(recall_at_k)
        f1_score.append(f1_score_at_k)

    return precision, recall, f1_score


def evaluate_ranking(df, ranking_mode='raw'):
    ''' 
    Returns the topk precision, recall and f1-scores given the specified evaluation type
    '''
    relevant_types = ['perturbation', 'level_shift', 'high_residual_nearest', 'high_residual_consistent', 'original']
    num_true_relevant_outliers = len(df[df['outlier_type'].isin(relevant_types)].index)

    if ranking_mode=='raw':
        df_sorted = df[df['is_outlier_raw_data_fit'] == 1].sort_values(by=['abs_residual'], ascending=False)
    elif ranking_mode=='raw_voting':
        df_sorted = df[df['raw_voting_score']>0].sort_values(by=['raw_voting_score', 'norm_residual'], ascending=False)
    elif ranking_mode=='combined_score':
        df_sorted = df[df['raw_voting_score']>0].sort_values(by=['weighted_score'], ascending=False)
    else:
        raise ValueError("The ranking_mode must be one of ['raw', 'raw_voting', 'combined_score']")

    precision, recall, f1_score = top_k_evaluator(
        vals=df_sorted['outlier_type'].values,
        relevant_types=relevant_types,
        num_true_relevant_outliers=num_true_relevant_outliers
    )
    return precision, recall, f1_score


def aggregate_time_series_plot(df, points_name, line_name, std_name, ylabel, filename):
    plt.scatter(df['timestamp'], df[points_name])
    plt.plot(df['timestamp'], df[line_name], linewidth=3)
    ax = plt.gca()
    # ax = ax.fill_between(df['timestamp'], df[line_name] - num_std_dev_for_outlier*df[std_name], df[line_name] + num_std_dev_for_outlier*df[std_name], facecolor='blue', alpha=0.2)
    ax = ax.fill_between(df['timestamp'], df[line_name + '_lower'], df[line_name + '_upper'], facecolor='blue', alpha=0.2)

    plt.xlabel('Time');plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

def create_evaluation_plots(df, save_path):
    precision, recall, f1_score = evaluate_ranking(df, ranking_mode='raw')
    precision_voting, recall_voting, f1_score_voting = evaluate_ranking(df, ranking_mode='raw_voting')
    precision_combined, recall_combined, f1_score_combined = evaluate_ranking(df, ranking_mode='combined_score')

    plt.plot(list(range(len(precision))), precision, label='raw')
    plt.plot(list(range(len(precision_voting))), precision_voting, label='raw_voting')
    plt.plot(list(range(len(precision_combined))), precision_combined, label='combined_score')
    plt.xlabel('k');plt.ylabel('Precision at k');plt.title("Precision @ k")
    plt.legend();plt.tight_layout()
    plt.savefig(save_path+'precision.svg')
    plt.clf()

    plt.plot(list(range(len(recall))), recall, label='raw')
    plt.plot(list(range(len(recall_voting))), recall_voting, label='raw_voting')
    plt.plot(list(range(len(recall_combined))), recall_combined, label='combined_score')
    plt.xlabel('k');plt.ylabel('Recall at k');plt.title("Recall @ k")
    plt.legend();plt.tight_layout()
    plt.savefig(save_path+'recall.svg')
    plt.clf()

    plt.plot(list(range(len(f1_score))), f1_score, label='raw')
    plt.plot(list(range(len(f1_score_voting))), f1_score_voting, label='raw_voting')
    plt.plot(list(range(len(f1_score_combined))), f1_score_combined, label='combined_score')
    plt.xlabel('k');plt.ylabel('F1-Score at k');plt.title("F1-Score @ k")
    plt.legend();plt.tight_layout()
    plt.savefig(save_path+'f1_score.svg')
    plt.clf()

def create_raw_data_fit_plot(df, raw_model, forecast, args):
    forecast_fig = raw_model.plot(forecast)
    # if args.input_time_series_df == None:
    if args.injection_mode == 'level_shift':
        plt.scatter(df[df['outlier_type']=='level_shift']['timestamp'].values, df[df['outlier_type']=='level_shift']['measure'].values, s=40, c='red', zorder=10, label='Injected Level-Shift Outliers')
    if args.injection_mode in ['high_residual_nearest', 'high_residual_consistent']:
        plt.scatter(df[df['outlier_type'].isin(['high_residual_nearest', 'high_residual_consistent'])]['timestamp'].values, df[df['outlier_type'].isin(['high_residual_nearest', 'high_residual_consistent'])]['measure'].values, s=40, c='red', zorder=10, label='Injected High Residual Outliers')
    if args.injection_mode == 'all':
        plt.scatter(df[(df['outlier_type']=='level_shift') & (df['is_outlier']==1)]['timestamp'].values, df[(df['outlier_type']=='level_shift') & (df['is_outlier']==1)]['measure'].values, s=40, c='red', zorder=10, label='Injected Level-Shift Outliers')
        plt.scatter(df[(df['outlier_type']=='high_residual_nearest') & (df['is_outlier']==1)]['timestamp'].values, df[(df['outlier_type']=='high_residual_nearest') & (df['is_outlier']==1)]['measure'].values, s=40, c='blue', zorder=10, label='Injected High Residual Outliers (Nearest)')
        plt.scatter(df[(df['outlier_type']=='high_residual_consistent') & (df['is_outlier']==1)]['timestamp'].values, df[(df['outlier_type']=='high_residual_consistent') & (df['is_outlier']==1)]['measure'].values, s=40, c='purple', zorder=10, label='Injected High Residual Outliers (Consistent)')
    plt.ylabel('Measure');plt.xlabel('Time');plt.legend()
    forecast_fig.set_size_inches(20, 10)
    forecast_fig.tight_layout()
    plt.savefig(args.output_dir + 'figures/raw_data_fit.svg')
    plt.clf()

def agg_models(df, args):
    '''
    Runs the aggergate models and the aggregate views

    Returns the aggregate_df
    '''
    df_agg = utils.aggregation.get_agg_df(df, args.window_size)
    df_agg, models_agg, forecasts_agg = utils.aggregation.get_aggregate_model_fit(df_agg=df_agg, args=args)
    df_agg = utils.aggregation.update_agg_df_with_agg_model_fit(
        df_agg=df_agg,
        forecast_num_outliers=forecasts_agg[0].head(len(df_agg.index)),
        forecast_residual_sum=forecasts_agg[1].head(len(df_agg.index))
    )
    utils.aggregation.create_aggregate_prophet_plot(
        df=df_agg, raw_model=models_agg[0], forecast=forecasts_agg[0], ylabel='Number of Outliers',
        measure_col_name='num_raw_data_fit_outliers', outliers_col_name='is_outlier_num_outliers',
        save_path=args.output_dir+'figures/num_outliers_time_series.svg'
    )
    utils.aggregation.create_aggregate_prophet_plot(
        df=df_agg, raw_model=models_agg[1], forecast=forecasts_agg[1], ylabel='Absolute Sum of Residual',
        measure_col_name='residual_sum', outliers_col_name='is_outlier_residual',
        save_path=args.output_dir+'figures/residual_time_series.svg'
    )
    return df_agg


def main(args):

    if args.input_time_series_df:
        # Read the input time series from file together with its input regions (if provided)
        df = pd.read_pickle(args.input_time_series_df)
        if args.input_regions_df:
            regions_df = pd.read_pickle(args.input_regions_df)
            df = utils.outlier_injection.insert_regions_to_df(df, regions_df, is_original=True)
    else:
        # Generate Time Series if no input time series was specified
        STS = utils.data_generation.SyntheticTimeSeries(
            num_periods=args.synthetic_num_periods, num_points_per_period=7*24,
            white_noise_std=args.white_noise_std, seed=args.seed)
        STS.generate()

        # Convert Time Series to dataframe and assign timestamps 
        df = STS.get_df()

    # Inject Outliers
    if args.injection_mode != None:
        # Get the regions where the injected outliers will be injected at 
        regions_df = utils.outlier_injection.get_non_overlapping_region_starts(
            region_length=args.inj_region_length,
            series_length=len(df.index),
            num_regions=args.inj_num_regions,
            seed=args.seed, variable_region_lengths=args.variable_inj_region_length
        )

        # If we are injecting multiple modes of outliers randomly select the mode for each region 
        if args.injection_mode == 'all':
            regions_df['injection_mode'] = np.nan
            valid_injection_modes = ['level_shift', 'high_residual_consistent', 'high_residual_nearest']
            regions_df['injection_mode'] = [random.choice(valid_injection_modes) for region_id in regions_df['region_id']]
        else:
            regions_df['injection_mode'] = args.injection_mode

        # Update dataframe to include the region information
        df = utils.outlier_injection.insert_regions_to_df(df, regions_df)

        if 'level_shift' in regions_df['injection_mode'].values:
            df = level_shift_injection(df, regions_df, seed=args.seed,
                boost_percent=args.inj_boost_percent, outlier_injection_rate=args.outlier_injection_rate)

        print('Raw time series has', len(df.index), 'data points')
        print('There are', len(df[df['outlier_type']=='noise']), 'white noise outliers')
        print('There are', len(df[df['outlier_type']=='level_shift']), 'injected level_shift outliers')

    # Fit model on raw data
    raw_model, forecast = utils.outlier_detection.raw_data_fit(df, args.confidence_interval_width)

    if not args.input_regions_df:
        if 'high_residual_consistent' or 'high_residual_nearest' in regions_df['injection_mode'].values:
            df = utils.outlier_injection.high_residual_injection(
                df=df, regions_df=regions_df, forecast=forecast, seed=args.seed,
                gap_percent=args.inj_gap_percent, outlier_injection_rate=args.outlier_injection_rate
            )
            # Re-fit the model based on the updated dataframe
            raw_model, forecast = utils.outlier_detection.raw_data_fit(df, args.confidence_interval_width)
            print('\nThere are', len(df[df['outlier_type'].isin(['high_residual_consistent', 'high_residual_nearest'])]), 'injected high residual outliers')
            print('\n\n')

    create_raw_data_fit_plot(df, raw_model, forecast, args=args)
    df = utils.outlier_detection.update_df_with_raw_model_fit(df=df, forecast=forecast)

    ###### Create the Aggregated Views and perform Outlier detection over them ######
    df_agg = agg_models(df, args)

    ##### Voting & Evaluation #####
    df = df.copy()[:math.floor(len(df.index) / len(args.window_size)) * len(df_agg.index)]
    df = utils.voting.add_voting_to_df(df=df, df_agg=df_agg, period=args.num_points_per_window, freq=args.point_frequency)
    create_evaluation_plots(df, args.output_dir+'figures/')

    ##### Run KDE, extract the density distribution and identify the top-k alerts #####
    df, density_df, clusters_df, selected_bandwidth = utils.clustering.perform_clustering(
        df=df, kernel=args.kernel,
        bandwidth_selection=args.bandwidth_selection, bandwidth=args.bandwidth, adjusted_local_minima=args.adjusted_local_minima
    )

    density_df.to_pickle(args.output_dir+"density_df.pickle")
    clusters_df.to_pickle(args.output_dir+"clusters_df.pickle")
    utils.clustering.get_timeseries_and_density_figure(df, density_df, clusters_df, len(regions_df.index), args.output_dir+'figures/')

    # Save Generated Dataframes
    print('Saving generated dataframes...')
    df.to_pickle(args.output_dir+"df.pickle")
    df_agg.to_pickle(args.output_dir+"df_agg.pickle")
    if 'regions_df' in locals():
        regions_df.to_pickle(args.output_dir+'regions_df.pickle')



if __name__ == "__main__":
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Evaluating detected outliers by in a synthetic time series by injecting them')

    parser.add_argument('-o', '--output_dir', metavar='output_dir', required=True,
    help='Path to the output directory where output files and figures are stored. \
    Path must terminate with backslash "\\"')

    parser.add_argument('--input_time_series_df', help='If specified, uses the provided dataframe as the input time series instead of generating a synthetic one. \
    Outliers can still be injected if desired.')

    parser.add_argument('--input_regions_df', help='If specified, uses the provided dataframe to specify regions of outliers that will need to be evaluated. \
    This argument should only be used the --input_time_series_df is specified.')

    ########### Aggregation related Parameters arguments ###########

    # Variables regarding aggregation 
    parser.add_argument('-ws', '--window_size', default='1D', help='Specifies the window size used for the aggregation. The window size is specified \
    using a string in the pandas aliases offsets format: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases (e.g., "4H", "2D", "1M" etc.) ')

    parser.add_argument('--num_points_per_window', type=int, default=24, help='Specifies the number of points (i.e., timestamps) that are present in each aggregation window. \
    For example if the window size is 1D and we have a timestmap per hour then the number of points per window is 24.')

    parser.add_argument('--point_frequency', default='1H', help='Specifies the the frequency of the data points in the time series. The frequency needs to be specifed using a string \
    in the pandas aliases offsets format: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases (e.g., "4H", "2D", "1M" etc.)')

    parser.add_argument('--confidence_interval_width', default=0.95, type=float,
        help='Specifies the confidence interval used by the raw data prophet model. If the aggregate views also use Prophet, use the same confidence interval there as well.'
    )

    parser.add_argument('-agg_m', '--aggregation_model', choices=['aggregate_ma_model', 'aggregate_prophet_model'],
        default="aggregate_prophet_model", help='Aggregation model used (currently either moving average or prophet)'
    )

    parser.add_argument('--rolling_window_size', default=10, type=int,help='Number of datapoints used to compute the moving average')
    parser.add_argument('--num_std_dev_for_outlier', default=2, type=float,
        help='Specifies the number of standard deviations from the moving moving average by which a point in an aggregated view is considered an outlier'
    )    

    ########### END Aggregation related Parameters arguments ###########


    ########### Outlier Injection related arguments ###########

    parser.add_argument('-im', '--injection_mode', metavar='injection_mode', choices=['level_shift', 'high_residual_consistent', 'high_residual_nearest', 'all'],
    help='Type of outliers injected')

    parser.add_argument('--inj_num_regions', default=30, type=int, help='Number of regions to be injected')
    parser.add_argument('--inj_region_length', default=24, type=int, help='The length of the injected region (in terms of the number of consecutive data points)')
    parser.add_argument('--variable_inj_region_length', type=int, nargs=2, metavar=('min_length', 'max_length'),
        help='If specified, the injected regions have variable length ranging between the specified minimum and maximum length. \
            The length is specified in terms of the number consecutive'
    )
    parser.add_argument('--inj_boost_percent', default=50, type=float, help='The percentage by which the selected points are boosted by')
    parser.add_argument('--inj_gap_percent', default=2, type=float, help='The relative gap percentage with respect to the confidence boundary')

    parser.add_argument('--outlier_injection_rate', type=utils.helpers.range_limited_float_type, default=1.0,
    help='If specified not all data points in a specified region for injection are used, instead only `outlier_injection_rate` \
    percent of points are injected with an outlier. The first and last points of a region are always injected and are not skipped. \
    Must be a floating number between 0 and 1.')

    ########### END outlier injection arguments ###########

    parser.add_argument('--bandwidth', type=float, default=24*60*60, help='The bandwidth used by KDE specified in seconds. \
        The bandwidth argument is used only if the --bandwidth_selection argument is set to custom.'
    )

    parser.add_argument('--bandwidth_selection', default='custom', choices=['custom', 'ISJ', 'silverman'],
        help='The mode for selecting the KDE bandwidth. If custom the the value specified in the --bandwidth is used'
    )

    parser.add_argument('--kernel', default='epa',
        choices=['gaussian', 'epa', 'exponential', 'linear', 'cosine'], 
        help='Specifies the kernel used in KDE'
    )

    parser.add_argument('--adjusted_local_minima', action='store_true',
        help='If specified then the local minima used to pick the boundaries of clusters are adjusted'
    )

    parser.add_argument('--synthetic_num_periods', default=40, type=int, help='Number of periods to be generated for the synthetic dataset.')

    parser.add_argument('--white_noise_std', default=0.05, type=float, help='The standard deviation of noise to the base time series. This only applies if we are generated a synthetic time series.')

    parser.add_argument('--future_window_size', type=int,
        help='If specified the prophet model is used to predict into the future. \
        The value is specified using an integer denoting the number of data points we want to predict ahead'
    )

    parser.add_argument('--seed', metavar='seed', type=int,
        help='Seed used for the random data generator. Must be a positive integer'
    )

    

    # Parse the arguments
    args = parser.parse_args()
    
    print('\nOutput directory:', args.output_dir)
    print('Window size:', args.window_size)
    print('Number of points per window:', args.num_points_per_window)
    print('Point frequency:', args.point_frequency)
    print('Confidence interval width:', args.confidence_interval_width)
    print('Aggregation model:', args.aggregation_model)

    if args.aggregation_model == 'aggregate_ma_model':
        print('Rolling Window size:', args.rolling_window_size)
        print('Number of Standard deviations for outlier:', args.num_std_dev_for_outlier)

    if args.input_time_series_df:
        print('Input time series dataframe:', args.input_time_series_df)
    if args.input_regions_df:
        print('Input regions dataframe:', args.input_regions_df)

    if args.injection_mode:
        print('Injection mode:', args.injection_mode)
        print('Injected number of regions:', args.inj_num_regions)
        if args.variable_inj_region_length:
            print("Injected regions have lengths ranging from", args.variable_inj_region_length[0],
                'to', args.variable_inj_region_length[1])
        else:
            print('Injected regions length:', args.inj_region_length)
        if args.injection_mode == 'level_shift':
            print('Injected boost percentage:', args.inj_boost_percent)
        if args.injection_mode in ['high_residual_consistent', 'high_residual_nearest']:
            print('Injected gap percent:', args.inj_gap_percent)
        if args.outlier_injection_rate:
            print('Outlier injection rate:', args.outlier_injection_rate)

    print('Number of periods to generate for the synthetic dataset:', args.synthetic_num_periods)
    print('White noise STD for the synthetic dataset:', args.white_noise_std)

    if args.kernel:
        print("KDE Kernel:", args.kernel)
        print("KDE Bandwidth:", args.bandwidth, 'seconds')

    if args.seed:   
            print('User specified seed:', args.seed)
            random.seed(args.seed)
            np.random.seed(args.seed)
    else:
        # Generate a random seed if not specified
        args.seed = random.randrange(2**32)
        random.seed(args.seed)
        print('No seed specified, picking one at random. Seed chosen is:', args.seed)
    print('\n\n')

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir + 'figures/').mkdir(parents=True, exist_ok=True)

    # Save the input arguments in the output_dir
    with open(args.output_dir + 'args.json', 'w') as fp:
        json.dump(vars(args), fp, sort_keys=True, indent=4)

    main(args)