import math
import pandas as pd
from prophet import Prophet

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(20, 11)}, font_scale=2)

from .helpers import normalize_vector

def create_aggregate_prophet_plot(df, raw_model, forecast, ylabel, measure_col_name, outliers_col_name, save_path):
    forecast_fig = raw_model.plot(forecast)
    plt.scatter(df[df[outliers_col_name]==1]['timestamp'].values, df[df[outliers_col_name]==1][measure_col_name].values, s=40, c='red', zorder=10, label='Detected Outliers')
    plt.ylabel(ylabel);plt.xlabel('Time');plt.legend()
    forecast_fig.set_size_inches(20, 10)
    forecast_fig.tight_layout()
    plt.savefig(save_path)
    plt.clf()

def create_aggregate_ma_plot(df, points_name, line_name, ylabel, save_path):
    plt.scatter(df['timestamp'], df[points_name])
    plt.plot(df['timestamp'], df[line_name], linewidth=3)
    ax = plt.gca()
    ax = ax.fill_between(df['timestamp'], df[line_name + '_lower'], df[line_name + '_upper'], facecolor='blue', alpha=0.2)
    plt.xlabel('Time');plt.ylabel(ylabel);plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()

def update_agg_df_with_agg_model_fit(df_agg, forecast_num_outliers, forecast_residual_sum):
    '''
    Update the dataframe to accommodate information from the fitted model on the aggregate views data

    Note that the df_agg dataframe and forecast dataframes must be aligned and have the same length
    (i.e., first row in df_agg corresponds to the first row in the `forecast_num_outliers` and `forecast_residual_sum`
    dataframes respectfully)
    '''
    # Outlier detection on the number of outliers and residual sum time series
    df_agg['is_outlier_num_outliers'] = 0
    df_agg['is_outlier_residual'] = 0
    df_agg.loc[(df_agg['num_raw_data_fit_outliers'] < forecast_num_outliers['yhat_lower']) | (df_agg['num_raw_data_fit_outliers'] > forecast_num_outliers['yhat_upper']), 'is_outlier_num_outliers'] = 1
    df_agg.loc[(df_agg['residual_sum'] < forecast_residual_sum['yhat_lower']) | (df_agg['residual_sum'] > forecast_residual_sum['yhat_upper']), 'is_outlier_residual'] = 1

    # Add predicted values and confidence intervals by prophet
    df_agg['predicted_num_raw_data_fit_outliers'] = forecast_num_outliers['yhat']
    df_agg['predicted_num_raw_data_fit_outliers_lower'] = forecast_num_outliers['yhat_lower']
    df_agg['predicted_num_raw_data_fit_outliers_upper'] = forecast_num_outliers['yhat_upper']
    df_agg['predicted_residual_sum'] = forecast_residual_sum['yhat']
    df_agg['predicted_residual_sum_lower'] = forecast_residual_sum['yhat_lower']
    df_agg['predicted_residual_sum_upper'] = forecast_residual_sum['yhat_upper']

    # Compute the abs residuals for the number of outliers and the residual sum time series, then normalize the residuals
    df_agg['num_raw_data_fit_outliers_abs_residual'] = abs(df_agg['num_raw_data_fit_outliers'] - forecast_num_outliers['yhat'])
    df_agg['residual_sum_abs_residual'] = abs(df_agg['residual_sum'] - forecast_residual_sum['yhat'])
    df_agg['num_raw_data_fit_outliers_norm_residual'] = normalize_vector(df_agg['num_raw_data_fit_outliers_abs_residual'].values)
    df_agg['residual_sum_norm_residual'] = normalize_vector(df_agg['residual_sum_abs_residual'].values)

    return df_agg

def get_agg_df(df, window_size):
    '''
    Given the raw time series dataframe `df` return the aggregate dataframe.

    This function does not run any model fitting
    '''
    # Create the aggregate time series over the specified `window_size` and compute the `num_raw_data_fit_outliers` and `residual_sum`
    df_agg = df.resample(window_size, on='timestamp').mean()[['measure']]
    df_agg['residual_sum'] = df.resample(window_size, on='timestamp').sum()['abs_residual']
    df_agg['num_raw_data_fit_outliers'] = df.resample(window_size, on='timestamp').sum()['is_outlier_raw_data_fit']

    df_agg.insert(0, 'timestamp', df_agg.index)
    df_agg = df_agg.reset_index(drop=True)

    return df_agg


def get_aggregate_model_fit(df_agg, args, create_figures=False):
    '''
    `window_size` must be a string that corresponds to the sampling rate (e.g. 10H, 1D etc.) 

    `args` The namespace of the parser that defines the aggregation model using `args.aggregation_model`

    Deprecated: 
        aggregation_model == 'aggregate_ma_model' is deprecated and isn't supported (Remove completely in future)
        create_figures option is also deprecated (Remove completely in future)

    Returns an aggregate dataframe and 2 two lists. A list of the models and a list of the forecasts
    '''

    # # Create the aggregate time series over the specified `window_size` and compute the `num_raw_data_fit_outliers` and `residual_sum`
    # df_agg = df.resample(args.window_size, on='timestamp').mean()[['measure']]
    # df_agg['residual_sum'] = df.resample(args.window_size, on='timestamp').sum()['abs_residual']
    # df_agg['num_raw_data_fit_outliers'] = df.resample(args.window_size, on='timestamp').sum()['is_outlier_raw_data_fit']

    # if args.aggregation_model == 'aggregate_ma_model':
    #     # Moving average and std of number of outliers and residual
    #     df_agg['ma_num_raw_data_fit_outliers'] = df_agg['num_raw_data_fit_outliers'].rolling(args.rolling_window_size).mean()
    #     df_agg['mstd_num_raw_data_fit_outliers'] = df_agg['num_raw_data_fit_outliers'].rolling(args.rolling_window_size).std()
    #     df_agg['ma_residual_sum'] = df_agg['residual_sum'].rolling(args.rolling_window_size).mean()
    #     df_agg['mstd_residual_sum'] = df_agg['residual_sum'].rolling(args.rolling_window_size).std()

    #     # Compute lower and upper boundaries for each aggregation
    #     df_agg['ma_num_raw_data_fit_outliers_lower'] = df_agg['ma_num_raw_data_fit_outliers'] - args.num_std_dev_for_outlier*df_agg['mstd_num_raw_data_fit_outliers']
    #     df_agg['ma_num_raw_data_fit_outliers_upper'] = df_agg['ma_num_raw_data_fit_outliers'] + args.num_std_dev_for_outlier*df_agg['mstd_num_raw_data_fit_outliers']

    #     df_agg['ma_residual_sum_lower'] = df_agg['ma_residual_sum'] - args.num_std_dev_for_outlier*df_agg['mstd_residual_sum']
    #     df_agg['ma_residual_sum_upper'] = df_agg['ma_residual_sum'] + args.num_std_dev_for_outlier*df_agg['mstd_residual_sum']

    #     # Outlier detection on the number of outliers and residual sum time series
    #     df_agg['is_outlier_num_outliers'] = 0
    #     df_agg['is_outlier_residual'] = 0
    #     df_agg.loc[(df_agg['num_raw_data_fit_outliers'] < df_agg['ma_num_raw_data_fit_outliers_lower']) | (df_agg['num_raw_data_fit_outliers'] > df_agg['ma_num_raw_data_fit_outliers_upper']), 'is_outlier_num_outliers'] = 1
    #     df_agg.loc[(df_agg['residual_sum'] < df_agg['ma_residual_sum_lower']) | (df_agg['residual_sum'] > df_agg['ma_residual_sum_upper']), 'is_outlier_residual'] = 1

    #     # Compute the abs residuals for the number of outliers and the residual sum time series, then normalize the residuals
    #     df_agg['num_raw_data_fit_outliers_abs_residual'] = abs(df_agg['num_raw_data_fit_outliers'] - df_agg['ma_num_raw_data_fit_outliers'])
    #     df_agg['residual_sum_abs_residual'] = abs(df_agg['residual_sum'] - df_agg['ma_residual_sum'])

    #     df_agg['num_raw_data_fit_outliers_norm_residual'] = normalize_vector(df_agg['num_raw_data_fit_outliers_abs_residual'].values)
    #     df_agg['residual_sum_norm_residual'] = normalize_vector(df_agg['residual_sum_abs_residual'].values)

    #     df_agg.insert(0, 'timestamp', df_agg.index)

    #     # Draw plots if specified
    #     if create_figures:
    #         create_aggregate_ma_plot(
    #             df=df_agg, points_name='num_raw_data_fit_outliers', line_name='ma_num_raw_data_fit_outliers',
    #             ylabel='Number of Outliers', save_path=args.output_dir+'figures/num_outliers_time_series.svg')
    #         create_aggregate_ma_plot(
    #             df=df_agg, points_name='residual_sum', line_name='ma_residual_sum',
    #             ylabel='Absolute Sum of Residual', save_path=args.output_dir+'figures/residual_time_series.svg')

    if args.aggregation_model == 'aggregate_prophet_model':
        # df_agg.insert(0, 'timestamp', df_agg.index)
        # df_agg = df_agg.reset_index(drop=True)

        # Extract the the aggregated time series
        df_prophet_num_outliers = df_agg.copy()[['timestamp', 'num_raw_data_fit_outliers']]
        df_prophet_residual_sum = df_agg.copy()[['timestamp', 'residual_sum']]
        df_prophet_num_outliers.columns = ['ds', 'y']
        df_prophet_residual_sum.columns = ['ds', 'y']

        # Run Prophet Models
        m_num_outliers = Prophet(interval_width=args.confidence_interval_width, weekly_seasonality=False, yearly_seasonality=False, daily_seasonality=False).fit(df_prophet_num_outliers)
        m_residual_sum = Prophet(interval_width=args.confidence_interval_width, weekly_seasonality=False, yearly_seasonality=False, daily_seasonality=False).fit(df_prophet_residual_sum)
        
        # Check if we are fitting Prophet on current data or we also predict in the future 
        if args.future_window_size:
            num_periods = math.ceil(args.future_window_size / args.num_points_per_window)
            future_num_outliers = m_num_outliers.make_future_dataframe(periods=num_periods, freq=args.window_size)
            future_residual_sum = m_residual_sum.make_future_dataframe(periods=num_periods, freq=args.window_size)
            fr_num_outliers = m_num_outliers.predict(future_num_outliers)
            fr_residual_sum = m_residual_sum.predict(future_residual_sum)
        else:
            fr_num_outliers = m_num_outliers.predict(df_prophet_num_outliers)
            fr_residual_sum = m_residual_sum.predict(df_prophet_residual_sum)

        return df_agg, [m_num_outliers, m_residual_sum], [fr_num_outliers, fr_residual_sum]

        # Outlier detection on the number of outliers and residual sum time series
        df_agg['is_outlier_num_outliers'] = 0
        df_agg['is_outlier_residual'] = 0
        df_agg.loc[(df_prophet_num_outliers['y'] < fr_num_outliers['yhat_lower']) | (df_prophet_num_outliers['y'] > fr_num_outliers['yhat_upper']), 'is_outlier_num_outliers'] = 1
        df_agg.loc[(df_prophet_residual_sum['y'] < fr_residual_sum['yhat_lower']) | (df_prophet_residual_sum['y'] > fr_residual_sum['yhat_upper']), 'is_outlier_residual'] = 1

        # Add predicted values and confidence intervals by prophet
        df_agg['predicted_num_raw_data_fit_outliers'] = fr_num_outliers['yhat']
        df_agg['predicted_num_raw_data_fit_outliers_lower'] = fr_num_outliers['yhat_lower']
        df_agg['predicted_num_raw_data_fit_outliers_upper'] = fr_num_outliers['yhat_upper']
        df_agg['predicted_residual_sum'] = fr_residual_sum['yhat']
        df_agg['predicted_residual_sum_lower'] = fr_residual_sum['yhat_lower']
        df_agg['predicted_residual_sum_upper'] = fr_residual_sum['yhat_upper']

        # Compute the abs residuals for the number of outliers and the residual sum time series, then normalize the residuals
        df_agg['num_raw_data_fit_outliers_abs_residual'] = abs(df_agg['num_raw_data_fit_outliers'] - fr_num_outliers['yhat'])
        df_agg['residual_sum_abs_residual'] = abs(df_agg['residual_sum'] - fr_residual_sum['yhat'])
        df_agg['num_raw_data_fit_outliers_norm_residual'] = normalize_vector(df_agg['num_raw_data_fit_outliers_abs_residual'].values)
        df_agg['residual_sum_norm_residual'] = normalize_vector(df_agg['residual_sum_abs_residual'].values)

        # Draw plots if specified
        if create_figures:
            print('\nCreating aggregation figures...')
            create_aggregate_prophet_plot(
                df=df_agg, raw_model=m_num_outliers, forecast=fr_num_outliers, ylabel='Number of Outliers',
                measure_col_name='num_raw_data_fit_outliers', outliers_col_name='is_outlier_num_outliers', save_path=args.output_dir+'figures/num_outliers_time_series.svg')
            create_aggregate_prophet_plot(
                df=df_agg, raw_model=m_residual_sum, forecast=fr_residual_sum, ylabel='Absolute Sum of Residual',
                measure_col_name='residual_sum', outliers_col_name='is_outlier_residual', save_path=args.output_dir+'figures/residual_time_series.svg')
            print('Finished creating aggregation figures...\n')

    return 0