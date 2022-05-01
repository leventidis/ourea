import pandas as pd

def get_timestamps(tm, period, freq):
    tms =  pd.date_range(start=tm, periods=period, freq=freq)
    return tms

def get_raw_timestamps(timestamps, period, freq):
    '''
    Given a list of `timestamps` in the aggregate level, return a list of timestamps in the raw level 
    '''
    raw_tms = []
    for tm in timestamps:
        tms = get_timestamps(tm, period=period, freq=freq)
        raw_tms.extend(tms)
    return raw_tms

def add_voting_to_df(df, df_agg, period=6, freq='1H'):
    '''
    Updates the df, by adding the votes and residuals of the outliers from the aggregated views
    '''
    # Get the outliers from aggregated time series
    num_outlier_tms = df_agg[df_agg['is_outlier_num_outliers']==1]['timestamp'].values
    residual_outlier_tms = df_agg[df_agg['is_outlier_residual']==1]['timestamp'].values

    # Transpose the outliers from the aggregated view into the raw view (also include their normalized residuals)
    df['is_outlier_num_outliers'] = 0
    df['is_outlier_residual'] = 0
    df.loc[(df['timestamp'].isin(get_raw_timestamps(num_outlier_tms, period=period, freq=freq))), 'is_outlier_num_outliers'] = 1
    df.loc[(df['timestamp'].isin(get_raw_timestamps(residual_outlier_tms, period=period, freq=freq))), 'is_outlier_residual'] = 1

    df['num_raw_data_fit_outliers_norm_residual'] = df_agg.loc[df_agg.index.repeat(period)]['num_raw_data_fit_outliers_norm_residual'][:len(df.index)].values
    df['residual_sum_norm_residual'] = df_agg.loc[df_agg.index.repeat(period)]['residual_sum_norm_residual'][:len(df.index)].values

    # Construct voting scores
    df['raw_voting_score'] = df['is_outlier_raw_data_fit'] + df['is_outlier_num_outliers'] + df['is_outlier_residual']
    df['weighted_score'] = df['norm_residual'] + df['num_raw_data_fit_outliers_norm_residual'] + df['residual_sum_norm_residual']

    return df