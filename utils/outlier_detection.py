from prophet import Prophet
from sklearn import preprocessing

def normalize_vector(vals):
    '''
    Given a list `vals` of numeric values. Return a normalized list that ranges between 0-1
    '''
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_vals = min_max_scaler.fit_transform(vals.reshape(-1, 1) )
    return scaled_vals

def raw_data_fit(df, confidence_interval_width=0.95, point_frequency='1H', future_window_size=None):
    '''
    Given a dataframe with the raw time series, fit a prophet model

    Returns the model and the forecast
    '''
    df_prophet = df.copy()[['timestamp', 'measure']]
    df_prophet.columns=['ds', 'y']

    # Fit the Prophet model on the whole dataframe
    m = Prophet(
        interval_width=confidence_interval_width,
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False
    ).fit(df_prophet)

    # Check if we are fitting Prophet on current data or we also predict in the future 
    if future_window_size:
        future = m.make_future_dataframe(periods=future_window_size, freq=point_frequency)
        forecast = m.predict(future)
    else:
        forecast = m.predict(df_prophet)

    return m, forecast

def update_df_with_raw_model_fit(df, forecast):
    '''
    Update the dataframe to accommodate information from the fitted model on the raw data

    If the forecast dataframe is larger than `df` we only update the entries in `df`
    that are mappable to the `forecast` dataframe
    '''
    df['yhat'] = forecast['yhat']
    df['yhat_lower'] = forecast['yhat_lower']
    df['yhat_upper'] = forecast['yhat_upper']
    df['residual'] = df['measure'] - df['yhat']
    df['abs_residual'] = abs(df['residual'])
    df['norm_residual'] = normalize_vector(df['abs_residual'].values)
    df['percent_error'] = abs((df['measure'] - df['yhat'])/df['yhat']) * 100

    # If y_hat is outside the boundaries set by yhat_lower and yhat_upper then it is classified as an outlier according the fitted model on the raw data
    df['is_outlier_raw_data_fit'] = 0
    df.loc[((df['measure'] < df['yhat_lower']) | (df['measure'] > df['yhat_upper'])), 'is_outlier_raw_data_fit'] = 1

    return df