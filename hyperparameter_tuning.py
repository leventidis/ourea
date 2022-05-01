import pandas as pd
import numpy as np
import json
import pickle
import itertools

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from prophet.serialize import model_to_json, model_from_json
from tqdm import tqdm

import matplotlib.pyplot as plt

# Read CSV and construct dataframe
merged_csv_path = "data/salerts/23901/aggregate_extreme_alerts.csv"
df = pd.read_csv(merged_csv_path, sep='|')

loyalty_offers_svc_df = df[(df['SERVICE_NAME'] == "LoyaltyOffers") & (df['agg_level'] == 'svc')]
loyalty_offers_svc_residual_df = loyalty_offers_svc_df.copy()[['EVENT_TIME', 'residual']]
loyalty_offers_svc_residual_df.columns=['ds', 'y']
loyalty_offers_svc_residual_df['ds'] = pd.to_datetime(loyalty_offers_svc_residual_df['ds'], format='%Y-%m-%d %H:%M:%S')


# Parameters Grid
param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.25, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = []  # Store the RMSEs for each params here
mapes = [] # Store the MAPEs for each param

# Use cross validation to evaluate all parameters
for params in tqdm(all_params):
    m = Prophet(**params).fit(loyalty_offers_svc_residual_df)  # Fit model with given params
    df_cv = cross_validation(m, initial='35 days', period='3 days', horizon = '7 days', parallel="processes")
    df_p = performance_metrics(df_cv, rolling_window=1)
    rmses.append(df_p['rmse'].values[0])
    mapes.append(df_p['mape'].values[0])

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
tuning_results['mape'] = mapes
print(tuning_results)

tuning_results.to_pickle("models/hyperparameters.pickle")