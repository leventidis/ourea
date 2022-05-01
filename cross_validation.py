import pandas as pd
import json
import pickle

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


# Fit the Prophet model on the whole dataframe
m = Prophet()
forecast = m.fit(loyalty_offers_svc_residual_df).predict(loyalty_offers_svc_residual_df)

print("Running Cross-Validation")
df_cv = cross_validation(m, initial='35 days', period='3 days', horizon = '7 days')
df_cv.to_pickle("models/cross_validation_df.pickle")
print(df_cv)
print("Finished Cross-Validation\n")

print("Computing Performance Metrics")
df_p = performance_metrics(df_cv)
print(df_p)
print("\nFinished Computing Performance Metrics\n")

fig = plot_cross_validation_metric(df_cv, metric='mape')
plt.tight_layout()
plt.savefig("figures/cv_mape.svg")