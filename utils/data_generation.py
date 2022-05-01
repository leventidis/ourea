import timesynth as ts
import numpy as np
import pandas as pd

import math
import random

from prophet import Prophet


class SyntheticTimeSeries:
    def __init__(self, num_periods, num_points_per_period, white_noise_std=0.05, seed=1):
        self.num_periods=num_periods
        self.num_points_per_period=num_points_per_period
        self.white_noise_std=white_noise_std
        self.seed=seed

        self.samples=None
        self.time_samples=None
        self.signals=None
        self.errors=None
        self.df=None

    def generate(self):
        '''
        Generates a synthetic time series and updates the class object accordingly
        '''

        np.random.seed(self.seed) # Set the seed so the synthetic data can be generated in the same way every time
        n_points=self.num_periods*self.num_points_per_period

        time_sampler = ts.TimeSampler(stop_time=2*math.pi*self.num_periods)
        time_samples = time_sampler.sample_regular_time(num_points=n_points)

        # Signal Generator
        pseudo_periodic = ts.signals.PseudoPeriodic(frequency=1, freqSD=0.0, ampSD=0)

        # Noise Generator
        white_noise = ts.noise.GaussianNoise(std=self.white_noise_std)

        # Time Series Initialization
        timeseries = ts.TimeSeries(signal_generator=pseudo_periodic, noise_generator=white_noise)

        # Sample from the Time Series
        samples, signals, errors = timeseries.sample(time_samples)

        self.samples=samples
        self.time_samples=time_samples
        self.signals=signals
        self.errors=errors

    def get_df(self, start_date='2020-06-1', freq='1H'):
        '''
        Returns a dataframe that converts the time series with timestamps and marks white noise generated outliers
        '''
        # Construct dataframe for the synthetic time series
        df = pd.DataFrame()
        df['time'] = self.time_samples
        df['timestamp'] = pd.date_range(start=start_date, periods=self.num_periods*self.num_points_per_period, freq=freq)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        df['measure'] = self.samples
        df['signal'] = self.signals
        df['noise'] = self.errors

        # Specify White Noise Outliers
        is_outlier = []
        outlier_type = []
        for val in df['noise'].values:
            if abs(val) >= 2 * self.white_noise_std:
                outlier_type.append('noise')
            else:
                outlier_type.append(np.nan)
            is_outlier.append(0)
            
        df['is_outlier'] = is_outlier
        df['outlier_type'] = outlier_type
        self.df = df
        return self.df