import random
import math
from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np

from alibi_detect.utils.perturbation import inject_outlier_ts

def select_region_length(region_length, seed, variable_region_lengths):
    '''
    If `variable_region_lengths` is specified select a random length for the range

    Returns an integer that corresponds to the length of the range
    '''

    if variable_region_lengths:
        # Choose a length randomly in the specified range of `variable_region_lengths`
        return random.randint(variable_region_lengths[0], variable_region_lengths[1])
    else:
        return region_length


def get_non_overlapping_region_starts(region_length, series_length, num_regions, seed, variable_region_lengths=None):
    '''
    Parameters
    ----------
    region_length (int): The length of each region
    series_length (int): the length of the time series (in terms of the number of datapoints)
    num_regions (int): The number of regions to select in the time series
    variable_region_lengths (list of two int): The range of region lengths if we are using variable region lengths

    If `variable_region_lengths` is specified then choose lengths randomly in the specified range

    Returns
    -------
    Returns a dataframe that specifies the start and end of `num_regions` generated regions 
    that are non overlapping
    '''
    # Ensure that the series is large enough to generate the specified number of regions
    if variable_region_lengths:
        assert variable_region_lengths[1] * num_regions <= series_length, "The series length is too small to generate " \
            + str(num_regions) + ' regions with maximal variable size of ' + str(variable_region_lengths[1]) 
    else:
        assert region_length * num_regions <= series_length, "The series length is too small to generate " \
            + str(num_regions) + ' regions of length ' + str(region_length) 

    region_starts = []
    region_lengths = []
    for _ in range(num_regions):
        region_length = select_region_length(region_length, seed, variable_region_lengths)
        temp = random.randint(0, series_length - region_length) 

        # Ensure that temp is after the end of every region and it does not appear earlier than `region_length`
        # from the start of any other region (prevent overlapping)
        while any(temp >= (start_idx - region_length) and temp <= (start_idx + length) for start_idx, length in zip(region_starts, region_lengths)):
            region_length = select_region_length(region_length, seed, variable_region_lengths)
            temp = random.randint(0, series_length - region_length)
        region_starts.append(temp)
        region_lengths.append(region_length)

    # Construct a dataframe for the regions
    # region_starts = sorted(list(region_starts))
    regions_dict = {'region_id': [], 'start': [], 'end': [], 'length': []}
    for i in range(len(region_starts)):
        regions_dict['region_id'].append(i)
        regions_dict['start'].append(region_starts[i])
        regions_dict['end'].append(region_starts[i]+region_lengths[i]-1)
        regions_dict['length'].append(region_lengths[i])

    regions_df = pd.DataFrame.from_dict(regions_dict)
    regions_df = regions_df.sort_values(by='start')
    regions_df.reset_index(drop=True, inplace=True)
    regions_df['region_id'] = regions_df.index

    return regions_df

def insert_regions_to_df(df, regions_df, is_original=False):
    '''
    Update the dataframe `df`, my specifying the corresponding regions each timestamp

    If `is_original` is specified, then the provided regions were labeled and not injected and we add
    the `outlier_type` column in the dataframe which specifies the outlier_type for each
    timestamp in a region as original 
    '''
    df['region_id'] = np.nan
    if is_original:
        df['outlier_type'] = np.nan
        df['is_outlier'] = 0
    for i in range(len(regions_df.index)):
        df.loc[regions_df['start'][i]:regions_df['start'][i]+regions_df['length'][i]-1, 'region_id'] = i
        if is_original:
            df.loc[regions_df['start'][i]:regions_df['start'][i]+regions_df['length'][i]-1, 'outlier_type'] = 'original'
            df.loc[regions_df['start'][i]:regions_df['start'][i]+regions_df['length'][i]-1, 'is_outlier'] = 1
        else:
            df.loc[regions_df['start'][i]:regions_df['start'][i]+regions_df['length'][i]-1, 'outlier_type'] = regions_df['injection_mode'][i]

    return df

def perturbation_injection(df, seed):
    '''
    Injects perturbation outliers. Modifies the dataframe `df` and marks the modified timestamps as perturbation outliers
    '''

    random.seed(seed)

    X = df['measure'].values.reshape(-1, 1).astype(np.float32)
    data = inject_outlier_ts(X, perc_outlier=3, perc_window=25, n_std=1.5, min_std=1)
    X_outlier, y_outlier, labels = data.data, data.target.astype(int), data.target_names

    df['measure'] = X_outlier

    # Update the outlier type
    for i in range(len(y_outlier)):
        if y_outlier[i] == 1:
            df.loc[i, 'is_outlier'] = 1
            df.loc[i, 'outlier_type'] = 'perturbation'
    return df

def get_indices_for_injection(regions_df_row, seed, outlier_injection_rate=1):
    '''
    Given a region and a specified outlier_injection_rate return randomly the indices for outliers will be injected
    '''
    all_indices = list(range(regions_df_row['start'], regions_df_row['end'] + 1))
    if outlier_injection_rate == 1:
        return  all_indices
    else:
        random.seed(seed)
        # The first and last point of a range are always selected
        num_points_randomly_select = math.ceil(regions_df_row['length']  * outlier_injection_rate) - 2
        selected_indices = random.sample(all_indices[1:-1], k=num_points_randomly_select)
        selected_indices += [all_indices[0], all_indices[-1]]

        return sorted(selected_indices)


def level_shift_injection(df, regions_df, seed, boost_percent=50, outlier_injection_rate=1):
    '''
    Given the time series dataframe `df` and the `regions_df`, modify `df` accordingly by level shifting the data

    Returns the modified dataframe and a list of the region starts selected
    '''

    for index, row in regions_df[regions_df['injection_mode'] == 'level_shift'].iterrows():
        # Select indices to inject with outliers
        indices = get_indices_for_injection(row, seed, outlier_injection_rate)
        for id in indices:
            df.loc[id, 'measure'] = df.loc[id, 'measure'] * (1+boost_percent/100)
            df.loc[id, 'is_outlier'] = 1
            df.loc[id, 'outlier_type'] = 'level_shift'

    return df

def level_shift_injection_old(df, seed, num_regions=5, region_length=24, boost_percent=50, outlier_injection_rate=1):
    '''
    Given a dataframe choose random non-overlapping regions level shift them

    Returns the modified dataframe and a list of the region starts selected
    '''
    random.seed(seed)

    # Get non-overlapping region start points
    regions_df = get_non_overlapping_region_starts(region_length=region_length, series_length=len(df.index), num_regions=num_regions)
    df = insert_regions_to_df(df, region_starts=regions_df['start'], region_lengths=num_regions*[region_length])

    
    for start_idx in regions_df['start']:
        df.loc[start_idx:start_idx+region_length-1, 'measure'] = df.loc[start_idx:start_idx+region_length-1]['measure'].values * (1+boost_percent/100)
        df.loc[start_idx:start_idx+region_length-1, 'is_outlier'] = 1
        df.loc[start_idx:start_idx+region_length-1, 'outlier_type'] = 'level_shift'

    return df, regions_df

def data_gap_injection(df, seed):
    '''
    Modify the df, by removing regions of a time series (i.e. filling gaps)

    The timestamps during the time gap and preceding the time gap are marked as outliers
    '''

    num_gaps=7
    gap_size=12

    # Get non-overlapping region start points
    region_starts = get_non_overlapping_region_starts(region_length=gap_size, series_length=len(df.index), num_regions=num_gaps)

    for start_idx in region_starts:
        df.iloc[start_idx:start_idx+gap_size]['measure'] = np.nan
        for i in range(start_idx, start_idx+gap_size):
            df.loc[i, 'is_outlier'] = 1
            df.loc[i, 'outlier_type'] = 'gap'
    return df

def high_residual_injection(df, regions_df, forecast, seed, gap_percent=5, outlier_injection_rate=1):
    '''
    Modify the dataframe by tweaking regions in the time series so that the values in the region
    have a consistently high residual. This is achieved by setting the values to be `gap_percent` away
    from the upper or lower limits predicted by the forecast (i.e. the forecasting model on the raw data)

    Returns the modified dataframe
    '''
    random.seed(seed)
    
    for index, row in regions_df[regions_df['injection_mode'].isin(['high_residual_consistent', 'high_residual_nearest'])].iterrows():
        # Select indices to inject with outliers
        indices = get_indices_for_injection(row, seed, outlier_injection_rate)
        
        if row['injection_mode']=='high_residual_consistent':
            # Consistently choose if the perturbation is towards the upper or lower boundary
            use_upper = random.choice([True, False])

        # Loop over each point in the selected region
        for i in indices:
            val = df.loc[i, 'measure']
            boundary_range = [forecast.loc[i, 'yhat_lower'], forecast.loc[i, 'yhat_upper']]
            if row['injection_mode']=='high_residual_nearest':
                # Perturb in the direction of the boundary (i.e. upper or lower boundary) that is closest to the current value
                lower_delta = abs(val - forecast.loc[i, 'yhat_lower'])
                upper_delta = abs(val - forecast.loc[i, 'yhat_upper'])
                if upper_delta <= lower_delta:
                    # Move point close to the upper bound
                    new_val = np.percentile(boundary_range, 100-gap_percent)
                else:
                    # Move point close to the lower bound
                    new_val = np.percentile(boundary_range, gap_percent)
            elif row['injection_mode']=='high_residual_consistent':
                if use_upper:
                    new_val = np.percentile(boundary_range, 100-gap_percent)
                else:
                    new_val = np.percentile(boundary_range, gap_percent)
            
            df.loc[i, 'measure'] = new_val
            df.loc[i, 'is_outlier'] = 1
            df.loc[i, 'outlier_type'] = row['injection_mode']

    return df
