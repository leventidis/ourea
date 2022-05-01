import numpy as np
import pandas as pd

from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from scipy import integrate
from KDEpy import FFTKDE

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import seaborn as sns
sns.set(rc={'figure.figsize':(20, 11)}, font_scale=2)

def find_clusters(points, minima):
    '''
    Define a cluster as the number of points between two minima

    Returns a list of lists, where each list corresponds to the timestamps of a discovered cluster 
    '''
    
    assert len(minima) >= 2, 'The minima array should have at least 2 points'

    clusters = []

    # First mimima
    clusters.append(points[points < minima[0]])

    # All the minimal (except the first and last)
    for i in range(1, len(minima)-1):
        clusters.append(points[(points >= minima[i]) * (points <= minima[i+1])])

    # Last Minima
    clusters.append(points[points >= minima[-1]])

    return clusters

def update_df_with_cluster_ids(df, clusters_df):
    '''
    Given a dataframe, update it so that it maps each row to its identified cluster id and its relevant metadata.
    '''
    df['cluster_id'] = np.nan
    # Remove columns from df if they appear in clusters_df
    df = df.drop(clusters_df.columns, axis=1, errors='ignore')
    for index, row in clusters_df.iterrows():
        df.loc[(df['unix_time'] >= row['start']) & (df['unix_time'] <= row['end']), 'cluster_id'] = row['cluster_id']

    # Join the clusters_df dataframe on df on the `cluster_id`
    df = df.join(clusters_df.set_index('cluster_id'), on='cluster_id', how='left')

    return df

def get_local_minima(arr):
    '''
    Given a list `arr` find its local minima.
    If there are flat minima present then only pick the boundaries of the minima flat regions
    '''
    minima_all = argrelextrema(arr, np.less_equal)[0]

    # Refine the minima list to avoid flat minima
    minima = []
    for min_idx in minima_all:
        # Loop over indices that aren't at the boundaries of the input list
        if ((min_idx > 0) and (min_idx < (len(arr) - 1))):
            if arr[min_idx -1] != arr[min_idx + 1]:
                minima.append(min_idx)
    
    return minima

def get_local_maxima(arr):
    '''
    Given a list `arr` find its local maxima.
    If there are flat maxima present then only pick the boundaries of the maxima flat regions
    '''
    maxima_all = argrelextrema(arr, np.greater_equal)[0]

    # Refine the maxima list to avoid flat maxima
    maxima = []
    for max_idx in maxima_all:
        # Loop over indices that aren't at the boundaries of the input list
        if ((max_idx > 0) and (max_idx < (len(arr) - 1))):
            if arr[max_idx -1] != arr[max_idx + 1]:
                maxima.append(max_idx)
    
    return maxima

def get_adjusted_local_minima(minima, maxima, density, threshold=0.5):
    '''
    A local minimum is added in the adjusted_minima list if its density value divided by 
    the density value at its local maximum is less than the specified `threshold`
    '''
    adjusted_minima = [minima[0]]
    
    best_maximum = 0
    for i in range(0, len(minima)-1):
        
        # Find the local maximum between minima[i] and minima[i+1]
        max_idx = None
        for cur_max_idx in maxima:
            if (cur_max_idx > minima[i] and cur_max_idx < minima[i+1]):
                max_idx = cur_max_idx
                break
        
        if max_idx != None:
            
            # Check if the density at max_idx is greater than the 'best_maximum'
            if density[max_idx] > best_maximum:
                best_maximum = density[max_idx]
            
            # Check if we add the current local minimum to the list
            if (density[minima[i+1]] / best_maximum) < threshold:
                adjusted_minima.append(minima[i+1])
                best_maximum = 0
        
        if max_idx == None:
            # There is no local maximum between two consecutive local minima 
            # (i.e., the two local minima are the boundaries of a flat curve section)
            # Add the rightmost local minimum 
            adjusted_minima.append(minima[i+1])
            best_maximum=0
    
    return adjusted_minima

def find_nearest_idx(arr, value):
    '''
    arr (numpy array): a numpy array of values

    Given a numpy array, find the index in the array that is closest to the specified `value`
    This function returns an index in `arr`
    '''
    idx = (np.abs(arr - value)).argmin()
    return idx

def get_points_area(samples, density_dist, point_min, point_max):
    '''
    Find the area under the density_dist between point_min and point_max

    This is done by finding what are the closest points in samples to point_min and point_max
    '''
    min_idx = find_nearest_idx(samples, point_min)
    max_idx = find_nearest_idx(samples, point_max)

    area = integrate.simps(x = samples[min_idx:max_idx+1], y=density_dist[min_idx:max_idx+1])
    return area

def get_clusters_df(samples, density_dist, minima_idx_list, points):
    '''
    samples (list of float): The time positions where the density is computed

    density_dist (list of float): The probability density of the alerts at each specified sampled time

    minima_idx_list (list of int): The indices in the `samples` array where the local minima occur

    points (list of float): A list with the timestamps of the detected alerts (i.e. the raw data points for which we estimated the density distribution)
    '''

    assert len(minima_idx_list) >= 2, 'The minima array should have at least 2 points'

    data = {'cluster_id': [], 'start': [], 'end': [], 'point_start': [], 'point_end': [], 'time_length': [], 'num_points': [], 'area': [], 'area_pts': []}

    # First Cluster (Everything before the first local minimum is a cluster)
    first_cluster_pts = points[points < samples[minima_idx_list[0]]]
    if len(first_cluster_pts) > 0:
        data['cluster_id'].append(0)
        data['start'].append(samples[0])
        data['end'].append(samples[minima_idx_list[0]])
        data['point_start'].append(min(first_cluster_pts))
        data['point_end'].append(max(first_cluster_pts))
        data['time_length'].append(data['end'][0] - data['start'][0])
        data['num_points'].append(len(first_cluster_pts))
        data['area'].append(integrate.simps(x = samples[0:minima_idx_list[0]+1], y=density_dist[0:minima_idx_list[0]+1]))
        data['area_pts'].append(get_points_area(samples, density_dist, min(first_cluster_pts), max(first_cluster_pts)))

    # Compute all clusters between two consecutive local minima
    for i in range(0, len(minima_idx_list)-1):
        cluster_pts = points[(points > samples[minima_idx_list[i]]) * (points <= samples[minima_idx_list[i+1]])]
        if len(cluster_pts) > 0:
            data['cluster_id'].append(i+1)
            start = samples[minima_idx_list[i]]
            end = samples[minima_idx_list[i+1]]
            data['start'].append(start)
            data['end'].append(end)
            data['point_start'].append(min(cluster_pts))
            data['point_end'].append(max(cluster_pts))
            data['time_length'].append(end-start)
            data['num_points'].append(len(cluster_pts))
            data['area'].append(integrate.simps(x = samples[minima_idx_list[i]:minima_idx_list[i+1]], y=density_dist[minima_idx_list[i]:minima_idx_list[i+1]]))
            data['area_pts'].append(get_points_area(samples, density_dist, min(cluster_pts), max(cluster_pts)))

    # Last Cluster (Everything after the last local minumum)
    last_cluster_pts = points[points >= samples[minima_idx_list[-1]]]
    if len(last_cluster_pts) > 0:
        data['cluster_id'].append(len(minima_idx_list))
        start = samples[minima_idx_list[-1]]
        end = samples[-1]
        data['start'].append(start)
        data['end'].append(end)
        data['point_start'].append(min(last_cluster_pts))
        data['point_end'].append(max(last_cluster_pts))
        data['time_length'].append(end-start)
        data['num_points'].append(len(last_cluster_pts))
        data['area'].append(integrate.simps(x = samples[minima_idx_list[-1]:], y=density_dist[minima_idx_list[-1]:]))
        data['area_pts'].append(get_points_area(samples, density_dist, min(last_cluster_pts), max(last_cluster_pts)))

    clusters_df = pd.DataFrame.from_dict(data)

    return clusters_df

def get_timeseries_and_density_figure(df, density, clusters, k, output_dir):
    df = df.sort_values(by='timestamp')

    k_clusters = clusters.sort_values(by='area', ascending=False).head(k).reset_index()

    # Plot the Figure
    fig, axs = plt.subplots(2, sharex=True)
    fig.subplots_adjust(hspace=0)

    axs[0].plot(df['unix_time'], df['measure'])
    axs[0].scatter(df[df['is_outlier']==1]['unix_time'].values, df[df['is_outlier']==1]['measure'].values, s=90, c='red', zorder=10, label='True Injected Outliers')
    axs[0].scatter(df[df['raw_voting_score']>0]['unix_time'].values, df[df['raw_voting_score']>0]['measure'].values, s=30, c='green', zorder=10, label='Detected Outliers')
    axs[0].set_ylabel('Measure');axs[0].legend()

    # Plot the alert regions
    for _, alert in k_clusters.iterrows():
        axs[0].add_patch(patches.Rectangle((alert['start'], -1.5), width=alert['end']-alert['start'], height=3, linewidth=0, color='yellow', zorder=10, alpha=0.40))

    axs[1].plot(density['unix_time'], density['density'])
    axs[1].set_ylabel('Density');axs[1].set_xlabel('Unix Time')
    for idx, alert in k_clusters.iterrows():
        axs[1].add_patch(patches.Rectangle((alert['start'], 0), width=alert['end']-alert['start'], height=density['density'].max(), linewidth=0, color='yellow', zorder=10, alpha=0.40))
        x_loc = alert['start'] + (alert['end'] - alert['start'])/2
        axs[1].text(x=x_loc, y=density['density'].max(), s=str(idx+1), fontsize=12, color='red', horizontalalignment='center')

    plt.tight_layout()

    plt.savefig(output_dir + 'timeseries_density_plot.svg')


def cluster(df, clustering_mode, output_dir, bandwidth=24*60*60):
    '''
    Bandwidth set to 1 day (TODO: Dynamically change it?)
    '''
    if clustering_mode == 'KDE':
        df['unix_time'] = df['timestamp'].apply(lambda x: x.timestamp())
        df_filtered = df.copy()[df['raw_voting_score']>0]    

        # Estimate the Kernel Density Distribution of the detected outliers across all views and plot it
        points = np.array(df_filtered['unix_time'].to_numpy()).reshape(-1,1)
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(points)
        s = np.linspace(df['unix_time'].min(),df['unix_time'].max(), num=3*len(df.index))
        log_density = kde.score_samples(s.reshape(-1,1))
        plt.plot(s, log_density);plt.xlabel('Time');plt.ylabel('Log Density');plt.tight_layout()
        plt.savefig(output_dir + 'log_density_distribution.svg')

        density_df = pd.DataFrame({'sample': s, 'log_density': log_density, 'density': np.exp(log_density)})
        density_df['timestamp'] = pd.to_datetime(density_df['sample'], unit='s')
        density_df.to_pickle(output_dir+'density_df.pickle')

        # Find Maxima and Minima points from the leg_density distribution
        mi, ma = argrelextrema(log_density, np.less)[0], argrelextrema(log_density, np.greater)[0]

        # Find the clusters and save them in a dataframe
        clusters_df = get_clusters_df(samples=s, density_dist=np.exp(log_density), minima_idx_list=mi, points=points)
        clusters_df.to_pickle(output_dir+"clusters_df.pickle")

        # Plot the timeseries and density side by side plot
        get_timeseries_and_density_figure(df, density_df, output_dir)

        # TODO: Post-process the clusters or filter out some clusters

        # Update the dataframe to include cluster information 
        df = update_df_with_cluster_ids(df, clusters_df)

    return df

def perform_clustering(df, kernel='gaussian', bandwidth_selection='custom', bandwidth=24*60*60, adjusted_local_minima=False, threshold=0.5):
    '''
    Performs KDE over the input dataframe to find the density function of outliers and identifies
    the clusters of outliers that form the alerts.

    KDE is performed using KDEpy

    Parameters
    ----------
    df (pandas dataframe): A dataframe of the raw time series where each timestamp is associated with a voting score 
    (the score corresponds to how many views voted it as an outlier) 

    kernel (string): The kernel function used for KDE. Must be one of ['gaussian', 'epa', 'tri', 'cosine']

    bandwidth (float): The bandwidth parameter used in KDE. Specified in number of seconds

    adjusted_local_minima (bool): If specified then local minima must be seperated by a significant peak

    threshold (float): If `adjusted_local_minima` is specified then the threshold specifies the maximum allowed ratio
    between a local minimum and preceding local maximum to be considered in the adjusted local minima list

    Returns
    -------
    Three dataframes and a float are returned

    df:  The input dataframe modified to include the associated cluster ids as well as a unix_time column
    density_df: A dataframe with the estimated density of outliers over the range of timestamps in the input dataframe
    clusters_df: A dataframe that describes the outlier clusters (i.e., the alerts)
    selected_bandwidth: The bandwidth that was eventually selected. If custom was selected then 'bandwidth' is simply returned back. Otherwise the bandwidth used by the selection method is returned
    '''

    df['unix_time'] = df['timestamp'].apply(lambda x: x.timestamp())
    outliers_df = df.copy()[df['raw_voting_score'] > 0]
    points = []
    # Each vote adds one point (a timestamp with more than 1 votes is added more than once)
    for _, row in outliers_df.iterrows():
        points += [row["unix_time"]] * row['raw_voting_score']


    # Estimate the Kernel Density Distribution of the detected outliers across all views and plot it
    # points = np.array(points).reshape(-1,1)
    # kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(points)
    s = np.linspace(df['unix_time'].min()-100,df['unix_time'].max()+100, num=3*len(df.index))

    if bandwidth_selection == 'custom':
        density = FFTKDE(kernel=kernel, bw=bandwidth).fit(points).evaluate(s)
        selected_bandwidth = bandwidth
    else:
        model = FFTKDE(kernel=kernel, bw=bandwidth_selection).fit(points)
        selected_bandwidth = model.bw
        density = model.evaluate(s)

    # density = np.exp(kde.score_samples(s.reshape(-1,1)))
    density_df = pd.DataFrame({'unix_time': s, 'density': density})
    density_df['timestamp'] = pd.to_datetime(density_df['unix_time'], unit='s')


    # Find the Minima points from the log_density distribution
    mi, ma = get_local_minima(density), get_local_maxima(density)

    if adjusted_local_minima:
        mi = get_adjusted_local_minima(mi, ma, density, threshold)

    # Find the clusters and save them in a dataframe
    clusters_df = get_clusters_df(
        samples=s, density_dist=density, minima_idx_list=mi,
        points=outliers_df['unix_time'].to_numpy().reshape(-1,1)
    )

    # Update the dataframe to include cluster ID information for each data point 
    df = update_df_with_cluster_ids(df, clusters_df)

    return df, density_df, clusters_df, selected_bandwidth