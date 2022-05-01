import pandas as pd
import numpy as np

from sklearn.metrics import ndcg_score
from tqdm import tqdm
from .helpers import natural_keys, compute_f1_score
from .metrics import range_based_precision, range_based_recall


class AlertEvaluation:
    k=1
    df_list = []
    density_df_list = []
    clusters_df_list = []
    alerts_df = None
    df_index_to_list_index = {}

    
    def __init__(self, df_full, k, regions_df_path):
        self.df_full = df_full
        self.k = k
        self.regions_df_path = regions_df_path
        
        regions_df = pd.read_pickle(regions_df_path)
        self.regions_df = update_regions_df(regions_df, self.df_full)

    def set_dfs_to_indices(self, num_points_in_history, streaming_num_points_step):
        for i in range(len(self.df_list)):
            self.df_index_to_list_index[num_points_in_history + i*streaming_num_points_step] = i


    def get_alerts_df(self):
        '''
        Returns
        -------
        Returns a dataframe with all top-k alerts raised while streaming

        If `regions_df` is specified then `k` is adjusted dynamically based on the true regions in regions_df 
        '''
        k=self.k

        # Dataframe that corresponds to all the alerts that were sent out while streaming
        alerts_df = pd.DataFrame(columns=['alert_time', 'cluster_start', 'cluster_point_start', 'alert_idx', 'cluster_start_idx'])

        # For each consecutive execution check the found clusters and if they make it into the top k
        for i in range(len(self.clusters_df_list)):
            clusters = self.clusters_df_list[i]
            clusters['rank'] = clusters['area'].rank(ascending=False)

            last_cluster_id = clusters['cluster_id'].max() 
            last_cluster_rank = clusters[clusters['cluster_id'] == last_cluster_id]['rank'].values[0]

            df = self.df_list[i]

            # Adjust k if `regions_df` is specified
            if self.regions_df_path is not None:
                new_k = len(self.regions_df[self.regions_df['unix_start'] <= df['unix_time'].max()].index)
                if k != new_k:
                    k=new_k
    
            # Check if last cluster makes it into the top-k
            if last_cluster_rank <= k:
                last_cluster_start = clusters[clusters['cluster_id'] == last_cluster_id]['start'].values[0]
                last_cluster_point_start = clusters[clusters['cluster_id'] == last_cluster_id]['point_start'].values[0]
                last_cluster_start_idx = df.iloc[(df['unix_time']-last_cluster_start).abs().argsort()[:1]].index.values[0]

                alert_time = df['unix_time'].max()
                if alerts_df.empty:
                    alerts_df.loc[len(alerts_df.index)] = [alert_time, last_cluster_start, last_cluster_point_start, len(df.index), last_cluster_start_idx]
                else:
                    # Check if there is already an alert after the current cluster_start
                    latest_recorded_alert = alerts_df['alert_time'].max()
                    if last_cluster_start > latest_recorded_alert:
                        alerts_df.loc[len(alerts_df.index)] = [alert_time, last_cluster_start, last_cluster_point_start, len(df.index), last_cluster_start_idx]
        
        return alerts_df

    def get_true_regions_coverage(self):
        '''
        Returns an updated alerts_df that denotes the true region(s) covered by an alert as well as if the alert happened during that region 
        '''
        # Process alerts_df to check for accuracy and what region it corresponds to if it is correct
        self.alerts_df['region_id_covered'] = np.nan
        self.alerts_df['alert_in_region'] = False
        for index, alert in self.alerts_df.iterrows():
            for _, region in self.regions_df.iterrows():
                # Check if the alert['alert_time'] happens during the interval of a true region
                if (alert['alert_time'] >= region['unix_start'] and alert['alert_time'] <= region['unix_end']):
                    self.alerts_df.loc[index, 'alert_in_region'] = True

                # Check if there is a true region that has an overlap with the interval [row['cluster_start'], row['alert_time']]
                if (region['unix_start'] <= alert['cluster_start'] <= region['unix_end']) or (alert['cluster_start'] <= region['unix_start'] <= alert['alert_time']):
                    self.alerts_df.loc[index, 'region_id_covered'] = int(region['region_id'])

        return self.alerts_df

    def update_alerts_df_with_eval_scores(self):
        '''
        Updates the alerts_df by computing the range based precision, recall and f1-scores @k 
        of the raised alerts so far.

        This also computes the range based precision, recall and f1-score variants with outlier bounds 
        (i.e., regions specified from alerts_df['cluster_point] starting from )
        '''
        self.alerts_df['range_based_precision@k'] = np.nan
        self.alerts_df['range_based_precision@k'] = np.nan
        self.alerts_df['range_based_f1_score@k'] = np.nan

        self.alerts_df['range_based_precision@k_outlier_bounds'] = np.nan
        self.alerts_df['range_based_precision@k_outlier_bounds'] = np.nan
        self.alerts_df['range_based_f1_score@k_outlier_bounds'] = np.nan

        for idx, row in self.alerts_df.iterrows():
            num_true_regions_at_alert_time = len(self.regions_df[self.regions_df['unix_start'] <= row['alert_time']].index)

            real_ranges = []
            predicted_ranges = []
            predicted_ranges_outlier_bounds = []

            # Get real ranges up to row['alert_time']
            real_ranges_at_alert_time = self.regions_df[self.regions_df['unix_start'] < row['alert_time']]
            for _, real_range_row in real_ranges_at_alert_time.iterrows():
                real_range = [real_range_row['unix_start'], min(real_range_row['unix_end'], row['alert_time'])]
                real_ranges.append(real_range)
            
            # Get the top-`num_true_regions_at_alert_time` clusters and their regions
            clusters_df = self.clusters_df_list[self.df_index_to_list_index[int(row['alert_idx'])]]
            # clusters_df = pd.read_pickle(clusters_df_path)
            clusters_df['rank'] = clusters_df['area'].rank(ascending=False)
            clusters_df = clusters_df.sort_values(by='area', ascending=False).head(num_true_regions_at_alert_time)

            for _, predicted_range_row  in clusters_df.iterrows():
                predicted_range = [predicted_range_row['start'], min(predicted_range_row['end'], row['alert_time'])]
                predicted_range_outlier_bound = [predicted_range_row['point_start'], min(predicted_range_row['point_end'], row['alert_time'])]
                predicted_ranges.append(predicted_range)
                predicted_ranges_outlier_bounds.append(predicted_range_outlier_bound)
            
            precision = range_based_precision(real_ranges, predicted_ranges)
            recall = range_based_recall(real_ranges, predicted_ranges)
            f1_score = compute_f1_score(precision, recall) 
            
            precision_outlier_bounds = range_based_precision(real_ranges, predicted_ranges_outlier_bounds)
            recall_precision_outlier_bounds = range_based_recall(real_ranges, predicted_ranges_outlier_bounds)
            f1_score_outlier_bounds = compute_f1_score(precision_outlier_bounds, recall_precision_outlier_bounds) 

            self.alerts_df.loc[idx, 'range_based_precision@k'] = precision
            self.alerts_df.loc[idx, 'range_based_recall@k'] = recall
            self.alerts_df.loc[idx, 'range_based_f1_score@k'] = f1_score

            self.alerts_df.loc[idx, 'range_based_precision@k_outlier_bounds'] = precision_outlier_bounds
            self.alerts_df.loc[idx, 'range_based_recall@k_outlier_bounds'] = recall_precision_outlier_bounds
            self.alerts_df.loc[idx, 'range_based_f1_score@k_outlier_bounds'] = f1_score_outlier_bounds
            
        return self.alerts_df






def update_regions_df(regions_df, df_full):
    '''
    Updates the regions_df dataframe by adding `unix_start` and `unix_end` columns as well as adding a `score` for each region
    '''

    df_full = df_full.sort_values(by='unix_time')

    # Add unix time to the regions_df
    regions_df['unix_start'] = df_full.loc[regions_df['start']]['unix_time'].values
    regions_df['unix_end'] = df_full.loc[regions_df['end']]['unix_time'].values

    # Add a severity score for each region
    # The severity score is the sum of the absolute difference between the `measure` and the `signal` in a given region
    regions_df['score'] = np.nan   
    for idx, row in regions_df.iterrows():
        start, end = row['start'], row['end']
        deltas = np.abs(df_full.loc[start:end]['measure'] - df_full.loc[start:end]['signal'])
        score = deltas.sum()
        regions_df.loc[idx, 'score'] = score

    return regions_df


def get_alerts_df(dir_paths, base_path, k=10, regions_df=None, adjusted_local_minima=False):
    '''
    Returns
    -------
    Returns a dataframe with all top-k alerts raised while streaming

    If `regions_df` is specified then `k` is adjusted dynamically based on the true regions in regions_df 
    '''

    # Get a list of all the diretories to loop through
    # dir_paths.remove(base_path)
    dir_paths.sort(key=natural_keys)

    # Dataframe that corresponds to all the alerts that were sent out while streaming
    alerts_df = pd.DataFrame(columns=['alert_time', 'cluster_start', 'cluster_point_start', 'alert_idx', 'cluster_start_idx'])

    # For each consecutive execution check the found clusters and if they make it into the top k
    for dir_path in tqdm(dir_paths):
        if adjusted_local_minima:
            clusters_df_path = dir_path + 'clusters_df_adj.pickle'
            df_path = dir_path + 'df_adj.pickle'
        else:
            clusters_df_path = dir_path + 'clusters_df.pickle'
            df_path = dir_path + 'df.pickle'

        clusters = pd.read_pickle(clusters_df_path)
        clusters['rank'] = clusters['area'].rank(ascending=False)

        last_cluster_id = clusters['cluster_id'].max() 
        last_cluster_rank = clusters[clusters['cluster_id'] == last_cluster_id]['rank'].values[0]

        df = pd.read_pickle(df_path)

        # Adjust k if `regions_df` is specified
        if regions_df is not None:
            new_k = len(regions_df[regions_df['unix_start'] <= df['unix_time'].max()].index)
            if k != new_k:
                k=new_k
  
        # Check if last cluster makes it into the top-k
        if last_cluster_rank <= k:
            last_cluster_start = clusters[clusters['cluster_id'] == last_cluster_id]['start'].values[0]
            last_cluster_point_start = clusters[clusters['cluster_id'] == last_cluster_id]['point_start'].values[0]
            last_cluster_start_idx = df.iloc[(df['unix_time']-last_cluster_start).abs().argsort()[:1]].index.values[0]

            alert_time = df['unix_time'].max()
            if alerts_df.empty:
                alerts_df.loc[len(alerts_df.index)] = [alert_time, last_cluster_start, last_cluster_point_start, len(df.index), last_cluster_start_idx]
            else:
                # Check if there is already an alert after the current cluster_start
                latest_recorded_alert = alerts_df['alert_time'].max()
                if last_cluster_start > latest_recorded_alert:
                    alerts_df.loc[len(alerts_df.index)] = [alert_time, last_cluster_start, last_cluster_point_start, len(df.index), last_cluster_start_idx]
    
    return alerts_df

def get_true_regions_coverage(alerts_df, regions_df):
    '''
    Returns an updated alerts_df that denotes the true region(s) covered by an alert as well as if the alert happened during that region 
    '''
    # Process alerts_df to check for accuracy and what region it corresponds to if it is correct
    alerts_df['region_id_covered'] = np.nan
    alerts_df['alert_in_region'] = False
    for index, alert in alerts_df.iterrows():
        for _, region in regions_df.iterrows():
            # Check if the alert['alert_time'] happens during the interval of a true region
            if (alert['alert_time'] >= region['unix_start'] and alert['alert_time'] <= region['unix_end']):
                alerts_df.loc[index, 'alert_in_region'] = True

            # Check if there is a true region that has an overlap with the interval [row['cluster_start'], row['alert_time']]
            if (region['unix_start'] <= alert['cluster_start'] <= region['unix_end']) or (alert['cluster_start'] <= region['unix_start'] <= alert['alert_time']):
                alerts_df.loc[index, 'region_id_covered'] = int(region['region_id'])

    return alerts_df

def get_region_id_covered_by_cluster(clusters_df, regions_df):
    '''
    Update the clusters_df with a new column named `region_id_covered` that specifies the region_id 
    (if any) covered by the current cluster
    '''
    clusters_df['region_id_covered'] = np.nan
    for index, cluster in clusters_df.iterrows():
        for _, region in regions_df.iterrows():
            # Check if there is a true region that has an overlap with the interval [row['cluster_start'], row['alert_time']]
            if (region['unix_start'] <= cluster['start'] <= region['unix_end']) or (cluster['start'] <= region['unix_start'] <= cluster['end']):
                clusters_df.loc[index, 'region_id_covered'] = int(region['region_id'])

    return clusters_df

def update_alerts_df_with_ndcg_scores(alerts_df, regions_df, scores_dir, adjusted_local_minima=False):
    '''
    Updates the alerts_df dataframe with a new column `ndcg@k_at_alert_time` that specifies the  
    ranking quality of the detected alerts to the true regions.

    Notice that k is specified as the number of true regions at the time of an alert
    '''
    alerts_df['ndcg@k_at_alert_time'] = np.nan
    for idx, row in alerts_df.iterrows():
        regions_at_alert_df = regions_df[regions_df['unix_start'] <= row['alert_time']]
        num_true_regions_at_alert_time = len(regions_at_alert_df.index)
        
        # Get top-k clusters at row['alert_idx'] time
        if adjusted_local_minima:
            clusters_df_path = scores_dir + 'size_'+str(int(row['alert_idx']))+'/clusters_df_adj.pickle'
        else:
            clusters_df_path = scores_dir + 'size_'+str(int(row['alert_idx']))+'/clusters_df.pickle'
        clusters_df = pd.read_pickle(clusters_df_path)
        clusters_df['rank'] = clusters_df['area'].rank(ascending=False)
        clusters_df = clusters_df.sort_values(by='area', ascending=False).head(num_true_regions_at_alert_time)
        clusters_df = get_region_id_covered_by_cluster(clusters_df, regions_df)

        region_id_to_area = dict(zip(regions_at_alert_df['region_id'], regions_at_alert_df['score']))
        region_id_to_predicted_score = {region_id:0 for region_id in region_id_to_area}
               
        # Populate the region_id_to_predicted_score using the clusters_df
        for _, cluster in clusters_df.iterrows():
            if cluster['region_id_covered'] >= 0:
                score = regions_df[regions_df['region_id'] == cluster['region_id_covered']]['score'].tolist()[0]
                region_id_to_predicted_score[cluster['region_id_covered']] = score

        gt_relevance = list(region_id_to_area.values())
        predicted_relevance = list(region_id_to_predicted_score.values())

        ndcg = ndcg_score(np.array([gt_relevance]), np.array([predicted_relevance]))
        alerts_df.loc[idx, 'ndcg@k_at_alert_time'] = ndcg
    
    return alerts_df

def update_alerts_df_with_eval_scores(alerts_df, regions_df, scores_dir, adjusted_local_minima=False):
    '''
    Updates the alerts_df by computing the range based precision, recall and f1-scores @k 
    of the raised alerts so far.

    This also computes the range based precision, recall and f1-score variants with outlier bounds 
    (i.e., regions specified from alerts_df['cluster_point] starting from )
    '''
    alerts_df['range_based_precision@k'] = np.nan
    alerts_df['range_based_precision@k'] = np.nan
    alerts_df['range_based_f1_score@k'] = np.nan

    alerts_df['range_based_precision@k_outlier_bounds'] = np.nan
    alerts_df['range_based_precision@k_outlier_bounds'] = np.nan
    alerts_df['range_based_f1_score@k_outlier_bounds'] = np.nan

    for idx, row in alerts_df.iterrows():
        num_true_regions_at_alert_time = len(regions_df[regions_df['unix_start'] <= row['alert_time']].index)

        real_ranges = []
        predicted_ranges = []
        predicted_ranges_outlier_bounds = []

        # Get real ranges up to row['alert_time']
        real_ranges_at_alert_time = regions_df[regions_df['unix_start'] < row['alert_time']]
        for _, real_range_row in real_ranges_at_alert_time.iterrows():
            real_range = [real_range_row['unix_start'], min(real_range_row['unix_end'], row['alert_time'])]
            real_ranges.append(real_range)
        
        # Get the top-`num_true_regions_at_alert_time` clusters and their regions
        if adjusted_local_minima:
            clusters_df_path = scores_dir + 'size_'+str(int(row['alert_idx']))+'/clusters_df_adj.pickle'
        else:
            clusters_df_path = scores_dir + 'size_'+str(int(row['alert_idx']))+'/clusters_df.pickle'
        clusters_df = pd.read_pickle(clusters_df_path)
        clusters_df['rank'] = clusters_df['area'].rank(ascending=False)
        clusters_df = clusters_df.sort_values(by='area', ascending=False).head(num_true_regions_at_alert_time)

        for _, predicted_range_row  in clusters_df.iterrows():
            predicted_range = [predicted_range_row['start'], min(predicted_range_row['end'], row['alert_time'])]
            predicted_range_outlier_bound = [predicted_range_row['point_start'], min(predicted_range_row['point_end'], row['alert_time'])]
            predicted_ranges.append(predicted_range)
            predicted_ranges_outlier_bounds.append(predicted_range_outlier_bound)
        
        precision = range_based_precision(real_ranges, predicted_ranges)
        recall = range_based_recall(real_ranges, predicted_ranges)
        f1_score = compute_f1_score(precision, recall) 
        
        precision_outlier_bounds = range_based_precision(real_ranges, predicted_ranges_outlier_bounds)
        recall_precision_outlier_bounds = range_based_recall(real_ranges, predicted_ranges_outlier_bounds)
        f1_score_outlier_bounds = compute_f1_score(precision_outlier_bounds, recall_precision_outlier_bounds) 

        alerts_df.loc[idx, 'range_based_precision@k'] = precision
        alerts_df.loc[idx, 'range_based_recall@k'] = recall
        alerts_df.loc[idx, 'range_based_f1_score@k'] = f1_score

        alerts_df.loc[idx, 'range_based_precision@k_outlier_bounds'] = precision_outlier_bounds
        alerts_df.loc[idx, 'range_based_recall@k_outlier_bounds'] = recall_precision_outlier_bounds
        alerts_df.loc[idx, 'range_based_f1_score@k_outlier_bounds'] = f1_score_outlier_bounds
        
    return alerts_df