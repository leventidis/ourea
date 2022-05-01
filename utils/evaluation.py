import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.core.fromnumeric import sort
import numpy as np
import pandas as pd
sns.set(rc={'figure.figsize':(20, 11)}, font_scale=2)

def get_top_k_eval_df(output_dir, raw, raw_voting, combined_score, cluster_w=[], cluster_area=[], cluster_outliers_area=[]):
    '''
    Save all the data shown in the top-k evaluation figure into a dataframe called 'top_k_eval_df'
    '''
    max_len = max(len(raw), len(raw_voting), len(combined_score), len(cluster_w), len(cluster_area), len(cluster_outliers_area))

    # Pad np.nan for the smaller lists
    raw += [np.nan] * (max_len - len(raw))
    raw_voting += [np.nan] * (max_len - len(raw_voting))
    combined_score += [np.nan] * (max_len - len(combined_score))
    cluster_w += [np.nan] * (max_len - len(cluster_w))
    cluster_area += [np.nan] * (max_len - len(cluster_area))
    cluster_outliers_area += [np.nan] * (max_len - len(cluster_outliers_area))

    k_vals = range(1, max_len+1)

    data = {
        'k': k_vals, 'raw': raw, 'raw_voting': raw_voting, 'combined_score': combined_score,
        'cluster_w': cluster_w, 'cluster_area': cluster_area, 'cluster_outliers_area': cluster_outliers_area}
    df = pd.DataFrame.from_dict(data)
   
    # Save the dataframe
    df.to_pickle(output_dir+"top_k_eval_df.pickle")


def outlier_clustering_evaluation(df, sort_mode='weighted_score'):
    '''
    sort_mode (str): Must be one of ['weighted_score', 'area', 'area_pts']


    The dataframe with a cluster ID specified for each timestamp

    Returns a list with the ground truth number of regions detected in a top-k fashion
    '''

    # Extract all detected outliers across all views and sort it based on the specified sort_mode 

    if sort_mode == 'weighted_score':
        df_filtered = df.copy()[df['raw_voting_score']>0].sort_values(by=['weighted_score'], ascending=False)
    elif sort_mode == 'area':
        df_filtered = df.copy()[df['raw_voting_score']>0].sort_values(by=['area', 'weighted_score'], ascending=False)
    elif sort_mode == 'area_pts':
        df_filtered = df.copy()[df['raw_voting_score']>0].sort_values(by=['area_pts', 'weighted_score'], ascending=False)

    used_clusters = set()
    top_k_row_ids = []
    # We only add an element whenever we discover a new cluster
    for index, row in df_filtered.iterrows():
        # Ensure the point maps to a cluster and we haven't used it already
        if (row['cluster_id'] >= 0 and row['cluster_id'] not in used_clusters):
            used_clusters.add(row['cluster_id'])
            top_k_row_ids.append(index)

    retrieved_rows_df = df_filtered.loc[top_k_row_ids]
    num_regions_using_clustering = []
    [num_regions_using_clustering.append(retrieved_rows_df[:i+1]['region_id'].nunique()) for i in range(len(retrieved_rows_df.index))]

    return num_regions_using_clustering

def region_top_k_plot(df, output_dir, using_clustering=True, k=30):
    '''
    Create a figure of the top-k versus number of unique regions covered for each ranking

    If `df_with_clusters` is specified then also use the ranking from the identified clusters
    '''
    # Extract ordered dataframes for each ranking to be evaluated
    df_raw = df[df['is_outlier_raw_data_fit'] == 1].sort_values(by=['abs_residual'], ascending=False)
    df_raw_voting = df[df['raw_voting_score']>0].sort_values(by=['raw_voting_score', 'norm_residual'], ascending=False)
    df_combined_score = df[df['raw_voting_score']>0].sort_values(by=['weighted_score'], ascending=False)

    raw_num_regions = []
    raw_voting_num_regions = []
    combined_score_num_regions = []

    # Compute the number of regions covered by each ranking 
    [raw_num_regions.append(df_raw[:i+1]['region_id'].nunique()) for i in range(len(df_raw.index))]
    [raw_voting_num_regions.append(df_raw_voting[:i+1]['region_id'].nunique()) for i in range(len(df_raw_voting.index))]
    [combined_score_num_regions.append(df_combined_score[:i+1]['region_id'].nunique()) for i in range(len(df_combined_score.index))]

    # Perform evaluation with clustering if specified
    if using_clustering:
        clustering_score_weighted_num_regions = outlier_clustering_evaluation(df, sort_mode='weighted_score')
        clustering_area_num_regions = outlier_clustering_evaluation(df, sort_mode='area')
        clustering_area_pts_num_regions = outlier_clustering_evaluation(df, sort_mode='area_pts')
    
    k_vals = range(1, k+1)

    # Draw line graph
    plt.plot(k_vals, raw_num_regions[:k], label='raw')
    plt.plot(k_vals, raw_voting_num_regions[:k], label='raw_voting')
    plt.plot(k_vals, combined_score_num_regions[:k], label='combined_score')

    if using_clustering:
        cluster_k=len(clustering_score_weighted_num_regions)
        print(cluster_k, 'clusters were detected by KDE')
        cluster_k_vals = range(1, cluster_k+1)
        plt.plot(cluster_k_vals, clustering_score_weighted_num_regions[:cluster_k], label='clustering (weighted scores)')
        plt.plot(cluster_k_vals, clustering_area_num_regions[:cluster_k], label='clustering (area)')
        plt.plot(cluster_k_vals, clustering_area_pts_num_regions[:cluster_k], label='clustering (outliers area)')

    plt.ylabel('Number of Unique Regions Covered');plt.xlabel('Top-k');plt.legend();plt.tight_layout()
    plt.savefig(output_dir+'region_evaluation_unique_regions.svg')
    plt.clf()

    # Save the data for the figure in a dataframe
    get_top_k_eval_df(
        output_dir=output_dir, raw=raw_num_regions, raw_voting=raw_voting_num_regions, combined_score=combined_score_num_regions,
        cluster_w=clustering_score_weighted_num_regions, cluster_area=clustering_area_num_regions, cluster_outliers_area=clustering_area_pts_num_regions
    )

def region_evaluation(df, regions_df, output_dir, using_clustering=True, k=30):
    raw_detection_delays=[]
    multi_level_delays=[]

    raw_missed_regions=0
    multi_level_missed_regions=0

    for _, row in regions_df.iterrows():

        raw_miss=True
        multi_level_miss=True

        # Raw Model Evaluation
        for i in range(row['start'], row['start']+row['length']):
            if df.loc[i]['is_outlier_raw_data_fit']:
                raw_miss=False
                raw_detection_delays.append(i-row['start'])
                break

        # Multi-Level Evaluation
        for i in range(row['start'], row['start']+row['length']):
            if df.loc[i]['raw_voting_score']:
                multi_level_miss=False
                multi_level_delays.append(i-row['start'])
                break

        if raw_miss:
            raw_missed_regions+=1
        if multi_level_miss:
            multi_level_missed_regions+=1

    # Draw Histogram and Boxplots for the detected delays between raw and multi level
    plt.hist([raw_detection_delays, multi_level_delays], bins=24, alpha=0.5, label=['raw', 'multi level'])
    plt.xlabel('Detection Delay');plt.ylabel('Frequency');plt.legend();plt.tight_layout()
    plt.savefig(output_dir+'region_evaluation_histogram.svg')
    plt.clf()

    plt.boxplot([raw_detection_delays, multi_level_delays], labels=['raw','multi level'])
    plt.ylabel('Detection Delay');plt.tight_layout()
    plt.savefig(output_dir+'region_evaluation_boxplot.svg')
    plt.clf()

    # Evaluate the number of distinct regions covered in the top-k
    region_top_k_plot(df=df, output_dir=output_dir, using_clustering=using_clustering, k=k)

    print("Raw Model has missed", raw_missed_regions, 'regions and has an average detection delay of',
        statistics.mean(raw_detection_delays), 'with a standard deviation of', statistics.stdev(raw_detection_delays))
    print("Multi-Level Model has missed", multi_level_missed_regions, 'regions and has an average detection delay of',
        statistics.mean(multi_level_delays), 'with a standard deviation of', statistics.stdev(multi_level_delays))