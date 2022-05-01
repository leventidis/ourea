import math

def range_based_recall(real_ranges, predicted_ranges, alpha=0.5, delta_mode='flat'):
    '''
    Returns the range based recall given the predicted and groundtruth range-based anomalies
    '''
    recall_sum = 0
    for real_range in real_ranges:
        recall_sum += recall_per_real_range(real_range=real_range, predicted_ranges=predicted_ranges, alpha=alpha, delta_mode=delta_mode)
    
    return recall_sum / len(real_ranges)

def range_based_precision(real_ranges, predicted_ranges, delta_mode='flat'):
    '''
    Returns the range based precision given the predicted and groundtruth range-based anomalies
    '''
    precision_sum = 0
    for predicted_range in predicted_ranges:
        precision_sum += precision_per_predicted_range(real_ranges, predicted_range, delta_mode)
    return precision_sum / len(predicted_ranges)

def recall_per_real_range(real_range, predicted_ranges, alpha, delta_mode):
    return alpha*existence_reward(real_range, predicted_ranges) + (1-alpha)*overlap_reward(real_range, predicted_ranges, delta_mode)

def precision_per_predicted_range(real_ranges, predicted_range, delta_mode):
    total_score = 0
    card_factor = cardinality_factor(single_range=predicted_range, multiple_ranges=real_ranges)
    for real_range in real_ranges:
        overlap_range = get_overlap_range(real_range, predicted_range)
        total_score += omega(anomaly_range=predicted_range, overlap=overlap_range, delta_mode=delta_mode)
    return total_score * card_factor

def existence_reward(real_range, predicted_ranges):
    for predicted_range in predicted_ranges:
        if regions_overlap(real_range, predicted_range):
            return 1
    return 0


def overlap_reward(real_range, predicted_ranges, delta_mode):
    total_reward = 0
    card_factor = cardinality_factor(single_range=real_range, multiple_ranges=predicted_ranges)
    for pred_range in predicted_ranges:
        overlap_range = get_overlap_range(real_range, pred_range)
        total_reward += omega(anomaly_range=real_range, overlap=overlap_range, delta_mode=delta_mode)
    return total_reward * card_factor

def regions_overlap(reg1, reg2):
    '''
    Return True if there is an overlap between `reg1` and `reg2` otherwise return False
    '''
    if ( (reg1[0] <= reg2[1]) and (reg2[0] <= reg1[1]) ):
        return True
    else:
        False

def get_overlap_range(reg1, reg2):
    '''
    Return as a 2 element list the overlapping region between `reg1` and `reg2`.
    If there is no overlap then just return [0,0] as the overlapping region
    '''
    if (not regions_overlap(reg1, reg2)):
        return [0, 0]
    else:
        overlap_start = max(reg1[0], reg2[0])
        overlap_end = min(reg1[1], reg2[1])
        return [overlap_start, overlap_end]

def cardinality_factor(single_range, multiple_ranges):
    '''
    Returns 1 if `single_range` overlaps at most once with a range in `multiple_ranges`

    Otherwise return gamma(single_range, multiple_ranges) which is usually equivalent to 1/x 
    where x is the number of overlaps single_range has with multiple_ranges
    '''
    num_overlaps = 0
    for cur_range in multiple_ranges:
        if (regions_overlap(single_range, cur_range)):
            num_overlaps += 1
    
    if num_overlaps <= 1:
        return 1
    else:
        # The gamma function is currently set as 1/x where x is the number of overlaps
        return 1/num_overlaps

def omega(anomaly_range, overlap, delta_mode):
    if overlap == [0,0]:
        # If there is no overlap we return 0
        return 0
    
    my_value = 0
    max_value = 0

    # TODO: Split the anomaly_range to appropriate units (currently the anomaly range is defined in unix time and we split it to hour points)
    point_size = 3600
    anomaly_length = math.floor(((anomaly_range[1] - anomaly_range[0]) / point_size)) + 1
    anomaly_range_pts = []
    for i in range(anomaly_length):
        anomaly_range_pts.append(anomaly_range[0] + i*point_size)

    for i in range(anomaly_length):
        bias = delta(i, anomaly_length, mode=delta_mode)
        max_value += bias
        if (anomaly_range_pts[i] >= overlap[0] and anomaly_range_pts[i] <= overlap[1]):
            my_value += bias
    
    if max_value == 0:
        return 0
    return my_value / max_value

def delta(i, anomaly_length, mode):
    if mode == 'flat':
        return 1
    elif mode == 'front_end_bias':
        return anomaly_length - i + 1