from sklearn import preprocessing
import argparse
import re

def normalize_vector(vals):
    '''
    Given a list `vals` of numeric values. Return a normalized list that ranges between 0-1
    '''
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_vals = min_max_scaler.fit_transform(vals.reshape(-1, 1) )
    return scaled_vals

def range_limited_float_type(arg):
    '''
    Used by argparse to specify a float argument ranging from 0 top 1
    '''
    try:
        f = float(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < 0 or f > 1:
        raise argparse.ArgumentTypeError("Argument must be < 0 and > 1")
    return f

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def compute_f1_score(precision, recall):
    '''
    Returns the f1-score given a precision and recall measurement

    Ensures that the denominator is not zero. If the denominator is zero then return 0 as the f1-score  
    '''
    if (precision + recall) == 0:
        return 0
    else:
        return (2*precision*recall) / (precision + recall)