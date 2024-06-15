import numpy as np
import torch 

def WIS():
    pass

def PICPS(alphas: list[float], intervals: dict, preds: np.ndarray):
    """ PI Coverage Probability Score. Proportion of observations within interval.
    This implementation can handle multiple intervals and returns the average
    absolute difference between proportion and nominal interval level

    Lower is better, as it indicates closer matching of expected and actual coverages
    
    Args:
    alphas [list]: list of confidence levels 
    intervals [dict]: dictionary of upper and lower bounds per confidence level
    preds [np.ndarray]: array of predictions

    Returns:
    [float]: PICP score
    """
    ci_scores = np.zeros((len(alphas))) # length is known so no need for slower, variable-length list
    for i, a in enumerate(alphas): # iterate over confidence levels
        lower, upper = intervals[a] # make tuple of lower and upper bounds
        assert lower.shape[0] == preds.shape[0], "Length of lower bounds needs to match length of predictions"
        assert upper.shape[0] == preds.shape[0], "Length of upper bounds needs to match length of predictions"
        ci_scores[i] = abs(np.mean((preds >= lower) & (preds <= upper)) - a)
    score = np.mean(ci_scores)
    print(f"PICPS: {score}")
    return score

def PINAW(alphas: list[float], minmaxes: dict, intervals: dict):
    """ PI Normalized Average Width. Average width of interval, smaller interval
    for given coverage probability is desired

    Args:
    alphas [list]: list of confidence levels 
    intervals [dict]: dictionary of upper and lower bounds per confidence level
    minmaxes [dict]: dictionary of minimum and maximum values (range) of sampled values per test point

    Returns:
    [float]: PINAW score
    """
    # per observation have interval range, as well as min and max
    pinaw_scores = np.zeros((len(alphas)))
    for i, a in enumerate(alphas):
        lower, upper = intervals[a]
        mins, maxes = minmaxes[a]
        pinaw_scores[i] = np.mean((upper-lower)/(maxes-mins))
    return np.mean(pinaw_scores)


def CWC(alphas: list[float], minmaxes: dict, intervals: dict, preds: np.ndarray, eta: float = 2):
    """ Coverage Width Criterion. Average width of interval with exponential penalty if coverage is not met

    Args:
    alphas [list]: list of confidence levels 
    intervals [dict]: dictionary of upper and lower bounds per confidence level
    minmaxes [dict]: dictionary of minimum and maximum values (range) of sampled values per test point
    preds [np.ndarray]: predictions (mean values)
    eta [float]: penalization strength

    Returns:
    [float]: CWC score
    """
    exp_scores = np.zeros((len(alphas)))
    for i, a in enumerate(alphas):
        lower, upper = intervals[a]
        picp = np.mean((preds >= lower) & (preds <= upper))
        if picp < a:
            exp_scores[i] = np.exp(-eta * (picp - a))
    score = PINAW(alphas, minmaxes, intervals) + np.mean(exp_scores)
    print(f"CWC: {score}")
    return score
