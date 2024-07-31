import numpy as np
import torch 
from torch.utils.data import Subset, DataLoader

def IS(levels: list[float], intervals: dict, y: np.ndarray): #Y NOT PREDS
    """ Weighted Interval Score. Decomposition of sharpness and penalties for 
    over- and underprediction. Lower is better.
    
    Args:
    alphas [list]: list of confidence levels 
    intervals [dict]: dictionary of upper and lower bounds per confidence level
    preds [np.ndarray]: array of predictions

    Returns:
    [float]: PICP score
    """
    is_scores = np.zeros((len(levels)))
    for i, l in enumerate(levels):
        lower, upper = intervals[l]
        assert lower.shape[0] == y.shape[0], "Length of lower bounds needs to match length of predictions"
        assert upper.shape[0] == y.shape[0], "Length of upper bounds needs to match length of predictions"
        under_mask, over_mask = y <= lower, y >= upper
        under_pen = 2/(1-l)*np.sum(lower[under_mask] - y[under_mask])/len(y) if not np.all(under_mask == False) else 0
        over_pen = 2/(1-l)*np.sum(y[over_mask] - upper[over_mask])/len(y) if not np.all(over_mask == False) else 0
        is_scores[i] = np.mean(upper - lower) + under_pen + over_pen
    score = np.mean(is_scores) # this is average interval score
    print(f"IS: {score}")
    return score

def WIS(levels: list[float], intervals: dict, y: np.ndarray, pred_med: np.ndarray):
    """ Weighted Interval Score. Requires predictive median as well. """
    wis_scores = np.zeros((len(levels)))
    for i, l in enumerate(levels):
        lower, upper = intervals[l]
        assert lower.shape[0] == y.shape[0], "Length of lower bounds needs to match length of predictions"
        assert upper.shape[0] == y.shape[0], "Length of upper bounds needs to match length of predictions"
        under_mask, over_mask = y <= lower, y >= upper
        under_pen = 2/(1-l)*np.sum(lower[under_mask] - y[under_mask])/len(y) if not np.all(under_mask == False) else 0
        over_pen = 2/(1-l)*np.sum(y[over_mask] - upper[over_mask])/len(y) if not np.all(over_mask == False) else 0
        wis_scores[i] = l/2 * (np.mean(upper - lower) + under_pen + over_pen) # include weight for final calculation
    score = 1/(len(levels)+0.5)*(0.5*np.mean(abs(y - pred_med)) + np.sum(wis_scores))
    print(f"WIS: {score}")
    return score

def coverages(levels: list[float], intervals:dict, y:np.ndarray):
    out = "Actual coverage per level |"
    for l in levels:
        lower, upper = intervals[l]
        out += f" {int(100*l)}%: {np.round(100*np.mean((y >= lower) & (y <= upper)), 2)} |"
    print(out)

def PICA(levels: list[float], intervals: dict, y: np.ndarray):
    """ PI Coverage Accuracy. Proportion of observations within interval.
    This implementation can handle multiple intervals and returns the average
    absolute difference between proportion and nominal interval level

    Lower is better, as it indicates closer matching of expected and actual coverages
    
    Args:
        :alphas: [list]: list of confidence levels 
        :intervals: [dict]: dictionary of upper and lower bounds per confidence level
        :preds: [np.ndarray]: array of predictions

    Returns:
        [float]: PICA score
    """
    ci_scores = np.zeros((len(levels))) # length is known so no need for slower, variable-length list
    for i, l in enumerate(levels): # iterate over confidence levels
        lower, upper = intervals[l] # make tuple of lower and upper bounds
        assert lower.shape[0] == y.shape[0], "Length of lower bounds needs to match length of predictions"
        assert upper.shape[0] == y.shape[0], "Length of upper bounds needs to match length of predictions"
        ci_scores[i] = abs(np.mean((y >= lower) & (y <= upper)) - l)
    score = np.mean(ci_scores)
    print(f"PICA: {score}")
    return score

def PINAW(levels: list[float], minmaxes: tuple, intervals: dict):
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
    pinaw_scores = np.zeros((len(levels)))
    mins, maxes = minmaxes
    for i, l in enumerate(levels):
        lower, upper = intervals[l]
        assert upper.shape[0] == lower.shape[0] == mins.shape[0] == maxes.shape[0], "Length of bounds and ranges needs to match"
        both_zero_mask = (mins == 0) & (maxes == 0)
        maxes[both_zero_mask] = 1
        pinaw_scores[i] = np.mean((upper-lower)/(maxes-mins))
    return np.mean(pinaw_scores)


def CWC(levels: list[float], minmaxes: tuple, intervals: dict, y: np.ndarray, eta: float = 2):
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
    exp_scores = np.zeros((len(levels)))
    for i, l in enumerate(levels):
        lower, upper = intervals[l]
        picp = np.mean((y >= lower) & (y <= upper))
        if picp < l:
            exp_scores[i] = np.exp(-eta * (picp - l))
    score = PINAW(levels, minmaxes, intervals) + np.sum(exp_scores)/len(y)
    print(f"CWC: {score}")
    return score

def evaluate_model(model, dataset, test_loader, test_batch_size, n_samples = 200, levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]):
    model.train()
    model = model.to("cpu")
    preds = np.zeros((test_batch_size, n_samples))
    mat, y = next(iter(test_loader))
    mat, y = mat.to("cpu"), y.to("cpu").numpy()
    preds = np.zeros((y.shape[0], n_samples))
    for i in range(n_samples):
        preds[:, i] = model(mat).sample().numpy()
    min_preds, max_preds = np.min(preds, axis=1), np.max(preds, axis=1)
    pred_median = np.quantile(preds, 0.5, axis=1)

    #print(len((min_preds == 0) and (max_preds == 0)))

    intervals_dict = {}
    for l in levels:
        intervals_dict[l] = (np.quantile(preds, (1-l)/2, 1), np.quantile(preds, (1+l)/2, 1))
    
    coverages(levels, intervals_dict, y)
    PICA(levels, intervals_dict, y)
    CWC(levels, (min_preds, max_preds), intervals_dict, y)
    WIS(levels, intervals_dict, y, pred_med=pred_median)
    IS(levels, intervals_dict, y)

    