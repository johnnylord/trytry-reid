import numpy as np
import torch


def compute_AP_CMC(indexes, target_indexes, invalid_indexes):
    """Compute ap(average precision) and CMC(cumulative matching characteristic)

    Args:
        indexes (list): list of ranked index
        target_indexes (list): list of target index
        invalid_indexes (list): list of invalid index
    """
    ap = 0.0
    cmc = np.zeros(len(indexes))

    if not len(target_indexes):
        return False, ap, cmc

    # Remove invalid indexes from indexes
    mask = np.in1d(indexes, invalid_indexes, invert=True)
    indexes = indexes[mask]

    # Find matching index
    mask = np.in1d(indexes, target_indexes)
    match_indexes = np.argwhere(mask==True)
    match_indexes = match_indexes.flatten()

    # Copmute ap and cmc
    cmc[match_indexes[0]:] = 1.

    n_targets = len(target_indexes)
    for i in range(1, n_targets+1):
        d_recall = 1. / n_targets
        precision = float(i) / (match_indexes[i-1]+1)

        if match_indexes[i-1] != 0:
            old_precision = float(i-1) / match_indexes[i-1]
        else:
            old_precision = 1.0

        ap = ap + d_recall*(old_precision+precision)/2

    return True, ap, cmc
