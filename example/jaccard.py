
import tqdm
import numpy as np
from scipy.spatial import KDTree
from scipy.signal import convolve2d


def find_nearby(A : np.ndarray, B : np.ndarray, max_distance):
    tree_A = KDTree(A)
    tree_B = KDTree(B)

    dist, ix = tree_A.query(B)
    A_in_B = dist <= max_distance
    dist, ix = tree_B.query(A)
    B_in_A = dist <= max_distance
    return A_in_B, B_in_A

#for jaccard, recall, precision calculation
# low recall will indicate missing structures
# low precision will indicate many, false positives
# typically due to multiple localizations being detected as one, or less likely noise being detected as a localization

def compute_stats(locs : np.ndarray, gt : np.ndarray, max_distance):
    locs_tree = KDTree(locs)
    gt_tree = KDTree(gt)

    dist, ix = locs_tree.query(gt)
    recall = (dist <= max_distance).sum() / len(gt)

    dist, ix = gt_tree.query(locs)
    precision = (dist <= max_distance).sum() / len(locs)

    return recall, precision

def compute_jaccard_points(A : np.ndarray, B : np.ndarray, max_distance):
    """
    Compute the jaccard index, defined as the probability 
    that if i take a random point from the combined set (A+B), it is part of 
    both A and B.

    Which is: p(A) * p(B|A) + p(B) * p(A|B),
    where 
    p(A) = len(A)/(len(A)+len(B))
    p(B|A) = len(A_in_B) / len(A)
    """
    A_in_B, B_in_A = find_nearby(A, B, max_distance)
    
    total = len(A)+len(B)
    return ((len(A)/total) * A_in_B.sum() / len(A) + 
            (len(B)/total) * B_in_A.sum() / len(B))



def jaccard_timeline(ds_est, ds_gt, timebins = 10, zoom = 20, threshold_dist = 0.5, dims=2):

    nframes = ds_est.numFrames

    jaccard = np.zeros(timebins)

    maps = []
    for i in tqdm.trange(timebins):
        f = int((i+1) * nframes / timebins)
        pts = ds_est.pos[ds_est.frame < f, :2]
        jaccard[i] = compute_jaccard_points(ds_gt.pos[:,:dims], pts[:,:dims], threshold_dist)

    return jaccard
