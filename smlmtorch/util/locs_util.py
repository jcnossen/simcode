# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import KDTree
import tqdm


def indices_per_frame(frame_ix):
    if len(frame_ix) == 0: 
        numFrames = 0
    else:
        numFrames = np.max(frame_ix)+1
    frames = [[] for i in range(numFrames)]
    for k in range(len(frame_ix)):
        frames[frame_ix[k]].append(k)
    for f in range(numFrames):
        frames[f] = np.array(frames[f], dtype=int)
    return frames

def filter_loc_pairs(distances, pairs):
    if len(pairs) == 0:
        return distances, pairs
    maxspots = np.max(pairs[:,0])+1
    bestpairs = np.ones(maxspots,dtype=np.int32)*-1
    for i in range(len(pairs)):
        a,b=pairs[i]
        if bestpairs[a] < 0 or distances[bestpairs[a]] > distances[i]:
            bestpairs[a] = i
    distances = distances[bestpairs[bestpairs>=0]]
    return distances, pairs[bestpairs[bestpairs>=0]]


def find_pairs(pos1, pos2, distance_px):
    combined = np.concatenate([pos1,pos2])
    tree = KDTree(combined)
    pairs = tree.query_pairs(distance_px, output_type='ndarray')

    # Remove all pairs that are within the same frame
    set_ix = pairs < len(pos1)
    pairs = pairs[set_ix[:,0] != set_ix[:,1]]
    
    # remove locs that have multiple pairs, picking the closest one
    offsets = combined[pairs[:,0]]-combined[pairs[:,1]]

    distances = np.sqrt((offsets**2).sum(1))
    distances, pairs = filter_loc_pairs(distances, pairs)
    distances, pairs = filter_loc_pairs(distances, pairs[:,[1,0]])
    pairs = pairs[:,[1,0]]
    pairs[:,1] -= len(pos1)
    return pairs


def array_split_minimum_binsize(framenum, binsize, use_tqdm=False):
    """
    Array split but with a minimum binsize. Unless len(framenum)<binsize, 
    all bins will have at least binsize indices
    """
    fn_sorted_idx = np.argsort(framenum)
    fn_s = framenum[fn_sorted_idx]
    curbin = []#list[int]()
    bins=[]
    nframes = max(fn_s)+1
    
    if use_tqdm:
        r = tqdm.trange(nframes)
    else:
        r = range(nframes)
    
    i = 0 # i and j index into the fn_s
    for f in r:
        # find all spots in this frame
        j = i
        while j<len(fn_s) and i<len(fn_s) and fn_s[j] == fn_s[i]:
            curbin.append(fn_sorted_idx[j])
            j += 1
            
        i=j

        if len(curbin) >= binsize:
            # advance frame
            bins.append(curbin)
            curbin = []
            
    if len(curbin)>0:
        bins.append(curbin)
        
    if len(bins[-1]) < binsize and len(bins)>1: #merge last bins if needed
        last = bins.pop()
        for i in last: bins[-1].append(i)
    return bins


if __name__ == '__main__':
    
    ix = np.random.randint(0, 10, size=100)
    bins = array_split_minimum_binsize(ix, 4)

    print(bins)
    print([ix[b] for b in bins])
    
    