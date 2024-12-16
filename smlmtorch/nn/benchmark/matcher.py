"""
GPT4 prompt:
given two tensors of XYZ coordinates of different length, 
create a matching class that does greedy hungarian matching based on shortest distance, 
in pytorch using cuda

NOTE: no modifications!
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

class GreedyHungarianMatcher:
    def __init__(self, coords1, coords2):
        self.coords1 = coords1.cpu().numpy()
        self.coords2 = coords2.cpu().numpy()

    def pairwise_distances(self):
        c1 = self.coords1[:,None]
        c2 = self.coords2[None,:]
        return np.sqrt(((c1 - c2) ** 2).sum(-1))

    def match(self, threshold=None):
        distances = self.pairwise_distances()
        row_indices, col_indices = linear_sum_assignment(distances)

        L = min(len(row_indices), len(col_indices))
        pairs = np.array([row_indices[:L], col_indices[:L]]).T
        pair_dist = np.sqrt(np.sum( (self.coords1[pairs[:, 0]] - self.coords2[pairs[:, 1]])**2, -1 ))

        return pairs[pair_dist < threshold] if threshold is not None else pairs

# Example usage
if __name__ == "__main__":
    coords1 = np.array([
        [1.0, 2.0, 3.0],
        [7.0, 8.0, 9.0],
        [4.0, 5.0, 6.0],
    ])

    coords2 = np.array([
        [1.5, 2.5, 3.5],
        [4.5, 5.5, 6.5],
    ])

    matcher = GreedyHungarianMatcher(coords1, coords2.numpy)
    matched_pairs = matcher.match()

    print("Matched Pairs:", matched_pairs)
