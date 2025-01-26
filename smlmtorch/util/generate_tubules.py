# -*- coding: utf-8 -*-
"""
Created on Thu May 12 12:17:22 2022

@author: jelmer
"""
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def generate_microtubule_points(width, depth, numtubules, linedensity, 
                              margin=0.1, spl_knots=4, spl_degree=2,
                              nudge_factor=0.1, stepsize_samples=200, return_knots=False):
    """
    Generate microtubule points with uniform spacing along splines.
    """
    from scipy.interpolate import InterpolatedUnivariateSpline
    import numpy as np

    assert spl_knots>spl_degree, "Spline degree should be smaller than number of knots"
    
    all_uniform_pts = []
    all_intermediate_pts = []
    all_knots = []
    
    while len(all_uniform_pts) < numtubules:
        # Generate endpoints within margins
        spl_ends = np.random.uniform(
            [width*margin, width*margin, 0],
            [width*(1-margin), width*(1-margin), depth], 
            size=(2,3)
        )
        
        # Generate knot points by linear interpolation with some extra nudge (mostly straight)
        knots = np.zeros((spl_knots, 3))
        knots[0] = spl_ends[0]
        knots[-1] = spl_ends[1]
        
        for i in range(1, spl_knots-1):
            base_point = spl_ends[0] + (spl_ends[1]-spl_ends[0]) * i/spl_knots
            move = (np.random.rand(3)-0.5) * np.array([width,width,depth]) * nudge_factor
            knots[i] = base_point + move
        
        t = np.linspace(0, 1, stepsize_samples)
        pts_intermediate = np.zeros((len(t), 3))
        for i in range(3):
            spl = InterpolatedUnivariateSpline(np.arange(spl_knots), knots[:,i], k=2)
            pts_intermediate[:,i] = spl(t * (spl_knots-1))
        
        # Calculate step sizes and total length
        stepsizes = np.linalg.norm(np.diff(pts_intermediate, axis=0), axis=1)
        total_length = np.sum(stepsizes)
        num_points = int(total_length * linedensity)
        
        if num_points < 10:  # Skip if too short
            continue
            
        cumulative_dist = np.cumsum(stepsizes)
        desired_distances = np.linspace(0, total_length, num_points)
        t_positions = np.linspace(0, 1, len(cumulative_dist))
        dist_to_t = InterpolatedUnivariateSpline(np.append(0, cumulative_dist), 
                                                np.append(0, t_positions), k=3)
        
        # Get uniformly spaced points
        new_t = dist_to_t(desired_distances)
        uniform_pts = np.zeros((num_points, 3))
        for dim in range(3):
            spl = InterpolatedUnivariateSpline(t, pts_intermediate[:,dim])
            uniform_pts[:,dim] = spl(new_t)
            
        all_uniform_pts.append(uniform_pts)
        all_intermediate_pts.append(pts_intermediate)
        all_knots.append(knots)
    
    # Stack all points and knots
    final_pts = np.vstack(all_uniform_pts)
    final_intermediate = np.vstack(all_intermediate_pts)
    final_knots = np.vstack(all_knots)

    if return_knots:
        return final_pts, final_intermediate, final_knots
    
    return final_pts


if __name__ == '__main__':
    np.random.seed(0)
    pts, pts_1, knots = generate_microtubule_points(20, 0, 
                linedensity=5, numtubules=10, spl_knots=4, spl_degree=3, return_knots=True)

    plt.figure()
    plt.scatter(pts[:,0],pts[:,1], s=0.4)
    #plt.scatter(pts_1[:,0],pts_1[:,1], s=4, c='r')
    plt.scatter(knots[:,0],knots[:,1], s=10, c='k')