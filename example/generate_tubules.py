# -*- coding: utf-8 -*-
"""
Created on Thu May 12 12:17:22 2022

@author: jelmer
"""
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def generate_microtubule_points(width, depth, numtubules, 
                                     linedensity,margin=0.1,plot=True,
                                     spl_knots=4, straight_ish=False, nudge_factor=0.1):
    # generate a bunch of microtubules
    from scipy.interpolate import InterpolatedUnivariateSpline
    
    ptslist=[]    

    for n in range(numtubules):

        if straight_ish:
            spl_ends = np.random.uniform([width*margin,width*margin,0],
                                    [width*(1-margin), width*(1-margin), depth], size=(2,3))
            knots = np.zeros((spl_knots,3))
            knots[0]=spl_ends[0]
            knots[-1]=spl_ends[1]
            for i in range(1,spl_knots-1):
                knots[i] = (spl_ends[0] + (spl_ends[1]-spl_ends[0]) * i/spl_knots + 
                    (np.random.rand(3)-0.5)*np.array([width,width,depth]) * nudge_factor)
        else:
            knots = np.random.uniform([width*margin,width*margin,0],
                                  [width*(1-margin), width*(1-margin), depth], size=(spl_knots,3))
         
        t = np.linspace(0,1,len(knots))
        spl_x = InterpolatedUnivariateSpline(t, knots[:,0], k=2)
        spl_y = InterpolatedUnivariateSpline(t, knots[:,1], k=2)
        spl_z = InterpolatedUnivariateSpline(t, knots[:,2], k=2)

        pts = []
        pos = 0
        prev = np.array((spl_x(pos),spl_y(pos),spl_z(pos)))
        step = 0.005
        total=0
        goaldist = 1
        with tqdm.tqdm() as pb:
            while pos<1:
                pos += step
                cur = np.array((spl_x(pos),spl_y(pos),spl_z(pos)))
                dist = np.sqrt(np.sum( (prev-cur)** 2))
                prev=cur
                step *= goaldist/dist
                pts.append(cur)
                pb.update(1)
                pb.set_description(f"pos={pos:.3f}. step={step:.3f}")
                total+=dist
                
        pts = np.array(pts)

        t = np.linspace(0,1,len(pts))
        numpts = int(linedensity * total)
        smp = np.zeros((numpts,3))
        rndt = np.linspace(0,1,numpts) #np.sort(np.random.uniform(0,1,size=numpts))
        for ax in np.arange(3):
            spl = InterpolatedUnivariateSpline(t,pts[:,ax], k=2)
            smp[:,ax] = spl(rndt)
                
        ptslist.append(smp)
        #ptslist.append(pts)

    pts=np.concatenate(ptslist)

    if plot:
        fig,ax=plt.subplots(1,2)
        ax[0].scatter(pts[:,0],pts[:,1], s=1)
        ax[0].set_xlim([0,width])
        ax[0].set_ylim([0,width])
        ax[0].set_title('Simulated tubules ground truth (XY)')

        ax[1].scatter(pts[:,0],pts[:,2], s=1)
        ax[1].set_xlim([0,width])
        ax[1].set_ylim([0,depth])
        ax[1].set_title('Simulated tubules ground truth (XZ)')
        
    return pts

