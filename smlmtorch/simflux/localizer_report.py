# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:29:33 2022

@author: jelmer
"""

import numpy as np
import time
import tqdm
import os
import matplotlib.pyplot as plt
from .localizer import SFDataset, SFLocalizer
import smlmtorch.simflux.pattern_estimator as pe 
import torch
from scipy.interpolate import InterpolatedUnivariateSpline
from smlmtorch.util.progbar import pb_range

from smlmtorch.simflux.pattern_estimator import simple_sine_fit

class LocalizationReporter:
    
    def __init__(self, ds: SFDataset, result_dir, mp: pe.ModulationPattern = None):
        self.mp = mp
        self.ds = ds
        self.result_dir = result_dir

    def plot_sigma_vs_time(self):
        numframes = np.max(framenum)+1
        ds.data.frame = np.maximum((ds.data.frame / sigmaFramesPerBin - 0.5).astype(np.int32),0)
        frames = ds.indicesPerFrame()
        self.medianSigma = np.array([np.median(ds.data.estim.sigma[idx],0) for idx in frames])
        self.sigma_t = (0.5+np.arange(len(frames))) * sigmaFramesPerBin
        
        #self.medianSigma = [self.medianSigma[0], *self.medianSigma, self.medianSigma[-1]]
            
        self.sigma_t[0] = 0
        self.sigma_t[-1] = (len(frames)-1) * sigmaFramesPerBin
        spl_x = InterpolatedUnivariateSpline(self.sigma_t, self.medianSigma[:,0], k=2)
        spl_y = InterpolatedUnivariateSpline(self.sigma_t, self.medianSigma[:,1], k=2)
        
        self.sigma = np.zeros((numframes,2))
        self.sigma[:,0] = spl_x(np.arange(numframes))
        self.sigma[:,1] = spl_y(np.arange(numframes))
        

    def draw_patterns(self, dims, **kwargs):
        for ep in pb_range(len(self.mp.mod), desc='Generating modulation pattern plots'):
            self.draw_pattern(ep, dims, **kwargs)

    def draw_pattern(self, ep, dims, me_threshold=0.2, label=None, numpts=2000, axes=None):
        ds = self.ds
        #ds = ds[ds.frame % 6 == 0]

        # should use all points later
        mod = self.mp.mod[ep]
        k = mod['k'][:dims]

        sel = np.arange(len(ds))
        np.random.shuffle(sel)
        sel = sel[:np.minimum(numpts, len(sel))]

        accepted = self.mp.mod_error(ds[sel]) < me_threshold
        rejected = np.logical_not(accepted)

        e = (ep - ds.frame[sel]) % self.mp.pattern_frames.size
        mod_phases = self.mp.mod_at_frame(ds.frame[sel], frame_window=self.mp.pattern_frames.size)['phase']
        spot_phases = mod_phases[np.arange(len(sel)), (ep-ds.frame[sel]) % self.mp.pattern_frames.size]
        I = ds.ibg[sel, e,0]

        normI = I / ds.ibg[sel][:, :,0].sum(-1)
        proj = (k[None] * ds.pos[sel][:, :dims]).sum(1) - spot_phases
        x = proj % (np.pi * 2)

        if axes is None:
            fig,axes = plt.subplots(figsize=(10,6))
        axes.scatter(x[accepted], normI[accepted], marker='.', c='b', label='Accepted')
        axes.scatter(x[rejected], normI[rejected], marker='.', c='r', label='Rejected')

        sigx = np.linspace(0, 2 * np.pi, 400)
        exc = mod['relint'] * (1 + mod['depth'] * np.sin(sigx))# - mod['phase']))
        axes.plot(sigx, exc, 'k', linewidth=4, label='Estimated P')

        axes.set_ylim([-0.01, 0.51])
        axes.set_xlabel('Phase [radians]')
        axes.set_ylabel(r'Normalized intensity ($I_k$)')
        lenk = np.sqrt(np.sum(k ** 2))
        label = f"{label}:" if label is not None else ""
        axes.set_title(f'{label} Pattern {ep}. K={lenk:.4f} ' + f" Phase={self.mp.mod[ep]['phase'] * 180 / np.pi:.3f}")
        #axes.colorbar()
        axes.legend()

        if self.result_dir is not None:
            plt.savefig(self.result_dir + f"pattern{ep}.png")

    def mod_over_fov(self, ds:SFDataset, nbins=10):
        nax = len(self.mp.pattern_frames)
        relint = np.zeros( (len(ds), nax) )
        ampl = np.zeros((len(ds),nax))
        
        for i, pf in enumerate(self.mp.pattern_frames):
            relint[:,i] = ds.ibg[:, pf, 0].sum(1)
            I = torch.tensor(ds.ibg[:,pf,0])
            ampl[:,i],_,_ = simple_sine_fit(I / I.sum(1,keepdim=True))

        relint /= relint.sum(1,keepdims=True)
        
        x_edges = np.linspace(0, ds.imgshape[1], nbins+1)
        y_edges = np.linspace(0, ds.imgshape[0], nbins+1)
        
        from scipy.stats import binned_statistic_2d
        
        relint_maps = np.zeros((nax,nbins,nbins))
        ampl_maps = np.zeros((nax,nbins,nbins))
        for i in range(nax):
            relint_maps[i] = binned_statistic_2d(ds.pos[:,0], ds.pos[:,1], 
                                                 statistic='median', values=relint[:,i], bins=[x_edges,y_edges])[0]
            ampl_maps[i] = binned_statistic_2d(ds.pos[:,0], ds.pos[:,1], 
                                               statistic='median', values=ampl[:,i], bins=[x_edges,y_edges])[0]

        count_map = binned_statistic_2d(ds.pos[:,0], ds.pos[:,1], values=None,
                            statistic='count', bins=[x_edges,y_edges])[0]
            
        return relint_maps, ampl_maps, count_map
        
    def plot_mod_fov(self, numpts=10000, pt_size=4, fov_map_bins=10):
        ds = self.ds
        relint_maps, ampl_maps, count_map = self.mod_over_fov(ds, nbins=fov_map_bins)

        relint = np.zeros( (len(ds), 2) )
        ampl = np.zeros((len(ds),2))
        
        for i, pf in enumerate(self.mp.pattern_frames):
            relint[:,i] = ds.ibg[:, pf, 0].sum(1)
            I = torch.tensor(ds.ibg[:,pf,0])
            ampl[:,i],_,_ = pe.simple_sine_fit(I / I.sum(1,keepdim=True))

        relint /= relint.sum(1,keepdims=True)
        
        plots = [ 
            (relint[:,0], 'Relative intensity (Ax0/Ax1)', 'relint', relint_maps[0]),
            (ampl[:,0], 'Modulation depth ax 0', 'depth0', ampl_maps[0]),
            (ampl[:,1], 'Modulation depth ax 1', 'depth1', ampl_maps[1])
        ]
        
        sel = np.arange(len(ds))
        np.random.shuffle(sel)
        sel = sel[:np.minimum(numpts, len(sel))]
        ds = ds[sel]

        for data, label, fn, fov_map in plots:        
            plt.figure()
            c=data[sel]
            c[c>1]=1
            plt.scatter(ds.pos[:,0],ds.pos[:,1], 
                        c=c, s=pt_size)
            plt.ylabel('Y Position [px]')
            plt.xlabel('X Position [px]')
            plt.colorbar(label=label)
            plt.title(f'{label} over FOV')
            plt.savefig(self.result_dir + f'fov_{fn}.png')
            
            plt.figure()
            plt.imshow(fov_map.T, interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.title(f'{label} over FOV')
            plt.ylabel('Y Position [px]')
            plt.xlabel('X Position [px]')
            plt.savefig(self.result_dir + f'fov_map_{fn}.png')
        
        plt.figure()
        plt.imshow(count_map.T, interpolation='nearest', origin='lower')
        plt.colorbar()
        plt.title(f'Loc. counts per bin over FOV (min={count_map.min():.0f})')
        plt.ylabel('Y Position [px]')
        plt.xlabel('X Position [px]')
        plt.savefig(self.result_dir + 'fov_counts_map.png')
            
    def scatterplot(self, datasets, ds_args=None,labels=None,
                    connected=False, # plot lines between the datasets
                    limits='imgshape', **kwargs):
        
        if connected:
            assert len(datasets[0]) == len(datasets[1])

        fig,ax=plt.subplots()
        
        if connected:
            from matplotlib.collections import LineCollection
            lines = [[(a[0],a[1]),(b[0],b[1])] for a,b in 
                     zip(datasets[0].pos, datasets[1].pos)]
            lc = LineCollection(lines, linewidth=1)
            ax.add_collection(lc)
        
        for i in range(len(datasets)):
            ds = datasets[i]
            plt.scatter(ds.pos[:,0], ds.pos[:,1], 
                        **(ds_args[i] if ds_args is not None else {}),
                        label=labels[i] if labels is not None else None, **kwargs)
        
        if type(limits) == np.ndarray or type(limits) == list:
            plt.xlim(limits[0])
            plt.ylim(limits[1])
            
        if limits == 'imgshape':
            plt.xlim([0, max([d.imgshape[1] for d in datasets])])
            plt.ylim([0, max([d.imgshape[0] for d in datasets])])
        else:
            ax.autoscale()
            ax.margins(0.1)

        
        if labels is not None:
            plt.legend()
        