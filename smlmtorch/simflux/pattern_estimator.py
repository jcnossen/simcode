# -*- coding: utf-8 -*-
"""
Find 2D SIM patterns in localization data, using pytorch to implement a fast DFT
@author: jelmer
"""
import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
import os
#os.chdir('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin')
os.environ['CUDA_PATH'] = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8'
import torch
import torch.fft
from scipy.interpolate import InterpolatedUnivariateSpline
from smlmtorch.simflux.dataset import SFDataset
from smlmtorch.smlm.dataset import Dataset
from torch import Tensor
import pickle
from smlmtorch.simflux.ndi_fit_torch import ndi_bg_fit, ndi_fit
from smlmtorch.util.locs_util import array_split_minimum_binsize
from smlmtorch.util.splines import CatmullRomSpline1D
from smlmtorch import config_dict
from smlmtorch.util.adaptive_gd import AdaptiveStepGradientDescent
from smlmtorch import config_dict
from smlmtorch import pb_range, progbar

ModulationType = np.dtype([('k', '<f4', (3,)), 
                            ('depth','<f4'),
                            ('phase','<f4'),
                            ('relint','<f4')
                            ])



class ModulationPattern:
    def __init__(self, pattern_frames, mod=None, moving_window = True, 
                 phase_offset_per_frame = None, info=None):
        """ 
        N = number of patterns
        nax = number of axes
        mod: [ N, [kx,ky,kz,depth,phase,relint]  ]
        #phase_drift: [nframes // len(mod), len(mod)]
        """
        assert mod is None or len(mod.shape) == 1
        #assert phase_per_frame is None or len(phase_per_frame.shape) == 1

        self.pattern_frames = pattern_frames
        self.mod = mod.copy() if mod is not None else None
        self.phase_offset_per_frame = phase_offset_per_frame.copy() if phase_offset_per_frame is not None else None

        self.moving_window = moving_window 
        self.info = info if info is not None else {}
        
    def save(self, fn):
        with open(fn, "wb") as s:
            pickle.dump(self.__dict__, s)
            
    @staticmethod
    def load(fn):
        with open(fn, "rb") as s:
            d = pickle.load(s)
        return ModulationPattern(**d)
        
    def clone(self):
        return ModulationPattern(self.pattern_frames,
                                 self.mod,
                                 self.moving_window,
                                 self.phase_offset_per_frame,
                                 self.info)

    def clamp_frame_ix(self, frame_ix):
        if self.phase_offset_per_frame is None:
            return frame_ix
        npat = len(self.mod)
        nf = len(self.phase_offset_per_frame) // npat
        return np.minimum(frame_ix // npat, nf-1)*npat + frame_ix % npat
       
    def mod_at_frame(self, start_frame_ix, frame_window):
        frame_ix = start_frame_ix[:, None] + np.arange(frame_window)[None]

        frame_ix = self.clamp_frame_ix(frame_ix)
        mod = self.mod[frame_ix % len(self.mod)].copy()
        
        # modify phase if interpolation data is available
        if self.phase_offset_per_frame  is not None:
            #print('adding phase offset')
            mod['phase'] -= self.phase_offset_per_frame [frame_ix]

        return mod
    
    def pos_phase(self, pos, frame):
        """
        Calculate the phase at given position and frame
        """
        pos = np.array(pos)
        mod = self.mod_at_frame(frame,1)[:,0]
        return ((mod['k'] * pos).sum(1) - mod['phase']) % (np.pi*2)
    
    def add_drift(self, drift_px):
        if self.phase_offset_per_frame is None:
            self.phase_offset_per_frame = np.zeros((len(drift_px)))
        
        for i in range(len(self.mod)):
            k = self.mod[i]['k'][:drift_px.shape[1]]
            self.phase_offset_per_frame[i:len(drift_px):len(self.mod)] += (drift_px[i::len(self.mod)]*k[None]).sum(1)

        L = len(self.phase_offset_per_frame)
        self.phase_offset_per_frame = self.phase_offset_per_frame[:L//len(self.mod)*len(self.mod)]
            
        self._center_phase_offsets()
        
    def _center_phase_offsets(self):
        """ Make sure mean of phase offsets is zero """
        
        for i in range(len(self.mod)):
            ofs = self.phase_offset_per_frame[i::len(self.mod)].mean()
            self.mod['phase'][i] += ofs
            self.phase_offset_per_frame[i::len(self.mod)] -= ofs

        self.wrap_mean_phase()
            
    def drift_trace_px(self):
        """
        Returns a XY drift trace in pixel units
        """
        
        nframes = len(self.phase_offset_per_frame)
        k = self.mod['k'][np.arange(nframes)%len(self.mod)]
        # phase = k * pos
        # pos = phase / k
        
        # normalized direction
        k_len = np.sqrt( (k**2).sum(1,keepdims=True) )
        k_norm = k / k_len
        return (self.phase_offset_per_frame[:,None] * k_norm / k_len)[:,:2] 
    
    def pitch(self):
        k = self.mod['k']
        k_len = np.sqrt((k**2).sum(1))
        pitch = 2*np.pi/k_len
        return pitch    

    @property
    def mean_phase(self):
        return self.mod['phase']
    
    @property
    def k(self):
        return self.mod['k']
    
    def __str__(self):
        pf = ""
        if self.phase_offset_per_frame  is not None:
            pf = f"{len(self.phase_offset_per_frame )} frames of phase data."
            
        axinfo = ', '.join([f'{np.rad2deg(ang):.1f} deg' for ang in self.angles])
        return f"{len(self.mod)} patterns, axes: [{axinfo}]. {pf}"

    def __repr__(self):
        return self.__str__()

    def compute_excitation(self, start_frame_ix, frame_window, pos):
        """
        
        """
        dims = pos.shape[1]
        mod = self.mod_at_frame(np.array(start_frame_ix), frame_window)
        k = mod['k'][:,:,:dims]
        spot_phase = (k * pos[:,None]).sum(-1) - mod['phase']
        exc = mod['relint']*(1+mod['depth']*np.sin(spot_phase))
        return exc
    
    def modulate(self, ds : SFDataset):
        """
        Modulate the intensities in the dataset
        """
        exc = self.compute_excitation(ds.frame, len(self.mod), ds.pos)

        ds_mod = ds[:]
        ds_mod.ibg[:, :, 0] = ds.photons[:,None] * exc
        return ds_mod

    def mod_error(self, ds):
        exc = self.compute_excitation(ds.frame, len(self.mod), ds.pos)
        sumI = np.sum(ds.ibg[:, :, 0],1)

        normI = ds.ibg[:, :, 0] / sumI[:,None]
        moderr = normI - exc
        return np.abs(moderr).max(1)
    
    @property
    def angles(self):
        k = self.mod['k']
        return np.arctan2(k[:,1],k[:,0])
    
    @property
    def depths(self):
        return self.mod['depth']
    
    @depths.setter
    def depths(self, v):
        self.mod['depth'] = v
    
    @property
    def angles_deg(self):
        return np.rad2deg(self.angles)

    def wrap_mean_phase(self):
        for i in range(len(self.mod)):
            self.mod['phase'][i] = (self.mod['phase'][i] + np.pi) % (2*np.pi) - np.pi
    
    def print_info(self, pixelsize=None, reportfn=print):
        k = self.mod['k']
        phase = self.mod['phase']
        depth = self.mod['depth']
        ri = self.mod['relint']
        
        for i in range(len(self.mod)):
            reportfn(f"Pattern {i}: kx={k[i,0]:.4f} ky={k[i,1]:.4f} Phase {phase[i]*180/np.pi:8.2f} Depth={depth[i]:5.2f} "+
                   f"Rel.Int={ri[i]:5.3f} ")
    
        for ang in range(len(self.pattern_frames)):
            pat=self.pattern_frames[ang]
            d = np.mean(depth[pat])
            phases = phase[pat]
            shifts = (np.diff(phases[-1::-1]) % (2*np.pi)) * 180/np.pi
            shifts[shifts > 180] = 360 - shifts[shifts>180]
            
            with np.printoptions(precision=3, suppress=True):
                reportfn(f"Angle {ang} shifts: {shifts} (deg) (patterns: {pat}). Depth={d:.3f}")

    def phase_drift_estimation_error(self):
        npat = self.pattern_frames.size
        est_err = np.zeros(len(self.pattern_frames))
        for i in range(len(self.pattern_frames)):
            pf = self.pattern_frames[i]
            phase_drift = self.phase_offset_per_frame
            phase_drift = phase_drift[:len(phase_drift)//npat*npat].reshape(-1,npat)
            phases = phase_drift[:,pf]
            est_err[i] = (phases - phases.mean(1,keepdims=True)).std(1).mean()
            print(f'Phase estimation error angle {i}: {np.rad2deg(est_err[i]):.2f} deg')

        return est_err
    
    def plot_phase_drift(self, nframes=None, ax=None, colors=None, label='Pattern {0}', **kwargs):
        if ax is None:
            fig,ax=plt.subplots(len(self.pattern_frames), 1,figsize=(8,6),sharex=True)
        if nframes is None:
            nframes = len(self.phase_offset_per_frame)

        npat = self.pattern_frames.size
        for j in range(len(self.pattern_frames)):
            pf = self.pattern_frames[j]
            for i in range(len(pf)):
                n = len(self.phase_offset_per_frame[pf[i]:nframes:npat])
                ax[j].plot(np.arange(n)*npat+pf[i], 
                    np.rad2deg( self.phase_offset_per_frame[pf[i]:nframes:npat] + self.mod['phase'][pf[i]]), 
                c=colors[i] if colors is not None else None,
                    label=label.format(pf[i]), **kwargs)

        for i in range(len(self.pattern_frames)):
            ax[i].legend()
        plt.xlabel('Frame')
        plt.ylabel('Phase [deg]')

    def phase_error_rmsd(self, mp2):
        npat = self.pattern_frames.size
        nframes = len(self.phase_offset_per_frame) // npat * npat
        phase1 = self.mod['phase'][None] + self.phase_offset_per_frame[:nframes].reshape(nframes//npat, npat)
        phase2 = mp2.mod['phase'][None] + mp2.phase_offset_per_frame[:nframes].reshape(nframes//npat, npat)
        return np.sqrt(np.mean((phase1-phase2)**2))

    def const_phase_offsets(self):
        r = self.clone()
        npat = self.pattern_frames.size
        po = r.phase_offset_per_frame[:len(r.phase_offset_per_frame)//npat*npat].reshape(-1,npat)
        for i in range(len(self.pattern_frames)):
            pf = self.pattern_frames[i]

            mean_drift = po[:,pf].mean(1)
            mean_drift -= mean_drift.mean()
            mean_offset = po[:,pf].mean(0)
            po[:,pf] = mean_drift[:,None] + mean_offset[None]
        r.phase_offset_per_frame = po.flatten()
        return r

def angles_to_mod(pitch_nm, pixelsize, angle_deg, depth, pattern_frames, z_pitch=None):
    """
    Assign mod array and phase_interp for simulation purposes
    """
    freq = 2*np.pi/np.array(pitch_nm)*pixelsize
    angle = np.deg2rad(angle_deg)
    mod = np.zeros(pattern_frames.size, dtype=ModulationType)

    for i,pf in enumerate(pattern_frames):
        mod['k'][pf,0] = np.cos(angle[i]) * freq[i]
        mod['k'][pf,1] = np.sin(angle[i]) * freq[i]
        mod['k'][pf,2] = 2*np.pi/z_pitch[i] if z_pitch is not None else 0
        mod['phase'][pf] = np.linspace(0,2*np.pi,len(pf),endpoint=False)

    mod['depth'] = depth
    mod['relint'] = 1/pattern_frames.size
    return mod


# Curve-fit y around the peak to find subpixel peak x
def quadraticpeak(y, x=None, npts=7, plotTitle=None):
    if x is None:
        x = np.arange(len(y))
    xmax = np.argmax(y)
    W = int((npts + 1) / 2)
    window = np.arange(xmax - W + 1, xmax + W)
    window = np.clip(window, 0, len(x) - 1)
    
    xw = x[window]
    yw = y[window]
    p = Polynomial.fit(xw, yw, 2)

    if plotTitle:
        plt.figure()
        plt.plot(xw, yw, label="data")
        sx = np.linspace(x[xmax - W], x[xmax + W], 100)
        plt.plot(sx, p(sx), label="fit")
        plt.legend()
        plt.title(plotTitle)

    return p.deriv().roots()[0]


@torch.jit.script
def _dft(xyI, kx,ky):
    p = xyI[:,0,None,None] * kx[None, None, :] + xyI[:,1,None,None] * ky[None, :, None]
    r = ( torch.cos(p) * xyI[:,2,None,None] ).sum(0)
    i = ( torch.sin(p) * xyI[:,2,None,None] ).sum(0)

    return torch.complex(r,i)    

def torch_dft(xyI, kx,ky, device, batch_size=None):
    if batch_size is None:
        batch_size = 20000

    with torch.no_grad():
        xyI = torch.tensor(xyI)
        kx = torch.tensor(kx, device=device)
        ky = torch.tensor(ky, device=device)

        return torch.stack([ _dft(s.clone(), kx, ky) for s in 
                                torch.split(xyI.to(device), batch_size)]).sum(0)


def cuda_dft(xyI, kx,ky, useCuda=True):
    from fastpsf import Context
    from fastpsf.simflux import SIMFLUX
    import numpy as np
    
    KX,KY = np.meshgrid(kx,ky)
    klist = np.zeros((len(kx)*len(ky),2),dtype=np.float32)
    klist[:,0] = KX.flatten()
    klist[:,1] = KY.flatten()
    with Context() as ctx:
        return SIMFLUX(ctx).DFT2D(xyI, klist, useCuda=useCuda).reshape((len(ky),len(kx)))
    


def render_sr_image_per_pattern(xy, I, img_shape, sr_factor):
    h,w=img_shape
    #img = np.zeros((h*sr_factor,w*sr_factor), dtype=np.float32)
    
    H,W = h*sr_factor, w*sr_factor
    img, xedges, yedges = np.histogram2d(xy[:,1]*sr_factor, xy[:,0]*sr_factor, 
                                         bins=[H,W], range=[[0, H], [0, W]], weights=I )
    
    return img
    

def estimate_angle_and_pitch_dft(xy, I, frame_ix, freq_minmax, imgshape,
                                  dft_peak_search_range, file_ix=0, fft_timebins=1,
                                  debug_images=False, sr_zoom=6, device=None, 
                                  results_dir=None, dft_batch_size=50000, fft_img_cb=None):
    """
    xy: [N, 2]
    I: [N, num phase steps]
    frame_ix: [N]
    """
    h,w=imgshape
    
    from smlmtorch.util.locs_util import indices_per_frame
    ipf = indices_per_frame(frame_ix)
    framebins = np.array_split(np.arange(len(ipf)), fft_timebins)
    
    ft_sum = torch.zeros((h*sr_zoom,w*sr_zoom),device=device)
    ft_smpimg  = ft_sum * 0
    for i in range(len(framebins)):
        ix = np.concatenate( [ipf[f] for f in framebins[i]] )
        # render all patterns
        smpimg = render_sr_image_per_pattern(xy[ix], I[ix].sum(1), imgshape, sr_zoom)
        ft_smpimg += torch.abs(torch.fft.fft2(torch.from_numpy(smpimg).to(device)))
        
        for ep in range(I.shape[1]):
            img = render_sr_image_per_pattern(xy[ix], I[ix][:,ep], imgshape, sr_zoom)
            
            #if results_dir is not None:
            #    plt.imsave(f"{results_dir}/ep{ep}_sr_render.png", img/np.max(img))
    
            ft_img = torch.fft.fft2(torch.from_numpy(img).to(device))
            ft_sum += torch.abs(ft_img)
         
    ft_smpimg /= torch.sum(ft_smpimg)
    ft_smpimg = torch.fft.fftshift(ft_smpimg)
    ft_sum = torch.fft.fftshift(ft_sum)
    
    freq = torch.fft.fftshift( torch.fft.fftfreq(h*sr_zoom) )*sr_zoom*2*np.pi
    XFreq, YFreq = torch.meshgrid(freq,freq, indexing='xy')
    Freq = torch.sqrt(XFreq**2+YFreq**2)

    mask = (Freq>freq_minmax[0]) & (Freq<freq_minmax[1])
    ft_smpimg[~mask] = 0
    ft_sum[~mask] = 0
        
    ft_sum /= ft_sum.sum()
    ft_sum = ft_sum - ft_smpimg

    #print(f"Max pixel frequency: {freq[0]:.2f}")
    
    ft_sum = ft_sum.cpu().numpy()

    if debug_images:
        saved_img = ft_sum*1
        plt.imsave(f"imgs/{file_ix}pattern-FFT.png", saved_img)

    #ft_sum = ft_sum / np.sum(ft_sum) - ft_smpimg
    
#        plt.imsave(self.outdir + f"pattern-{pattern_indices}-FFT-mask.png", mask)
    
    if results_dir:
        plt.imsave(f"{results_dir}/pattern-{file_ix}-FFT-norm.png", ft_sum)
        
    if fft_img_cb is not None:
        fft_img_cb(ft_sum, file_ix)

    max_index = np.argmax(ft_sum)
    max_indices = np.unravel_index(max_index, ft_sum.shape)
    
    W=10
    if results_dir:
        plt.imsave(f'{results_dir}/pattern-{file_ix}-FFT-peak.png', 
               ft_sum[max_indices[0]-W:max_indices[0]+W,
                      max_indices[1]-W:max_indices[1]+W])
    
    #print(f'Freq peak value:{ft_sum[max_indices]}')
            
    peak_yx = freq[list(max_indices)]
    peak_xy = _find_dft2_peak(peak_yx [[1,0]], xy, I, dft_peak_search_range, 
                              file_ix, device, results_dir, batch_size=dft_batch_size)

    return peak_xy    
    

def _find_dft2_peak(xy, loc_xy, I, dft_peak_search_range=0.02, file_ix=0, device=0, 
                    results_dir=None, batch_size=None):
    def compute_peak_img(x,y,S):
        kxrange = np.linspace(x-S, x+S, 50)
        kyrange = np.linspace(y-S, y+S, 50)

        """
        img = torch.zeros((len(kyrange),len(kxrange)))
        for ep in range(I.shape[1]):
            xyI = np.concatenate((loc_xy, I[:,[ep]]), 1)

            sig = torch_dft(xyI, kxrange, kyrange, device=device,batch_size=batch_size).cpu()
            img += torch.abs(sig**2)
        img = img.numpy()
         """
        img = np.zeros((len(kyrange),len(kxrange)))
        for ep in range(I.shape[1]):
            xyI = np.concatenate((loc_xy, I[:,[ep]]), 1)

            sig = cuda_dft(xyI, kxrange, kyrange, useCuda=True)
            img += np.abs(sig**2)
         
        peak = np.argmax(img)
        peak = np.unravel_index(peak, img.shape)
        kx_peak = quadraticpeak(img[peak[0], :], kxrange, npts=11, plotTitle=None)#='X peak')
        ky_peak = quadraticpeak(img[:, peak[1]], kyrange, npts=11, plotTitle=None)#='Y peak')

        return img, kx_peak, ky_peak
    
    peakimg, kxpeak, kypeak= compute_peak_img(*xy, dft_peak_search_range)
    if results_dir:
        plt.imsave(f"{results_dir}/pattern-{file_ix}-DFT-peak1.png", peakimg)
    
    peakimg2, kxpeak2, kypeak2 = compute_peak_img(kxpeak, kypeak, dft_peak_search_range)
    if results_dir:
        plt.imsave(f"{results_dir}/pattern-{file_ix}-DFT-peak2.png", peakimg2)

    #print(f"KXPeak={kxpeak:.2f},{kypeak:.2f},{kxpeak2:.2f},{kypeak2:.2f}")
    return kxpeak2, kypeak2



def _estimate_phase_and_depth_bin(pos, I, frame_ix, k, iterations, accept_percentile, verbose=True):
    nsteps = I.shape[1]

    def process_step(ep):
        sumI = I.sum(1)

        intensity = sumI
        spotPhaseField = (k[None] * pos[:,:len(k)]).sum(1)
        
        basefreq = torch.tensor([-1, 0, 1],device=pos.device)
        weights = torch.ones(len(intensity), device=pos.device)

        depth_trace=[]
        phase_trace=[]
        for it in range(iterations):
            # DFT on modulated and unmodulated data
            # [N, ep, freqs] -> sum(0) -> [ep, freqs]
            f = (weights[:,None] * I[:,ep,None] * torch.exp(-1j * spotPhaseField[:,None] * basefreq[None,:])).sum(0)
            B = (weights[:,None] * sumI[:,None] * torch.exp(-1j * spotPhaseField[:,None] * (basefreq-1)[None,:])).sum(0)
            A = (weights[:,None] * sumI[:,None] * torch.exp(-1j * spotPhaseField[:,None] * basefreq[None,:])).sum(0)
            C = (weights[:,None] * sumI[:,None] * torch.exp(-1j * spotPhaseField[:,None] * (basefreq+1)[None,:])).sum(0)
            M = torch.stack([B,C,A],1)
                            
            # Actually x[1] is just the complex conjugate of x[0], 
            # so it seems a solution with just 2 degrees of freedom is also possible
#            x, residual, rank, s = np.linalg.lstsq(M,b,rcond=None)
            b,c,a = torch.linalg.solve(M, f)
           
            depth = torch.real(2*torch.abs(b)/a)
            phase = -torch.angle(b*1j)
            relint = torch.real(2*a)/2  

            q = relint * (1+depth*torch.sin( (k[None]*pos[:,:len(k)]).sum(1) - phase))
            normI = I[:,ep] / sumI
            errs = (normI-q)**2
            moderr = errs.mean().cpu().numpy()
            
            errs_ = errs.cpu().numpy()
            median_err = np.percentile(errs_, accept_percentile)# median(errs)
            weights = errs < median_err
            
            depth_trace.append(depth)
            phase_trace.append(phase)
            
        if verbose:
            f_min = int(frame_ix.min().numpy())
            f_max = int(frame_ix.max().numpy())
            print(f"Frame {f_min, f_max}, EP={ep}. Depth estimation per iteration: ", ','.join([f"{d:.2f}" for d in depth_trace]))
            #phase_trace = np.diff(np.array(phase_trace))
            #print("Phase estimation per iteration: ", ','.join([f"{d:.2f}" for d in phase_trace]))

#        print(f"ep{ep}. depth={depth}, phase={phase}")
        return phase, depth, relint, moderr

    with torch.no_grad():
        return np.array([process_step(ep) for ep in range(nsteps)])

def _estimate_phase_and_depth_bin_fast(ix_per_bin, xyz, I, nframes, device, k, pattern_frames,
                     iterations, verbose, accept_percentile):

    nbins = len(ix_per_bin)
    max_binsize = max([len(ix) for ix in ix_per_bin])
    pos_b = torch.zeros((nbins, max_binsize, xyz.shape[1]))
    I_b = torch.zeros((nbins, max_binsize, I.shape[1]))
    mask = torch.zeros((nbins, max_binsize), dtype=bool)
    for i,ix in enumerate(ix_per_bin):
        pos_b[i,:len(ix)] = xyz[ix]
        I_b[i,:len(ix)] = I[ix]
        mask[i,:len(ix)] = True

    pos_b = pos_b.to(device)
    I_b = I_b.to(device)
    mask = mask.to(device)
    k = k.to(device)

    intensity = I_b.sum(2)
    spot_phase = (k[None,None] * pos_b).sum(2)
    basefreq = torch.tensor([-1, 0, 1],device=pos_b.device)

    weights = mask*1

    n_ep = I.shape[1] # num excitation patterns in angle

    phases = torch.zeros((nbins, n_ep), device=device)
    depths = torch.zeros((nbins, n_ep), device=device)

    for it in range(iterations):
        A = (weights[:,:,None] * intensity[:,:,None] * torch.exp(-1j * spot_phase[:,:,None] * basefreq[None,None])).sum(1)
        B = (weights[:,:,None] * intensity[:,:,None] * torch.exp(-1j * spot_phase[:,:,None] * (basefreq-1)[None,None])).sum(1)
        C = (weights[:,:,None] * intensity[:,:,None] * torch.exp(-1j * spot_phase[:,:,None] * (basefreq+1)[None,None])).sum(1)
        M = torch.stack([B,C,A],2)

        for ep in range(n_ep):
            f = (weights[:,:,None] * I_b[:,:,ep,None] * torch.exp(-1j * spot_phase[:,:,None] * basefreq[None,None])).sum(1)
            r = torch.linalg.solve(M, f)
            b,c,a = r[:,0], r[:,1], r[:,2]

            depths[:,ep] = torch.real(2*torch.abs(b)/a)
            phases[:,ep] = -torch.angle(b*1j)

            #results[0,ep] = phase
            #results[1,ep] = depth

        normI_pred = (1+depths[:,None]*torch.sin (spot_phase[:,:,None]-phases[:,None])) / n_ep
        normI_meas = I_b / torch.clamp(intensity[:,:,None],1e-9)
        errs = ((normI_meas-normI_pred)**2 * mask[:,:,None]).max(2)[0] # take max over all exposures
        errs[torch.logical_not(mask)] = float('nan')
        err_threshold = torch.nanquantile(errs, accept_percentile/100, dim=1, keepdim=True)
        weights = mask * (errs < err_threshold)

    return phases.T.cpu().numpy(), depths.T.cpu().numpy()

    """
        q = relint[:,None] * (1+depth[:,None]*torch.sin( (k[None,None]*pos_b).sum(2) - phase[:,None]))
        #q = relint * (1+depth*torch.sin( (k[None]*pos[:,:len(k)]).sum(1) - phase))
        normI = I_b[:,:,ep] / torch.clamp(intensity,1e-9)
        errs = (normI-q)**2 * mask
        if accept_percentile>0:
            errs[torch.logical_not(mask)] = float('nan')
            err_threshold = torch.nanquantile(errs, accept_percentile/100, dim=1, keepdim=True)
            weights = mask * (errs < err_threshold)
    """
    return 0


def indices_per_frame(frame_indices):
    if len(frame_indices) == 0: 
        numFrames = 0
    else:
        numFrames = torch.max(frame_indices)+1
    frames = [[] for i in range(numFrames)]
    for k in range(len(frame_indices)):
        frames[frame_indices[k]].append(k)
    for f in range(numFrames):
        frames[f] = torch.tensor(frames[f], dtype=torch.int64)
    return frames
    
        
def _estimate_angles(xy, intensities, frame_ix, imgshape, pattern_frames, pitch_minmax_nm,
                      pixelsize, moving_window, **kwargs):

    freq_minmax = 2*np.pi / (torch.tensor(pitch_minmax_nm[::-1].copy()) / pixelsize)
    npat = pattern_frames.size
    
    assert intensities.shape[1] == npat
    
    k = np.zeros((npat, 2))
    ri = np.zeros(npat)
    
    for i, pf in enumerate(pattern_frames):
        #I = torch.stack( [ intensities[torch.arange(len(xy)), ( pf[j] + npat//2 - frame_ix) % npat] for j in range(len(pf)) ], -1 )
        if moving_window:
            I = np.stack( [ intensities[np.arange(len(xy)), ( pf[j] - frame_ix) % npat] for j in range(len(pf)) ], -1 )
        else:
            I = intensities[:,pf]
        
        k_i = estimate_angle_and_pitch_dft(xy = xy, 
                                     I = I,
                                     frame_ix = frame_ix,
                                     freq_minmax = freq_minmax,
                                     imgshape = imgshape, 
                                     dft_peak_search_range = 0.03,
                                     file_ix = i,
                                     **kwargs)

        if k_i[np.abs(k_i).argmax()] < 0:
            k_i = -np.array(k_i)
        kx,ky = k_i
        
        angle_rad = np.arctan2(ky,kx)
        freq_px= np.sqrt(kx**2+ky**2)
        pitch_px = 2*np.pi/freq_px
        k[pf] = [kx,ky]
        ri[pf] = I.sum()
            
        print(f"Angle: {np.rad2deg(angle_rad):.2f}, Pitch: {pitch_px * pixelsize:.2f} nm")

    ri /= ri.sum()

    return k,ri



def _estimate_phases(xyz:Tensor, intensities:Tensor, k, frame_ix:Tensor, numframes, pattern_frames, spots_per_bin, 
                    device, moving_window, fig_callback = None, **kwargs):

    if spots_per_bin > 0:
        ix_per_bin = array_split_minimum_binsize(np.array(frame_ix), binsize = spots_per_bin, use_tqdm=False)
    else:
        ix_per_bin = [np.array(frame_ix)]
    
    if len(ix_per_bin) == 2:
        raise ValueError(f'spots_per_bin should be set to result in either 1 or more than 2 frame bins (now {len(ix_per_bin)}, for {len(frame_ix)} spots.)')

    npat = np.array(pattern_frames).size
    num_angles = pattern_frames.shape[0]
    num_steps = pattern_frames.shape[1]
    npat = pattern_frames.size
    
    with torch.no_grad():
        phases = np.zeros((num_angles, num_steps, len(ix_per_bin)))
        depths = np.zeros((num_angles, num_steps, len(ix_per_bin)))
        for i, pf in enumerate(pattern_frames):
            if moving_window:
                I = torch.stack( [ intensities[torch.arange(len(xyz)), ( pf[j] #+ npat//2 
                            - frame_ix.long()) % npat] for j in range(len(pf)) ], -1 )
            else:
                I = intensities[:,pf]
            phases[i], depths[i] = _estimate_phase_and_depth_bin_fast(ix_per_bin, xyz, I, numframes, device, k[i], pattern_frames, **kwargs)

        phases = np.unwrap(phases, axis=-1)

    # store interpolated phase for every frame
    frame_bin_t = [frame_ix[ipb].float().mean() for ipb in ix_per_bin]
    phase_interp = np.zeros((num_angles, num_steps, numframes))
    for i in range(len(pattern_frames)):
        for j,p_ix in enumerate(pattern_frames[i]):
            if len(ix_per_bin)>1:
                spl = InterpolatedUnivariateSpline(frame_bin_t, phases[i,j], k=2)
                phase_interp[i,j] = spl(np.arange(numframes))
            else:
                phase_interp[i,j] = phases[i,j]

    if fig_callback is not None:        
        fig,axes=plt.subplots(len(pattern_frames),squeeze=False)
        for i, pf in enumerate(pattern_frames):
            for j in range(len(pf)):
                l=axes[i][0].plot(frame_bin_t, phases[i,j], '.', label=f'step {j}')
                axes[i][0].plot(phase_interp[i,j],'--', color=l[0].get_color())
            axes[i][0].set_title(f"angle {i} phases")
            axes[i][0].legend()
        plt.tight_layout()
        fig_callback('phases')

        fig,axes=plt.subplots(len(pattern_frames),squeeze=False)
        for i, pf in enumerate(pattern_frames):
            for j in range(len(pf)):
                axes[i][0].plot(depths[i,j],'.-', label=f'step {j}')
            axes[i][0].set_title(f"angle {i} depths")
            axes[i][0].legend()
        plt.tight_layout()
        fig_callback('depths')
        
    return depths, phases, phase_interp, dict(frame_bin_t=frame_bin_t)

def estimate_kz(xyz: Tensor, intensities: Tensor, kxy, z_pitch_range, frame_ix, frame_bins, 
                    fig_callback = None):
        
    nframes = frame_ix.max()+1
    frame_bin_ix = (frame_ix / nframes * frame_bins).long()
    ix_per_bin = indices_per_frame(frame_bin_ix)
    
    numsteps = intensities.shape[1]
    
    pdre = np.zeros((len(z_pitch_range), numsteps, frame_bins, 4))
    
    for i,z_pitch in progbar(enumerate(z_pitch_range),total=len(z_pitch_range)):
        for j in range(frame_bins):
            loc_ix = ix_per_bin[j]
            
            kz = 2*np.pi/z_pitch
            k = torch.tensor([kxy[0], kxy[1], kz], device=xyz.device)

            pdre[i, :, j] = _estimate_phase_and_depth_bin(xyz[loc_ix], intensities[loc_ix], frame_ix[loc_ix], k, 
                                                iterations=1, 
                                                accept_percentile=100, # redundant if iterations=1 
                                                verbose=False)
            
            #moderr[i, j] = _moderror(xyz[loc_ix], intensities[loc_ix], frame_ix[loc_ix])
    
    if fig_callback is not None:
        fig,ax=plt.subplots(2,sharex=True)

        kz_vs_depth = pdre[:,:,:,1].mean((1,2))        
        kz_peak = quadraticpeak(kz_vs_depth, z_pitch_range)
        if kz_peak > z_pitch_range[-1]: 
            kz_peak = None
        else:          
            print(f"depth based kz estimate: {kz_peak:.4f}")

        kz_vs_moderr = pdre[:,:,:,3].mean((1,2))
        kz_peak_moderr = quadraticpeak(-kz_vs_moderr, z_pitch_range)
        if kz_peak_moderr  > z_pitch_range[-1]: 
            kz_peak_moderr = None
        else:        
            print(f"mod error based kz estimate: {kz_peak_moderr:.4f}")
        
        ax[0].plot(z_pitch_range, kz_vs_depth, label=f'Peak {kz_peak:.2f} um' if kz_peak is not None else None)
        ax[1].plot(z_pitch_range, kz_vs_moderr, label=f'Peak {kz_peak_moderr:.2f} um' if kz_peak_moderr is not None else None)
        ax[1].set_xlabel('Z pitch [um]')
        ax[0].set_ylabel('Depth')
        ax[1].set_ylabel('Moderr')
        ax[0].legend()
        ax[1].legend()
        
        fig_callback('kz')
    
    return pdre

def simple_sine_fit(I : Tensor):
    """
    Sine fits as done in the SIMPLE paper, resulting in modulation depth per spot and per axis.
    Pattern is defined as I = A/2 * (1+cos(2pi * (x+phase))) + b
    """
    
    _x = torch.sqrt(
        (I[:,0]-I[:,1])**2 + 
        (I[:,0]-I[:,2])**2 +
        (I[:,1]-I[:,2])**2)
    
    ampl = 2*np.sqrt(2)/3 * _x
    
    phase = -1/torch.pi*torch.arctan((-2*I[:,0]+I[:,1]+I[:,2]+np.sqrt(2)*_x)/
                                     (np.sqrt(3)*(I[:,1]-I[:,2])))
    
    bg = 1/3*(I[:,0]+I[:,1]+I[:,2]-np.sqrt(2)*_x)
    
    return ampl, phase/(2*torch.pi), bg


    
    
def estimate_angles(pitch_minmax_nm, ds: SFDataset, pattern_frames, result_dir, 
                    moving_window=True, **kwargs):

    k, ri = _estimate_angles(ds.pos[:,:2], ds.ibg[:,:,0], 
                            ds.frame, ds.imgshape,
                            pitch_minmax_nm=pitch_minmax_nm,
                    pattern_frames = pattern_frames, 
                    results_dir = result_dir, 
                    moving_window=moving_window,
                    pixelsize = ds['pixelsize'], **kwargs)
    
    mod = np.zeros(pattern_frames.size, dtype=ModulationType)
    mod['k'][:,:2] = k
    mod['relint'] = ri

    #print(k, ri)
    return ModulationPattern(pattern_frames, mod = mod, moving_window = moving_window)
    
        
def estimate_phases(ds: SFDataset, mp: ModulationPattern, spots_per_bin, numframes=None,
                    accept_percentile = 50, iterations = 10, verbose=True, device='cpu',
                    fig_callback = None):
    """
    estimate phase, depth, and a coarse estimate of phase drift
    phase_binsize: minimum number of localizations in each phase estimation bin
    """
    
    dims = ds.pos.shape[1]
    
    if numframes is None:
        numframes = ds.numFrames + mp.pattern_frames.size - 1
    
    k = torch.tensor( mp.mod['k'][mp.pattern_frames[:,0]][:,:dims] )
    
    depth_per_bin, phase_per_bin, phase_interp, estimation_info = _estimate_phases(
        torch.from_numpy(ds.pos), 
        intensities = torch.from_numpy(ds.ibg[:,:,0]),
        k=k,
        frame_ix = torch.from_numpy(ds.frame), 
        pattern_frames = mp.pattern_frames, 
        spots_per_bin = spots_per_bin, 
        device = torch.device(device) if type(device)==str else device,
        accept_percentile = accept_percentile,
        iterations = iterations, 
        moving_window = mp.moving_window,
        fig_callback = fig_callback, 
        numframes = numframes,
        verbose = verbose)

    npat = mp.pattern_frames.size
    depths = np.zeros((npat,))
    phases = np.zeros((npat,))
    for i, pf in enumerate(mp.pattern_frames):
        depths[pf] = depth_per_bin[i].mean(1)
        phases[pf] = phase_per_bin[i].mean(1)            
                
    mp.mod['phase'] = phases
    mp.mod['depth'] = depths

    mp.info = dict(mp.info, **estimation_info) if mp.info is not None else estimation_info

    # phase interp stores the interpolated phases for all patterns seperately,
    # also at frames where that particular pattern is not being used.
    mp.phase_offset_per_frame = np.zeros((phase_interp.shape[-1]))
    for i,pf in enumerate(mp.pattern_frames):
        for j in range(len(pf)): 
            mp.phase_offset_per_frame[pf[j]::npat] = phase_interp[i, j, pf[j]::npat] - phases[pf[j]]
        
    mp._center_phase_offsets()
    
    return mp



def estimate_phase_drift(ds: SFDataset, mp: ModulationPattern, frame_binsize, 
                         device=None, me_threshold=0.1, loss_max=0.01, max_iterations=2000, print_step=100, **kwargs):
    """
    Estimate phase drift, keeping same-angle phase steps and depth fixed
    """
    ds = ds[mp.mod_error(ds) < me_threshold]
    
    print(f'size of remaining ds:{len(ds)}')
    #ds = ds[ds.frame % len(mp.mod) == 0]
    
    mp = mp.clone()    
    dev = torch.device(device) if device is not None else None
    intensities = torch.from_numpy(ds.ibg[:,:,0]).to(dev)

    from smlmtorch.util.locs_util import indices_per_frame
    frame_bins = indices_per_frame(ds.frame // len(mp.mod) // frame_binsize)# array_split_minimum_binsize(ds.frame, frame_binsize)
    
    for axis, pf in enumerate(mp.pattern_frames):
        if mp.moving_window:
            # Rotate the 1st axis depending on frame index
            I = torch.stack( [ intensities[torch.arange(len(ds.pos)), ( pf[j] #+ npat//2 
                - ds.frame) % mp.pattern_frames.size] for j in range(len(pf)) ], -1 )
        else:
            I = intensities[:,pf]

        drift = torch.from_numpy(mp.phase_offset_per_frame[pf[0]::len(mp.mod)]).to(dev)
        
        if frame_binsize == 1:
            param = [drift]
            drift.requires_grad = True
        
            frame_ix = torch.from_numpy(ds.frame // len(mp.mod)).long()
            get_drift = lambda ix: drift[ix]
        else:
            knots = torch.zeros((len(frame_bins),1))
            for i,fb in enumerate(frame_bins):
                knots[i,0] = drift[i*frame_binsize].mean()
            spl = CatmullRomSpline1D(knots.to(dev))
            param = spl.parameters()
            get_drift = lambda ix: spl(ix / frame_binsize)[:,0]
            frame_ix = torch.from_numpy(ds.frame // len(mp.mod)).to(dev)
       
        optim = AdaptiveStepGradientDescent(param, **kwargs)

        phases = torch.from_numpy(mp.mod['phase'][pf]).to(dev)
        depth = mp.mod['depth'][pf].mean()
        
        #I = torch.from_numpy(intensities[:,pf]).to(dev)
        normI = I/I.sum(1,keepdims=True)
        k = mp.mod['k'][:,:ds.pos.shape[1]]
        xyz_phase = torch.from_numpy((ds.pos * k[pf[0]][None]).sum(1)).to(dev)
        
        for i in range(max_iterations):
            def loss_fn():
                optim.zero_grad()
                loc_phase = xyz_phase[:,None] - (get_drift(frame_ix)[:,None] + phases[None])
                mu = 1 + depth * torch.sin(loc_phase)
                mu = mu / mu.sum(1,keepdims=True)
                loss = torch.clamp( (mu-normI)**2, max=loss_max).mean()
                
                loss.backward()
                return loss
        
            with torch.no_grad():
                loss = optim.step(loss_fn)
                
            if i % print_step == 0:
                loss =  loss.detach().cpu().numpy()
                print(f"Iteration={i}. Loss: {loss}. Stepsize: {optim.stepsize}")
                
            if optim.finished:
                break
            
        for i,f in enumerate(pf):
            d = get_drift(torch.arange(len(mp.phase_offset_per_frame[f::len(mp.mod)])).to(dev)).detach().cpu().numpy()
            mp.phase_offset_per_frame[f::len(mp.mod)] = d #  all steps are assigned the same offset
                    
    return mp
    

def estimate_phase_drift_cv(ds, mp : ModulationPattern, result_dir, frame_binsize, **kwargs):
    """
    estimate_phase_drift, but also estimate estimation precision using 2-fold cross validation.
    ds: Dataset
    """
    mask = np.random.randint(2, size=len(ds))
    iterations = 3
    
    pat = []
    for j, ds_half in enumerate([ds[mask==1], ds[mask==0]]):
        mp_r = mp.clone()  # iteratively update (modulation error will update the list of localizations)
        mp_r.phase_offset_per_frame[:] = 0
        for i in range(iterations):
            print()
            mp_r = estimate_phase_drift(ds_half, mp_r, frame_binsize, **kwargs)
            
        pat.append(mp_r)
        mp_r.save(result_dir + f"pattern-cv{i}.pickle")

    rms_err = np.sqrt( np.mean( (pat[0].phase_offset_per_frame - pat[1].phase_offset_per_frame)**2 ) )
    print(f"RMS Phase error: {np.rad2deg(rms_err):.5f} deg.")
    
    # Compute final:
    mp_r = mp.clone()
    for i in range(iterations):
        mp_r = estimate_phase_drift(ds, mp_r, frame_binsize, **kwargs)
    
    mp_r.save(result_dir + "pattern.pickle")
            
    return config_dict(pattern=mp_r, org_mp=mp, cv=pat, 
                  rms_err=rms_err, frame_binsize=frame_binsize)


def plot_phase_drift(result, time_range, mp_gt = None):
    cv = result.cv
    rms_err_rad = result.rms_err

    L=1000
    t = np.arange(L)

    pf = result.pattern.pattern_frames
    pat_ix = [ pf_i[0] for pf_i in pf ]
    fig,ax=plt.subplots(len(pat_ix),1,sharex=True)
    for i in range(len(pat_ix)):
        #ax[i].plot(np.rad2deg(.phase_offset_per_frame[pat_ix[i]::pf.size]), label='Estimated (SMLM)')
        ax[i].plot(t,np.rad2deg(cv[0].phase_offset_per_frame[i::pf.size][s]), '-',label='Crossval. bin 1')
        ax[i].plot(t,np.rad2deg(cv[1].phase_offset_per_frame[i::pf.size][s]), '-',label='Crossval. bin 2')

        ax[i].plot(t,np.rad2deg(result.org_mp.phase_offset_per_frame[pat_ix[i]::pf.size]), label='Estimated')
        ax[i].plot(t,np.rad2deg(result.pattern.phase_offset_per_frame[pat_ix[i]::pf.size]), label='Estimated (Refined)')
        #ax[i].plot(mp_dc.phase_offset_per_frame[i::pattern_frames.size], label='Refined')
        if mp_gt is not None:
            ax[i].plot(np.rad2deg(mp_gt.phase_offset_per_frame[pat_ix[i]::pf.size]), label='GT')
        ax[i].set_xlabel('Frame')
        ax[i].set_ylabel('Phase [deg]')

        k = result.pattern.mod['k'][i]
        ang = np.rad2deg(np.arctan2(k[1],k[0]))

        ax[i].set_title(f'Estimated phase drift for pattern {i} [angle {ang:.1f} deg]. \nRMS Phase err: {np.rad2deg(rms_err_rad):.2f} deg. Frames/bin={result.frame_binsize}')
    plt.legend()

    #ax[i].set_title(f'Estimated phase drift for pattern {i} [angle {ang:.1f} deg]. \nRMS Phase err: {np.rad2deg(rms_err_rad):.2f} deg. Frames/bin={phase_drift_binsize}')
    plt.suptitle('Estimated phase drift')
    plt.legend()
    plt.tight_layout()
    

def ndi_fit_dataset(ds: SFDataset, mp: ModulationPattern, ndims:int = 2, 
            device=None, ndi_with_bg = False, fixed_intensity = False):

    assert ndims==2
    assert len(mp.pattern_frames) == 2
    mod_per_frame = mp.mod_at_frame(np.arange(ds.numFrames), mp.pattern_frames.size)
    
    batchsize = 20000
    nbatch = (len(ds) + batchsize - 1) // batchsize
            
    dev = device
    
    limitrange = mp.pitch().max()
    
    param_range = torch.tensor([
        [-limitrange/2,limitrange/2],
        [1, 1e9], [1, 1e9],
        [0, 1e2], [0, 1e2]
    ]).to(dev)

    results = []
    crlbs = []

    nax = len(mp.pattern_frames)

    # todo: fix this for 3D case

    if fixed_intensity:
        fitter = ndi_fit_fixed_intensity
        param_range = param_range[[0]*ndims].contiguous()
    else:
        if ndi_with_bg:
            fitter = ndi_bg_fit
            param_range = param_range[[0]*ndims + [1]*nax + [2]*nax].contiguous()
        else:
            fitter = ndi_fit
            param_range = param_range[[0]*ndims + [1]*nax].contiguous()
    
    with progbar(total=len(ds)) as pb:
        for i, bix in enumerate(np.array_split(np.arange(len(ds)), nbatch)):
            
            mod_batch = mod_per_frame[ds.frame[bix]]
            mod_batch['phase'] -= ds.pos[bix,1][:,None] * mod_batch['k'][:,:,1]
            mod_batch['phase'] -= ds.pos[bix,0][:,None] * mod_batch['k'][:,:,0]
            mod_batch_fl = np.reshape(mod_batch.view(np.float32), (len(mod_batch), len(mp.mod), 6))
            mod_batch_fl = mod_batch_fl [:,mp.pattern_frames]
        
            initial = np.zeros( (len(bix), 6) )
            initial[:,:ndims] = 0#ds.pos[bix][:,:ndims] # pos - roipos
            initial[:,ndims:ndims+2] = (ds.photons[bix]/2)[:,None]

            if fixed_intensity:
                initial = initial[:,:ndims] # only XY
            else:
                if ndi_with_bg:
                    initial[:,ndims+2:ndims+4] = ds.background[bix,None]/len(mp.mod)
                else:
                    initial = initial[:,:ndims+2]

            I_smp_mu = ds.ibg[bix,:,0][:,mp.pattern_frames]
            I_smp_sig = ds.ibg_crlb[bix,:,0][:, mp.pattern_frames]
        
            params, traces, errtrace, crlb = fitter(torch.tensor(initial,device=dev), 
                                                  torch.tensor(I_smp_mu, device=dev), 
                                                  torch.tensor(I_smp_sig, device=dev),
                                                  torch.tensor(mod_batch_fl, device=dev),
                                                  param_limits=param_range,
                                                  ndims=ndims)
            
            results.append(params)
            crlbs.append(crlb)
            
            pb.update(len(bix))

    results = torch.cat(results).cpu().numpy()
    crlbs = torch.cat(crlbs).cpu().numpy()
    
    ds = ds.copy()
    ds.pos[:,:ndims] = results[:,:ndims] + ds.pos[:,:2]# + ds.data.roipos[:,[-1,-2]]
    ds.crlb.pos[:,:ndims] = crlbs[:,:ndims]
    ds.photons = results[:,ndims:ndims+2].sum(1)
    return ds
    
    
def merge_estimates(ds1: SFDataset, ds2: SFDataset, ndims:int =2):
    
    assert len(ds1) == len(ds2)
    
    combined = ds1[:]
    
    total_fi = ds1.crlb.pos[:,:ndims]**-2 + ds2.crlb.pos[:,:ndims]**-2
    combined.crlb.pos[:,:ndims] = total_fi ** -0.5
    
    combined.pos[:,:ndims] = (
        ds1.pos[:,:ndims] * ds1.crlb.pos[:,:ndims] ** -2 / total_fi + 
        ds2.pos[:,:ndims] * ds2.crlb.pos[:,:ndims] ** -2 / total_fi
    )
    
    return combined

def ndi_fit_and_filter(ds: SFDataset, mp: ModulationPattern, max_distance=0.2, **kwargs):
    ndi_ds = ndi_fit_dataset(ds, mp, **kwargs)
    dist = np.sqrt ( ( (ndi_ds.pos[:,:2] - ds.pos[:,:2])**2 ).sum(1) )
    mask = (dist<max_distance)
    return ndi_ds[mask]

def ndi_fit_and_merge(ds: SFDataset, mp: ModulationPattern, max_distance=0.2, ndims:int = 2, 
            device=None, ndi_with_bg = False):

    ndi_ds = ndi_fit_dataset(ds, mp, ndims, device, ndi_with_bg)
    combined = merge_estimates(ds, ndi_ds, ndims)
    dist = np.sqrt ( ( (ndi_ds.pos[:,:2] - ds.pos[:,:2])**2 ).sum(1) )
    mask = (dist<max_distance)
    ndi_ds = ndi_ds[mask] 
    combined = combined[mask]

    return combined, ndi_ds
    
if __name__ == '__main__':
    pixelsize = 100
    pf = np.array([[0,2,4],[1,3,5]])
    mod = angles_to_mod([200,200], pixelsize, [10,100], 0.9, pf)
    ds = SFDataset(10000, 2, [50,50], numPatterns=6)

    nframes = 600
    ds.frame = np.random.randint(0, nframes, size=len(ds))
    ds.pos = np.random.uniform([0,0], [ds.imgshape[1],ds.imgshape[0]], size=(len(ds),2))
    ds.photons = 1000
    ds['pixelsize'] = pixelsize

    mp = ModulationPattern(pf, mod, moving_window=True)

    mp_drift = np.random.normal(0, 0.002, size=(nframes,2))
    mp_drift = np.cumsum(mp_drift, axis=0)
    # smooth along first axis
    from scipy.ndimage import convolve 
    drift_smooth_window = 10
    mp_drift = convolve(mp_drift, np.ones((drift_smooth_window,1)), mode='nearest') #
    mp.add_drift(mp_drift)
    plt.figure()
    plt.plot(mp_drift)
    plt.title('Phase drift in px')
    plt.legend(['x', 'y'])

    ds = mp.modulate(ds)
    ds.ibg[:,:,0] += np.random.normal(0, 100, size=ds.ibg[:,:,0].shape)

    mp_est = estimate_angles([150,250], ds, pf, None, mp.moving_window)
    device = 'cuda:0'
    spots_per_bin = 100
    mp_est = estimate_phases(ds, mp_est, spots_per_bin, 
        accept_percentile=50,
        device=device, iterations=1, verbose=False)

    phase_error_rmsd_rad = mp_est.phase_error_rmsd(mp)
    print(f"RMSD of phase drift (1 it): {phase_error_rmsd_rad:.2f} rad, {np.rad2deg(phase_error_rmsd_rad):.2f} deg")

    mp_est = estimate_phases(ds, mp_est, spots_per_bin, 
        accept_percentile=80,
        device=device, iterations=5, verbose=False)

    phase_error_rmsd_rad = mp_est.phase_error_rmsd(mp)
    print(f"RMSD of phase drift (5 it): {phase_error_rmsd_rad:.2f} rad, {np.rad2deg(phase_error_rmsd_rad):.2f} deg")

    #ds.renderFigure(zoom=20)

    #drift_trace = mp_est.drift_trace_px()
    mp_est_eo = mp_est.const_phase_offsets()

    col = ['r', 'g', 'k']
    fig,ax=plt.subplots(2, 1,figsize=(10,8),sharex=True)
    mp_est.plot_phase_drift(ax=ax, colors=col, label='Est {0}', linestyle='--', lw=3)
    mp.plot_phase_drift(ax=ax, colors=col, label='GT {0}')

    phase_error_rmsd_rad = mp_est.phase_error_rmsd(mp)
    print(f"RMSD of phase drift: {phase_error_rmsd_rad:.2f} rad, {np.rad2deg(phase_error_rmsd_rad):.2f} deg")

    phase_error_rmsd_rad_co = mp_est_eo.phase_error_rmsd(mp)
    print(f"RMSD of phase drift (const offset): {phase_error_rmsd_rad_co:.2f} rad, {np.rad2deg(phase_error_rmsd_rad_co):.2f} deg")

    # Phase RMSD depends strongly on settings of loss_max and me_threshold
    mp_r = estimate_phase_drift(ds, mp_est, 1, device=device, max_step=1e2, loss_max=0.2, me_threshold=0.3)
    mp_r.plot_phase_drift(ax=ax, colors=col, label='Refined {0}', linestyle=':', lw=6)

    phase_error_rmsd_rad_r = mp_r.phase_error_rmsd(mp)
    print(f"RMSD of phase drift (refined): {phase_error_rmsd_rad_r:.2f} rad, {np.rad2deg(phase_error_rmsd_rad_r):.2f} deg")
